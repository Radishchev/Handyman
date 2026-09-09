#!/usr/bin/env python3
"""
Handyman handwriting synthesis model v2.

Architecture:
    text -> embedding -> BiGRU -> per-character context
    trajectory + writer style -> LSTM1
    LSTM1 -> monotonic Gaussian-window attention
    trajectory + LSTM1 + attended text -> LSTM2
    LSTM1 + LSTM2 + attended text -> MDN
    MDN -> dx, dy distribution + EOS

Compatible with the existing Handyman training data:
    trajectory shape [B, T, 3] = [dx, dy, eos]
    text_ids, text_mask, writer_ids

This replaces the old deterministic dx/dy regression model.
"""

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path("/home/aurlin/Projects/Handyman")
DATA_DIR = PROJECT_ROOT / "TRAINING_DATA"


class GaussianWindow(nn.Module):
    """Monotonic Gaussian attention over the character sequence."""

    def __init__(self, hidden_dim: int, text_dim: int, num_components: int = 10):
        super().__init__()
        self.num_components = num_components
        self.to_params = nn.Linear(hidden_dim, 3 * num_components)

    def forward(
        self,
        h_t: torch.Tensor,
        text_context: torch.Tensor,
        text_mask: torch.Tensor,
        kappa_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # h_t: [B,H]
        params = self.to_params(h_t)
        alpha_hat, beta_hat, kappa_hat = params.chunk(3, dim=-1)

        # Clamp before exp to avoid exploding attention parameters early in training.
        alpha = torch.exp(alpha_hat.clamp(-7.0, 7.0))
        beta = torch.exp(beta_hat.clamp(-7.0, 7.0))
        delta = torch.exp(kappa_hat.clamp(-7.0, 7.0))

        # Monotonic text position.
        kappa = kappa_prev + delta

        length = text_context.shape[1]
        u = torch.arange(
            length, device=h_t.device, dtype=h_t.dtype
        ).view(1, 1, length)

        phi = alpha.unsqueeze(-1) * torch.exp(
            -beta.unsqueeze(-1) * (kappa.unsqueeze(-1) - u) ** 2
        )
        phi = phi.sum(dim=1)  # [B,L]
        phi = phi * text_mask.to(dtype=phi.dtype)

        # Normalize for a stable context vector. Keep phi itself for visualization
        # and for generation stopping decisions.
        phi_sum = phi.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = phi / phi_sum

        w_t = torch.bmm(
            weights.unsqueeze(1), text_context
        ).squeeze(1)

        return w_t, kappa, phi


class MDNOutput(nn.Module):
    """
    Mixture density output for (dx, dy) plus EOS.

    Each Gaussian component contains:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho
    """

    def __init__(self, hidden_dim: int, num_mixtures: int = 20):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.to_params = nn.Linear(hidden_dim, num_mixtures * 6 + 1)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.to_params(hidden)
        m = self.num_mixtures

        pi_logits = out[..., 0:m]
        mu_x = out[..., m:2 * m]
        mu_y = out[..., 2 * m:3 * m]
        sigma_x_hat = out[..., 3 * m:4 * m]
        sigma_y_hat = out[..., 4 * m:5 * m]
        rho_hat = out[..., 5 * m:6 * m]
        eos_logit = out[..., 6 * m]

        return {
            "pi": F.softmax(pi_logits, dim=-1),
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": torch.exp(sigma_x_hat.clamp(-7.0, 7.0)).clamp_min(1e-4),
            "sigma_y": torch.exp(sigma_y_hat.clamp(-7.0, 7.0)).clamp_min(1e-4),
            "rho": torch.tanh(rho_hat).clamp(-0.999, 0.999),
            "eos_logit": eos_logit,
        }


def mdn_loss(
    params: Dict[str, torch.Tensor],
    dx_target: torch.Tensor,
    dy_target: torch.Tensor,
    eos_target: torch.Tensor,
    mask: torch.Tensor,
    eos_weight: float = 1.0,
) -> torch.Tensor:
    """
    MDN negative log-likelihood + EOS BCE.

    Shapes:
        targets/mask: [B,T]
        mixture params: [B,T,M]
    """
    pi = params["pi"]
    mu_x = params["mu_x"]
    mu_y = params["mu_y"]
    sigma_x = params["sigma_x"]
    sigma_y = params["sigma_y"]
    rho = params["rho"]

    dx = dx_target.unsqueeze(-1)
    dy = dy_target.unsqueeze(-1)

    z_x = (dx - mu_x) / sigma_x
    z_y = (dy - mu_y) / sigma_y

    one_minus_rho2 = (1.0 - rho.pow(2)).clamp_min(1e-6)

    z = z_x.pow(2) + z_y.pow(2) - 2.0 * rho * z_x * z_y
    log_norm = (
        -math.log(2.0 * math.pi)
        - torch.log(sigma_x)
        - torch.log(sigma_y)
        - 0.5 * torch.log(one_minus_rho2)
    )
    log_gaussian = log_norm - z / (2.0 * one_minus_rho2)

    log_pi = torch.log(pi.clamp_min(1e-8))
    log_prob = torch.logsumexp(log_pi + log_gaussian, dim=-1)
    stroke_nll = -log_prob

    eos_bce = F.binary_cross_entropy_with_logits(
        params["eos_logit"],
        eos_target,
        reduction="none",
    )

    total = (stroke_nll + eos_weight * eos_bce) * mask
    return total.sum() / mask.sum().clamp_min(1.0)


def sample_mdn(
    params: Dict[str, torch.Tensor],
    temperature: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample one [dx,dy,eos] from a single-step MDN.

    Lower temperature -> tighter/more repeatable handwriting.
    Higher temperature -> more variation.
    """
    temperature = max(float(temperature), 1e-3)

    pi = params["pi"]
    mu_x = params["mu_x"]
    mu_y = params["mu_y"]
    sigma_x = params["sigma_x"] * temperature
    sigma_y = params["sigma_y"] * temperature
    rho = params["rho"]

    # Temperature the mixture weights as well.
    logits = torch.log(pi.clamp_min(1e-8)) / temperature
    pi_temp = F.softmax(logits, dim=-1)

    component = torch.multinomial(pi_temp, num_samples=1).squeeze(-1)
    idx = component.unsqueeze(-1)

    mx = mu_x.gather(-1, idx).squeeze(-1)
    my = mu_y.gather(-1, idx).squeeze(-1)
    sx = sigma_x.gather(-1, idx).squeeze(-1)
    sy = sigma_y.gather(-1, idx).squeeze(-1)
    r = rho.gather(-1, idx).squeeze(-1)

    z1 = torch.randn_like(mx)
    z2 = torch.randn_like(mx)

    dx = mx + sx * z1
    dy = my + sy * (
        r * z1 + torch.sqrt((1.0 - r.pow(2)).clamp_min(1e-6)) * z2
    )

    eos_prob = torch.sigmoid(params["eos_logit"])
    eos = (torch.rand_like(eos_prob) < eos_prob).float()

    return dx, dy, eos


class HandwritingModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_writers: int,
        text_embedding_dim: int = 64,
        text_hidden_dim: int = 128,
        writer_embedding_dim: int = 32,
        trajectory_input_dim: int = 3,
        rnn_hidden_dim: int = 400,
        num_window_components: int = 10,
        num_mixtures: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_window_components = num_window_components
        self.num_mixtures = num_mixtures

        self.text_embedding = nn.Embedding(
            vocab_size,
            text_embedding_dim,
            padding_idx=0,
        )

        self.text_encoder = nn.GRU(
            text_embedding_dim,
            text_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        text_context_dim = text_hidden_dim * 2

        self.writer_embedding = nn.Embedding(
            num_writers,
            writer_embedding_dim,
        )

        lstm1_input_dim = (
            trajectory_input_dim
            + text_context_dim
            + writer_embedding_dim
        )
        self.lstm1 = nn.LSTM(
            lstm1_input_dim,
            rnn_hidden_dim,
            batch_first=True,
        )

        self.window = GaussianWindow(
            rnn_hidden_dim,
            text_context_dim,
            num_window_components,
        )

        lstm2_input_dim = (
            trajectory_input_dim
            + rnn_hidden_dim
            + text_context_dim
        )
        self.lstm2 = nn.LSTM(
            lstm2_input_dim,
            rnn_hidden_dim,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        mdn_input_dim = (
            rnn_hidden_dim * 2
            + text_context_dim
        )
        self.mdn = MDNOutput(
            mdn_input_dim,
            num_mixtures=num_mixtures,
        )

    def encode_text(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.text_embedding(text_ids)
        encoded, _ = self.text_encoder(embedded)
        return encoded.masked_fill(
            ~text_mask.unsqueeze(-1),
            0.0,
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        trajectory: torch.Tensor,
        writer_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher forcing.

        trajectory [B,T,3].
        Predictions correspond to trajectory[:,1:].
        """
        batch_size, total_steps, _ = trajectory.shape
        device = trajectory.device

        if total_steps < 2:
            raise ValueError("Trajectory must contain at least 2 points.")

        text_context = self.encode_text(text_ids, text_mask)
        writer_vec = self.writer_embedding(writer_ids)

        h1 = torch.zeros(
            1, batch_size, self.rnn_hidden_dim,
            device=device,
        )
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(h1)

        kappa = torch.zeros(
            batch_size,
            self.num_window_components,
            device=device,
        )
        w_t = torch.zeros(
            batch_size,
            text_context.shape[-1],
            device=device,
        )

        mdn_inputs = []

        for t in range(total_steps - 1):
            x_t = trajectory[:, t, :]

            lstm1_in = torch.cat(
                [x_t, w_t, writer_vec],
                dim=-1,
            ).unsqueeze(1)

            out1, (h1, c1) = self.lstm1(
                lstm1_in,
                (h1, c1),
            )
            h1_t = out1.squeeze(1)

            w_t, kappa, _ = self.window(
                h1_t,
                text_context,
                text_mask,
                kappa,
            )

            lstm2_in = torch.cat(
                [x_t, h1_t, w_t],
                dim=-1,
            ).unsqueeze(1)

            out2, (h2, c2) = self.lstm2(
                lstm2_in,
                (h2, c2),
            )
            h2_t = out2.squeeze(1)

            mdn_inputs.append(
                torch.cat([h1_t, h2_t, w_t], dim=-1)
            )

        mdn_inputs = torch.stack(mdn_inputs, dim=1)
        mdn_inputs = self.dropout(mdn_inputs)

        return self.mdn(mdn_inputs)

    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        writer_id: torch.Tensor,
        max_len: int = 700,
        temperature: float = 0.8,
        min_len: int = 40,
        end_patience: int = 12,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Returns [T,3] = [dx,dy,eos].

        EOS means end-of-stroke, not end-of-sentence, so generation never
        stops simply because eos=1. Instead, the monotonic attention window
        must reach the final text character for several consecutive steps.
        """
        device = text_ids.device

        text_context = self.encode_text(text_ids, text_mask)
        writer_vec = self.writer_embedding(writer_id)

        h1 = torch.zeros(
            1, 1, self.rnn_hidden_dim,
            device=device,
        )
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(h1)

        kappa = torch.zeros(
            1,
            self.num_window_components,
            device=device,
        )
        w_t = torch.zeros(
            1,
            text_context.shape[-1],
            device=device,
        )

        x_t = torch.zeros(1, 3, device=device)
        points = []

        text_len = int(text_mask.sum().item())
        end_counter = 0

        for _ in range(max_len):
            lstm1_in = torch.cat(
                [x_t, w_t, writer_vec],
                dim=-1,
            ).unsqueeze(1)

            out1, (h1, c1) = self.lstm1(
                lstm1_in,
                (h1, c1),
            )
            h1_t = out1.squeeze(1)

            w_t, kappa, phi = self.window(
                h1_t,
                text_context,
                text_mask,
                kappa,
            )

            lstm2_in = torch.cat(
                [x_t, h1_t, w_t],
                dim=-1,
            ).unsqueeze(1)

            out2, (h2, c2) = self.lstm2(
                lstm2_in,
                (h2, c2),
            )
            h2_t = out2.squeeze(1)

            mdn_in = torch.cat(
                [h1_t, h2_t, w_t],
                dim=-1,
            )
            params = self.mdn(mdn_in)

            dx, dy, eos = sample_mdn(
                params,
                temperature=temperature,
            )

            x_t = torch.stack(
                [dx, dy, eos],
                dim=-1,
            )
            points.append(x_t.squeeze(0).cpu())

            peak = int(torch.argmax(phi[0]).item())

            if peak >= max(text_len - 1, 0):
                end_counter += 1
            else:
                end_counter = 0

            if len(points) >= min_len and end_counter >= end_patience:
                break

        if not points:
            return torch.zeros(1, 3)

        return torch.stack(points, dim=0)


def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def load_dataset_info() -> Tuple[int, int]:
    with open(
        DATA_DIR / "vocabulary.json",
        "r",
        encoding="utf-8",
    ) as f:
        vocabulary = json.load(f)

    with open(
        DATA_DIR / "writers.json",
        "r",
        encoding="utf-8",
    ) as f:
        writers = json.load(f)

    return len(vocabulary["tokens"]), len(writers["writers"])


def main() -> None:
    print("=" * 75)
    print("HANDWRITING MODEL v2 SELF TEST")
    print("Gaussian window + MDN")
    print("=" * 75)

    vocab_size, num_writers = load_dataset_info()

    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of writers: {num_writers}")

    model = HandwritingModel(
        vocab_size=vocab_size,
        num_writers=num_writers,
    )

    print(f"Trainable parameters: {count_parameters(model):,}")

    batch_size = 2
    text_len = 15
    traj_len = 60

    text_ids = torch.randint(
        1,
        vocab_size,
        (batch_size, text_len),
    )
    text_mask = torch.ones(
        batch_size,
        text_len,
        dtype=torch.bool,
    )
    trajectory = torch.randn(
        batch_size,
        traj_len,
        3,
    )
    trajectory[:, :, 2] = (
        torch.rand(batch_size, traj_len) > 0.9
    ).float()
    writer_ids = torch.randint(
        0,
        num_writers,
        (batch_size,),
    )

    params = model(
        text_ids,
        text_mask,
        trajectory,
        writer_ids,
    )

    loss = mdn_loss(
        params,
        trajectory[:, 1:, 0],
        trajectory[:, 1:, 1],
        trajectory[:, 1:, 2],
        torch.ones_like(trajectory[:, 1:, 2]),
    )

    print(f"Forward pass loss: {loss.item():.4f}")
    assert torch.isfinite(loss)

    generated = model.generate(
        text_ids[:1],
        text_mask[:1],
        writer_ids[:1],
        max_len=80,
    )

    print(f"Generated trajectory: {tuple(generated.shape)}")
    print("ALL MODEL CHECKS PASSED")


if __name__ == "__main__":
    main()
