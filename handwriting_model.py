#!/usr/bin/env python3
"""
Handyman Handwriting Synthesis Model v3 - corrected causal training.

Architecture:
    text -> embedding -> BiGRU -> per-character context
                                      |
                                      v
                         monotonic Gaussian attention
                                      |
    previous trajectory + global position + writer style
                                      |
                                      v
                                  LSTM
                                      |
                                      v
                         MDN(dx,dy) + EOS

The important causal rule is:

    input at time t   = trajectory[t]
    prediction at t   = trajectory[t+1]

Global position is also kept causal:

    use position BEFORE applying trajectory[t]
    predict the next movement/position
    then advance position using trajectory[t]

This matches autoregressive generation, where the model receives
the previous generated point and predicts the next point.
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

POSITION_SCALE = 100.0


# ============================================================
# MONOTONIC GAUSSIAN WINDOW
# ============================================================

class GaussianWindow(nn.Module):
    """Monotonic Gaussian attention over the text characters."""

    def __init__(
        self,
        hidden_dim: int,
        text_dim: int,
        num_components: int = 10,
    ):
        super().__init__()

        self.num_components = num_components

        self.to_params = nn.Linear(
            hidden_dim,
            3 * num_components,
        )

    def forward(
        self,
        h_t: torch.Tensor,
        text_context: torch.Tensor,
        text_mask: torch.Tensor,
        kappa_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # h_t: [B,H]
        params = self.to_params(h_t)

        alpha_hat, beta_hat, delta_hat = params.chunk(
            3,
            dim=-1,
        )

        # Positive and numerically bounded parameters.
        alpha = torch.exp(
            alpha_hat.clamp(-7.0, 7.0)
        )

        beta = torch.exp(
            beta_hat.clamp(-7.0, 7.0)
        )

        delta = torch.exp(
            delta_hat.clamp(-7.0, 7.0)
        )

        # Keep attention movement stable.
        delta = delta.clamp(max=2.0)

        # Monotonic attention position.
        kappa = kappa_prev + delta

        text_len = text_context.shape[1]

        u = torch.arange(
            text_len,
            device=h_t.device,
            dtype=h_t.dtype,
        ).view(1, 1, text_len)

        phi_components = (
            alpha.unsqueeze(-1)
            * torch.exp(
                -beta.unsqueeze(-1)
                * (kappa.unsqueeze(-1) - u) ** 2
            )
        )

        # [B,L]
        phi = phi_components.sum(dim=1)

        phi = phi * text_mask.to(phi.dtype)

        # Normalize so attended context is numerically stable.
        phi_sum = phi.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        weights = phi / phi_sum

        # [B,text_dim]
        w_t = torch.bmm(
            weights.unsqueeze(1),
            text_context,
        ).squeeze(1)

        return w_t, kappa, phi


# ============================================================
# MDN OUTPUT
# ============================================================

class MDNOutput(nn.Module):
    """
    Mixture density output for (dx,dy) plus EOS.

    Each Gaussian component has:
        pi
        mu_x
        mu_y
        sigma_x
        sigma_y
        rho

    There is one additional EOS logit.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_mixtures: int = 20,
    ):
        super().__init__()

        self.num_mixtures = num_mixtures

        self.to_params = nn.Linear(
            hidden_dim,
            num_mixtures * 6 + 1,
        )

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

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
            "pi": F.softmax(
                pi_logits,
                dim=-1,
            ),
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": torch.exp(
                sigma_x_hat.clamp(-7.0, 7.0)
            ).clamp_min(1e-4),
            "sigma_y": torch.exp(
                sigma_y_hat.clamp(-7.0, 7.0)
            ).clamp_min(1e-4),
            "rho": torch.tanh(
                rho_hat
            ).clamp(-0.999, 0.999),
            "eos_logit": eos_logit,
        }


# ============================================================
# MDN LOSS
# ============================================================

def mdn_loss(
    params: Dict[str, torch.Tensor],
    dx_target: torch.Tensor,
    dy_target: torch.Tensor,
    eos_target: torch.Tensor,
    mask: torch.Tensor,
    eos_weight: float = 1.0,
) -> torch.Tensor:
    """
    Bivariate Gaussian-mixture negative log likelihood
    plus EOS binary cross entropy.
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

    one_minus_rho2 = (
        1.0 - rho.pow(2)
    ).clamp_min(1e-6)

    z = (
        z_x.pow(2)
        + z_y.pow(2)
        - 2.0 * rho * z_x * z_y
    )

    log_norm = (
        -math.log(2.0 * math.pi)
        - torch.log(sigma_x)
        - torch.log(sigma_y)
        - 0.5 * torch.log(one_minus_rho2)
    )

    log_gaussian = (
        log_norm
        - z / (2.0 * one_minus_rho2)
    )

    log_pi = torch.log(
        pi.clamp_min(1e-8)
    )

    log_prob = torch.logsumexp(
        log_pi + log_gaussian,
        dim=-1,
    )

    stroke_nll = -log_prob

    eos_bce = F.binary_cross_entropy_with_logits(
        params["eos_logit"],
        eos_target,
        reduction="none",
    )

    total = (
        stroke_nll
        + eos_weight * eos_bce
    ) * mask

    return total.sum() / mask.sum().clamp_min(1.0)


# ============================================================
# POSITION LOSS
# ============================================================

def position_loss(
    predicted_position: torch.Tensor,
    target_position: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Smooth L1 loss for normalized global position.
    """

    loss = F.smooth_l1_loss(
        predicted_position,
        target_position,
        reduction="none",
    )

    # [B,T,2] -> [B,T]
    loss = loss.mean(dim=-1)

    loss = loss * mask

    return loss.sum() / mask.sum().clamp_min(1.0)


# ============================================================
# MDN SAMPLING
# ============================================================

@torch.no_grad()
def sample_mdn(
    params: Dict[str, torch.Tensor],
    temperature: float = 0.65,
    eos_threshold: float = 0.5,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Sample one dx, dy and EOS value."""

    temperature = max(
        float(temperature),
        1e-3,
    )

    pi = params["pi"]

    sigma_x = params["sigma_x"] * temperature
    sigma_y = params["sigma_y"] * temperature

    logits = (
        torch.log(
            pi.clamp_min(1e-8)
        )
        / temperature
    )

    pi_temp = F.softmax(
        logits,
        dim=-1,
    )

    component = torch.multinomial(
        pi_temp,
        num_samples=1,
    ).squeeze(-1)

    idx = component.unsqueeze(-1)

    mu_x = params["mu_x"].gather(
        -1,
        idx,
    ).squeeze(-1)

    mu_y = params["mu_y"].gather(
        -1,
        idx,
    ).squeeze(-1)

    sigma_x_selected = sigma_x.gather(
        -1,
        idx,
    ).squeeze(-1)

    sigma_y_selected = sigma_y.gather(
        -1,
        idx,
    ).squeeze(-1)

    rho = params["rho"].gather(
        -1,
        idx,
    ).squeeze(-1)

    z1 = torch.randn_like(mu_x)
    z2 = torch.randn_like(mu_x)

    dx = (
        mu_x
        + sigma_x_selected * z1
    )

    dy = (
        mu_y
        + sigma_y_selected
        * (
            rho * z1
            + torch.sqrt(
                (1.0 - rho.pow(2))
                .clamp_min(1e-6)
            )
            * z2
        )
    )

    eos_probability = torch.sigmoid(
        params["eos_logit"]
    )

    eos = (
        eos_probability >= eos_threshold
    ).float()

    return dx, dy, eos


# ============================================================
# MODEL
# ============================================================

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
        position_scale: float = POSITION_SCALE,
    ):
        super().__init__()

        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_window_components = num_window_components
        self.num_mixtures = num_mixtures
        self.position_scale = position_scale

        # ----------------------------------------------------
        # TEXT ENCODER
        # ----------------------------------------------------

        self.text_embedding = nn.Embedding(
            vocab_size,
            text_embedding_dim,
            padding_idx=0,
        )

        self.text_encoder = nn.GRU(
            input_size=text_embedding_dim,
            hidden_size=text_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        text_context_dim = (
            text_hidden_dim * 2
        )

        # ----------------------------------------------------
        # WRITER STYLE
        # ----------------------------------------------------

        self.writer_embedding = nn.Embedding(
            num_writers,
            writer_embedding_dim,
        )

        # ----------------------------------------------------
        # GENERATOR INPUT
        # ----------------------------------------------------

        # Previous trajectory point:
        #   dx, dy, eos
        #
        # plus attended text
        # plus writer style
        # plus current global position:
        #   x, y
        lstm_input_dim = (
            trajectory_input_dim
            + text_context_dim
            + writer_embedding_dim
            + 2
        )

        self.decoder_input_norm = nn.LayerNorm(
            lstm_input_dim
        )

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=rnn_hidden_dim,
            batch_first=True,
        )

        # ----------------------------------------------------
        # MONOTONIC ATTENTION
        # ----------------------------------------------------

        self.window = GaussianWindow(
            hidden_dim=rnn_hidden_dim,
            text_dim=text_context_dim,
            num_components=num_window_components,
        )

        # ----------------------------------------------------
        # OUTPUT REPRESENTATION
        # ----------------------------------------------------

        # h_t + attended text + current position + writer
        output_dim = (
            rnn_hidden_dim
            + text_context_dim
            + 2
            + writer_embedding_dim
        )

        self.output_norm = nn.LayerNorm(
            output_dim
        )

        self.dropout = nn.Dropout(dropout)

        # Auxiliary global-position prediction.
        self.position_head = nn.Sequential(
            nn.Linear(
                output_dim,
                256,
            ),
            nn.Tanh(),
            nn.Linear(
                256,
                2,
            ),
        )

        # ----------------------------------------------------
        # MDN
        # ----------------------------------------------------

        self.mdn = MDNOutput(
            hidden_dim=output_dim,
            num_mixtures=num_mixtures,
        )

    # ========================================================
    # TEXT ENCODER
    # ========================================================

    def encode_text(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:

        embedded = self.text_embedding(
            text_ids
        )

        encoded, _ = self.text_encoder(
            embedded
        )

        return encoded.masked_fill(
            ~text_mask.unsqueeze(-1),
            0.0,
        )

    # ========================================================
    # CAUSAL FORWARD / TEACHER FORCING
    # ========================================================

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        trajectory: torch.Tensor,
        writer_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Causal teacher forcing.

        trajectory: [B,T,3]

        Input:
            trajectory[:,0:-1]

        Target:
            trajectory[:,1:]

        Therefore the model never receives the point it is
        simultaneously being asked to predict.
        """

        batch_size, total_steps, _ = trajectory.shape
        device = trajectory.device

        if total_steps < 2:
            raise ValueError(
                "Trajectory must contain at least 2 points."
            )

        text_context = self.encode_text(
            text_ids,
            text_mask,
        )

        writer_vec = self.writer_embedding(
            writer_ids
        )

        # Inputs are previous points.
        inputs = trajectory[:, :-1, :]
        steps = inputs.shape[1]

        # Absolute position of every real trajectory point.
        full_position = (
            torch.cumsum(
                trajectory[..., 0:2],
                dim=1,
            )
            / self.position_scale
        )

        # Prediction at input t corresponds to real point t+1.
        target_position = full_position[:, 1:, :]

        # ----------------------------------------------------
        # INITIAL STATES
        # ----------------------------------------------------

        h = torch.zeros(
            1,
            batch_size,
            self.rnn_hidden_dim,
            device=device,
        )

        c = torch.zeros_like(h)

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

        # Position BEFORE the current input movement.
        position = torch.zeros(
            batch_size,
            2,
            device=device,
        )

        mdn_inputs = []
        predicted_positions = []

        # ----------------------------------------------------
        # CAUSAL LOOP
        # ----------------------------------------------------

        for t in range(steps):

            # This is the known previous point.
            x_t = inputs[:, t, :]

            # IMPORTANT:
            # position is still BEFORE x_t is applied.
            decoder_input = torch.cat(
                [
                    x_t,
                    w_t,
                    writer_vec,
                    position,
                ],
                dim=-1,
            )

            decoder_input = (
                self.decoder_input_norm(
                    decoder_input
                )
                .unsqueeze(1)
            )

            lstm_out, (h, c) = self.lstm(
                decoder_input,
                (h, c),
            )

            h_t = lstm_out[:, 0, :]

            # Update text attention after obtaining
            # the recurrent state.
            w_t, kappa, _ = self.window(
                h_t,
                text_context,
                text_mask,
                kappa,
            )

            # Still pre-movement position.
            combined = torch.cat(
                [
                    h_t,
                    w_t,
                    position,
                    writer_vec,
                ],
                dim=-1,
            )

            combined = self.output_norm(
                combined
            )

            combined = self.dropout(
                combined
            )

            mdn_inputs.append(combined)

            predicted_positions.append(
                self.position_head(
                    combined
                )
            )

            # NOW advance position using the known
            # previous input movement.
            position = (
                position
                + x_t[:, 0:2]
                / self.position_scale
            )

        mdn_input = torch.stack(
            mdn_inputs,
            dim=1,
        )

        predicted_position = torch.stack(
            predicted_positions,
            dim=1,
        )

        return {
            "params": self.mdn(
                mdn_input
            ),
            "position": predicted_position,
            "target_position": target_position,
        }

    # ========================================================
    # LOSS
    # ========================================================

    def calculate_loss(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        trajectory: torch.Tensor,
        writer_ids: torch.Tensor,
        trajectory_mask: torch.Tensor,
        eos_weight: float = 1.0,
        position_weight: float = 0.10,
    ) -> Dict[str, torch.Tensor]:

        outputs = self.forward(
            text_ids=text_ids,
            text_mask=text_mask,
            trajectory=trajectory,
            writer_ids=writer_ids,
        )

        params = outputs["params"]

        targets = trajectory[:, 1:, :]
        mask = trajectory_mask[:, 1:].float()

        loss_mdn = mdn_loss(
            params=params,
            dx_target=targets[..., 0],
            dy_target=targets[..., 1],
            eos_target=targets[..., 2],
            mask=mask,
            eos_weight=eos_weight,
        )

        loss_position = position_loss(
            predicted_position=outputs["position"],
            target_position=outputs["target_position"],
            mask=mask,
        )

        total = (
            loss_mdn
            + position_weight * loss_position
        )

        return {
            "loss": total,
            "mdn_loss": loss_mdn,
            "position_loss": loss_position,
        }

    # ========================================================
    # AUTOREGRESSIVE GENERATION
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        writer_ids: torch.Tensor,
        max_steps: int = 1200,
        temperature: float = 0.65,
        eos_threshold: float = 0.5,
        end_patience: int = 25,
        max_position: float = 1000.0,
        min_steps: int = 30,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Returns:
            [B,T,3] = [dx,dy,eos]

        EOS means end-of-stroke, not end-of-sentence.
        Sentence completion is controlled by attention.
        """

        self.eval()

        device = text_ids.device
        batch_size = text_ids.shape[0]

        text_context = self.encode_text(
            text_ids,
            text_mask,
        )

        writer_vec = self.writer_embedding(
            writer_ids
        )

        h = torch.zeros(
            1,
            batch_size,
            self.rnn_hidden_dim,
            device=device,
        )

        c = torch.zeros_like(h)

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

        position = torch.zeros(
            batch_size,
            2,
            device=device,
        )

        # Previous point.
        x_t = torch.zeros(
            batch_size,
            3,
            device=device,
        )

        outputs = []

        text_len = text_mask.sum(
            dim=1
        ).long()

        end_counter = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )

        for _ in range(max_steps):

            # position is BEFORE x_t is applied.
            decoder_input = torch.cat(
                [
                    x_t,
                    w_t,
                    writer_vec,
                    position,
                ],
                dim=-1,
            )

            decoder_input = (
                self.decoder_input_norm(
                    decoder_input
                )
                .unsqueeze(1)
            )

            lstm_out, (h, c) = self.lstm(
                decoder_input,
                (h, c),
            )

            h_t = lstm_out[:, 0, :]

            w_t, kappa, phi = self.window(
                h_t,
                text_context,
                text_mask,
                kappa,
            )

            combined = torch.cat(
                [
                    h_t,
                    w_t,
                    position,
                    writer_vec,
                ],
                dim=-1,
            )

            combined = self.output_norm(
                combined
            )

            params = self.mdn(
                combined
            )

            dx, dy, eos = sample_mdn(
                params,
                temperature=temperature,
                eos_threshold=eos_threshold,
            )

            # Safety clamp prevents one pathological MDN
            # sample from immediately destroying the layout.
            dx = dx.clamp(-30.0, 30.0)
            dy = dy.clamp(-30.0, 30.0)

            # If a batch item is already finished, output
            # zeros for subsequent positions.
            active = ~finished

            dx = torch.where(
                active,
                dx,
                torch.zeros_like(dx),
            )

            dy = torch.where(
                active,
                dy,
                torch.zeros_like(dy),
            )

            eos = torch.where(
                active,
                eos,
                torch.ones_like(eos),
            )

            x_t = torch.stack(
                [
                    dx,
                    dy,
                    eos,
                ],
                dim=-1,
            )

            outputs.append(
                x_t.detach()
            )

            # Advance global position AFTER prediction.
            position = (
                position
                + x_t[:, 0:2]
                / self.position_scale
            )

            # Hard safety limit.
            position = position.clamp(
                -max_position / self.position_scale,
                max_position / self.position_scale,
            )

            # Attention peak indicates which text character
            # the model is currently reading.
            peak = torch.argmax(
                phi,
                dim=-1,
            )

            at_end = (
                peak
                >= (text_len - 1).clamp_min(0)
            )

            end_counter = torch.where(
                at_end,
                end_counter + 1,
                torch.zeros_like(end_counter),
            )

            enough_steps = (
                len(outputs) >= min_steps
            )

            if enough_steps:
                newly_finished = (
                    end_counter >= end_patience
                )
                finished = (
                    finished
                    | newly_finished
                )

            if bool(finished.all()):
                break

        if not outputs:
            return torch.zeros(
                batch_size,
                0,
                3,
                device=device,
            )

        return torch.stack(
            outputs,
            dim=1,
        )


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# DATASET INFO
# ============================================================

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

    return (
        len(vocabulary["tokens"]),
        len(writers["writers"]),
    )


# ============================================================
# SELF TEST
# ============================================================

def main() -> None:
    print("=" * 75)
    print("HANDWRITING MODEL v3 SELF TEST")
    print("CAUSAL TEACHER FORCING + POSITION OBJECTIVE")
    print("=" * 75)

    vocab_size, num_writers = load_dataset_info()

    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of writers: {num_writers}")

    model = HandwritingModel(
        vocab_size=vocab_size,
        num_writers=num_writers,
    )

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

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
        torch.rand(
            batch_size,
            traj_len,
        )
        > 0.9
    ).float()

    trajectory_mask = torch.ones(
        batch_size,
        traj_len,
    )

    writer_ids = torch.randint(
        0,
        num_writers,
        (batch_size,),
    )

    outputs = model(
        text_ids,
        text_mask,
        trajectory,
        writer_ids,
    )

    expected_steps = traj_len - 1

    print(
        f"MDN sequence shape: "
        f"{outputs['params']['mu_x'].shape}"
    )

    print(
        f"Position prediction shape: "
        f"{outputs['position'].shape}"
    )

    assert (
        outputs["position"].shape
        == (batch_size, expected_steps, 2)
    ), (
        "Position prediction must have "
        "T-1 timesteps."
    )

    assert (
        outputs["params"]["mu_x"].shape
        == (
            batch_size,
            expected_steps,
            model.num_mixtures,
        )
    )

    losses = model.calculate_loss(
        text_ids=text_ids,
        text_mask=text_mask,
        trajectory=trajectory,
        writer_ids=writer_ids,
        trajectory_mask=trajectory_mask,
    )

    print(
        f"MDN loss:       "
        f"{losses['mdn_loss'].item():.6f}"
    )

    print(
        f"Position loss:  "
        f"{losses['position_loss'].item():.6f}"
    )

    print(
        f"Total loss:     "
        f"{losses['loss'].item():.6f}"
    )

    assert torch.isfinite(
        losses["loss"]
    )

    generated = model.generate(
        text_ids[:1],
        text_mask[:1],
        writer_ids[:1],
        max_steps=80,
        min_steps=10,
    )

    print(
        f"Generated trajectory shape: "
        f"{tuple(generated.shape)}"
    )

    assert (
        generated.ndim == 3
        and generated.shape[0] == 1
        and generated.shape[2] == 3
    )

    print()
    print("ALL V3 MODEL CHECKS PASSED")


if __name__ == "__main__":
    main()
