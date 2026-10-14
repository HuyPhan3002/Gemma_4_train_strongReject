"""
BART-based conditional embedding diffusion for watermark recovery.

Training:  X_0 = Embed(T_wm)  --noise-->  X_t  --decoder(h)-->  X_0_hat
Inference: Z_T ~ N(0,I)  --iterative DDIM-->  Z_0  --decode-->  T_wm_hat

Condition h = BARTEncoder(T_attack) via decoder cross-attention.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration


def make_alpha_bar_linear(
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Linear beta schedule; returns cumulative product of alphas."""
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def make_alpha_bar_cosine(
    num_steps: int,
    s: float = 0.008,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal); alpha_bar[-1] is near zero."""
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps, dtype=torch.float64, device=device)
    f = torch.cos((t / num_steps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    return alphas_cumprod[1:].float()


def make_alpha_bar(
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    schedule_type: str = "linear",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if schedule_type == "cosine":
        return make_alpha_bar_cosine(num_steps, device=device)
    if schedule_type != "linear":
        raise ValueError(f"Unknown schedule_type={schedule_type!r}; use 'linear' or 'cosine'.")
    return make_alpha_bar_linear(num_steps, beta_start, beta_end, device=device)


def print_alpha_bar_summary(
    alpha_bar: torch.Tensor,
    schedule_type: str,
    num_diffusion_steps: int,
) -> None:
    mid = num_diffusion_steps // 2
    print(f"schedule_type: {schedule_type}")
    print(f"alpha_bar[0]: {alpha_bar[0].item():.6f}")
    print(f"alpha_bar[mid={mid}]: {alpha_bar[mid].item():.6f}")
    print(f"alpha_bar[-1]: {alpha_bar[-1].item():.6f}")


# BART-base reference used to transfer noise_scale across embedding dims.
# With σ=0.06 and D=768, E[||σ ε||_2] ≈ 0.06 * sqrt(768) ≈ 1.66.
_BART_REF_NOISE_SCALE = 0.06
_BART_REF_D_MODEL = 768


def compute_embedding_scale(
    embed_weight: torch.Tensor,
    *,
    min_norm: float = 1e-3,
    nonzero_quantile: float = 0.25,
) -> torch.Tensor:
    """
    Robust L2 scale for mapping raw token embeddings into unit-median diffusion space.

    LLM tables (especially Gemma) contain many near-zero unused rows. A plain
    vocabulary-wide median is dominated by those rows, so content tokens stay at
    norm ≫ 1 after "normalization" and a BART-tuned noise_scale becomes ~100x too
    small (exactly the failure mode seen with Gemma 4 E4B).
    """
    norms = embed_weight.detach().float().norm(dim=-1)
    nonzero = norms > min_norm
    if bool(nonzero.any()):
        usable = norms[nonzero]
        thr = torch.quantile(usable, nonzero_quantile)
        usable = usable[usable >= thr]
        scale = usable.median() if usable.numel() > 0 else norms.median()
    else:
        scale = norms.median()
    return scale.to(dtype=torch.float32).clamp(min=1e-4)


def recommend_noise_scale(
    d_model: int,
    *,
    target_noise_l2: Optional[float] = None,
    bart_noise_scale: float = _BART_REF_NOISE_SCALE,
    bart_d_model: int = _BART_REF_D_MODEL,
) -> float:
    """
    Recommend noise_scale for unit-median-normalized embeddings.

    Matches BART's expected pure-noise L2 by default:
        σ * sqrt(D) ≈ bart_noise_scale * sqrt(bart_d_model)
    so Gemma-4 (D≈2560) gets σ ≈ 0.033 rather than the old 0.012 default.
    """
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    if target_noise_l2 is None:
        target_noise_l2 = bart_noise_scale * math.sqrt(bart_d_model)
    return float(target_noise_l2 / math.sqrt(float(d_model)))


def print_noise_scale_summary(
    *,
    noise_scale: float,
    d_model: int,
    embedding_scale: Optional[float] = None,
    typical_x0_norm: float = 1.0,
) -> None:
    """Log noise/signal geometry after embedding normalization."""
    expected_noise_l2 = noise_scale * math.sqrt(float(d_model))
    ratio = expected_noise_l2 / max(typical_x0_norm, 1e-8)
    print(
        f"noise_scale: {noise_scale:.6f} | d_model={d_model} | "
        f"E[||noise||_2]≈{expected_noise_l2:.4f} | "
        f"noise/x0≈{ratio:.4f} (target ~1 after unit-median norm)"
    )
    if embedding_scale is not None:
        print(f"embedding_scale (raw→diffusion): {embedding_scale:.6f}")


def build_valid_token_mask(
    embed_weight: torch.Tensor,
    tokenizer: Optional[Any] = None,
    *,
    min_norm: float = 1e-3,
    exclude_name_substrings: Tuple[str, ...] = ("unused",),
) -> torch.Tensor:
    """
    Boolean mask over vocab: True = allowed for CE / nearest-neighbor decode.

    Gemma reserves many ``<unused*>`` rows. Their embeddings form dense basins that
    nearest-neighbor decode snaps to when predictions are slightly off-manifold.
    Mask them (and near-zero rows) so recovery can only emit real content tokens.
    """
    norms = embed_weight.detach().float().norm(dim=-1)
    valid = norms > min_norm
    if tokenizer is None:
        return valid

    vocab = getattr(tokenizer, "get_vocab", lambda: None)()
    if isinstance(vocab, dict):
        for token, idx in vocab.items():
            name = str(token).lower()
            if any(s in name for s in exclude_name_substrings):
                if 0 <= int(idx) < valid.numel():
                    valid[int(idx)] = False
        return valid

    # Fallback: probe ids via convert_ids_to_tokens (slower, once at init).
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is None:
        return valid
    for idx in range(int(valid.numel())):
        tok = convert(idx)
        name = str(tok).lower() if tok is not None else ""
        if any(s in name for s in exclude_name_substrings):
            valid[idx] = False
    return valid


class BartEmbeddingDiffusion(nn.Module):
    """
    Conditional embedding diffusion in BART shared-embedding space.

    The decoder predicts clean embeddings X_0 from noisy embeddings X_t,
    conditioned on encoder hidden states h from the attacked text.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        num_diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        pad_token_id: int = 1,
        max_target_length: int = 256,
        target_length_margin: int = 20,
        emb_loss_weight: float = 1.0,
        ce_loss_weight: float = 0.5,
        cos_loss_weight: float = 0.1,
        ce_min_alpha_bar: float = 0.9,
        ce_max_alpha_bar: Optional[float] = None,
        pure_noise_train_frac: float = 0.0,
        denoiser_backbone: str = "encoder_cross",
        use_residual: bool = True,
        num_attn_heads: int = 8,
        noise_scale: float = 0.06,
        schedule_type: str = "linear",
        timestep_sampling: str = "low_bias",
    ):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(model_name)
        self.d_model = self.bart.config.d_model
        self.num_diffusion_steps = num_diffusion_steps
        self.pad_token_id = pad_token_id
        self.max_target_length = max_target_length
        self.target_length_margin = int(target_length_margin)
        self.emb_loss_weight = emb_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.cos_loss_weight = cos_loss_weight
        self.ce_min_alpha_bar = ce_min_alpha_bar
        self.ce_max_alpha_bar = ce_max_alpha_bar
        self.pure_noise_train_frac = pure_noise_train_frac
        self.denoiser_backbone = denoiser_backbone
        self.use_residual = use_residual
        self.noise_scale = noise_scale
        self.schedule_type = schedule_type
        self.timestep_sampling = timestep_sampling
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.time_embed = nn.Embedding(num_diffusion_steps, self.d_model)
        nn.init.zeros_(self.time_embed.weight)
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        if use_residual:
            nn.init.zeros_(self.output_proj.weight)
        else:
            nn.init.eye_(self.output_proj.weight)

        if denoiser_backbone == "encoder_cross":
            self.attack_cross_attn = nn.MultiheadAttention(
                self.d_model,
                num_attn_heads,
                batch_first=True,
            )
            self.denoise_norm = nn.LayerNorm(self.d_model)
        elif denoiser_backbone != "decoder":
            raise ValueError(
                f"Unknown denoiser_backbone={denoiser_backbone!r}; "
                "use 'encoder_cross' or 'decoder'."
            )

        alpha_bar = make_alpha_bar(
            num_diffusion_steps,
            beta_start,
            beta_end,
            schedule_type=schedule_type,
        )
        self.register_buffer("alpha_bar", alpha_bar, persistent=True)
        print_alpha_bar_summary(alpha_bar, schedule_type, num_diffusion_steps)

    @property
    def shared(self) -> nn.Embedding:
        return self.bart.model.shared

    @property
    def lm_head(self) -> nn.Module:
        return self.bart.lm_head

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate to the underlying BART model (required by HF Trainer)."""
        self.bart.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        """Delegate to the underlying BART model (required by HF Trainer)."""
        self.bart.gradient_checkpointing_disable()

    def encode_attack(
        self,
        attack_input_ids: torch.Tensor,
        attack_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs = self.bart.model.encoder(
            input_ids=attack_input_ids,
            attention_mask=attack_attention_mask,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.shared(input_ids)

    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        target_attention_mask: torch.Tensor,
        epsilon: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if epsilon is None:
            epsilon = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1).to(dtype=x0.dtype)
        scaled_eps = self.noise_scale * epsilon
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * scaled_eps

        pad_mask = target_attention_mask.unsqueeze(-1).float()
        x_t = pad_mask * x_t + (1.0 - pad_mask) * x0
        return x_t, epsilon

    def predict_x0(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        target_attention_mask: torch.Tensor,
        attack_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embed(t).unsqueeze(1)
        decoder_inputs = z_t + t_emb

        if self.denoiser_backbone == "decoder":
            decoder_outputs = self.bart.model.decoder(
                inputs_embeds=decoder_inputs,
                attention_mask=target_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attack_attention_mask,
                return_dict=True,
            )
            hidden = decoder_outputs.last_hidden_state
        else:
            key_padding_mask = ~attack_attention_mask.bool()
            cross, _ = self.attack_cross_attn(
                decoder_inputs,
                encoder_hidden_states,
                encoder_hidden_states,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            fused = self.denoise_norm(decoder_inputs + cross)
            encoder_outputs = self.bart.model.encoder(
                inputs_embeds=fused,
                attention_mask=target_attention_mask,
                return_dict=True,
            )
            hidden = encoder_outputs.last_hidden_state

        delta = self.output_proj(hidden)
        if self.use_residual:
            return z_t + delta
        return delta

    def logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(embeddings)
        bias = getattr(self.bart, "final_logits_bias", None)
        if bias is not None:
            logits = logits + bias
        return logits

    def match_reference_embedding_scale(
        self,
        recovered: torch.Tensor,
        reference: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position L2 norm match (used for CE so lm_head sees usable magnitudes)."""
        ref_norm = reference.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        rec_norm = recovered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scaled = recovered * (ref_norm / rec_norm)
        mask = attention_mask.unsqueeze(-1).float()
        return mask * scaled + (1.0 - mask) * recovered

    def embedding_mse_loss(
        self,
        x0: torch.Tensor,
        x0_recovered: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mse_per_token = F.mse_loss(x0_recovered, x0, reduction="none")
        mask = target_attention_mask.unsqueeze(-1).float()
        return (mse_per_token * mask).sum() / mask.sum().clamp(min=1.0)

    def embedding_cosine_loss(
        self,
        x0: torch.Tensor,
        x0_recovered: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        cos = F.cosine_similarity(x0_recovered, x0, dim=-1)
        loss_per_token = 1.0 - cos
        mask = target_attention_mask.float()
        return (loss_per_token * mask).sum() / mask.sum().clamp(min=1.0)

    def token_ce_loss(
        self,
        x0_recovered: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        t: torch.Tensor,
        x0_reference: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Token CE on low-noise (alpha_bar >= ce_min) and/or high-noise
        (alpha_bar <= ce_max) timesteps.
        """
        embeddings = x0_recovered
        if x0_reference is not None:
            embeddings = self.match_reference_embedding_scale(
                x0_recovered, x0_reference, target_attention_mask
            )

        logits = self.logits_from_embeddings(embeddings.float())
        per_token_ce = F.cross_entropy(
            logits.transpose(1, 2),
            target_input_ids,
            ignore_index=self.pad_token_id,
            reduction="none",
        )

        alpha_bar_t = self.alpha_bar[t]
        ce_gate = alpha_bar_t >= self.ce_min_alpha_bar
        if self.ce_max_alpha_bar is not None:
            ce_gate = ce_gate | (alpha_bar_t <= self.ce_max_alpha_bar)
        ce_gate = ce_gate.to(per_token_ce.dtype)
        token_weights = ce_gate.unsqueeze(1) * target_attention_mask.float()
        denom = token_weights.sum().clamp(min=1.0)
        loss_ce = (per_token_ce * token_weights).sum() / denom
        ce_frac = token_weights.sum() / target_attention_mask.float().sum().clamp(min=1.0)
        return loss_ce, ce_frac

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample diffusion timesteps according to self.timestep_sampling."""
        t_max = self.num_diffusion_steps - 1
        if self.timestep_sampling == "uniform":
            return torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)

        u = torch.rand(batch_size, device=device)
        t_low = (u * u * t_max).long()
        if self.timestep_sampling == "low_bias":
            return t_low

        # mixed: 50% low-bias, 50% uniform
        mask = torch.rand(batch_size, device=device) < 0.5
        t_uni = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        return torch.where(mask, t_low, t_uni)

    def resolve_infer_target_length(
        self,
        attack_attention_mask: torch.Tensor,
        max_target_length: int,
        wm_attention_mask: Optional[torch.Tensor] = None,
        infer_length_strategy: str = "attack_length",
        target_length_margin: Optional[int] = None,
    ) -> Tuple[int, torch.Tensor]:
        """
        Choose denoising width and per-example valid lengths for inference.

        Returns (infer_width, valid_lengths) where infer_width is the tensor
        sequence dimension and valid_lengths marks active positions.
        """
        margin = (
            self.target_length_margin
            if target_length_margin is None
            else int(target_length_margin)
        )
        attack_lengths = attack_attention_mask.sum(dim=1)
        budget_lengths = (attack_lengths + margin).clamp(min=1, max=max_target_length)

        if infer_length_strategy == "max_target_length":
            infer_width = max_target_length
            valid_lengths = budget_lengths
        elif (
            infer_length_strategy == "target_length_if_available"
            and wm_attention_mask is not None
        ):
            valid_lengths = wm_attention_mask.sum(dim=1).clamp(max=max_target_length)
            infer_width = int(valid_lengths.max().item())
        else:
            valid_lengths = budget_lengths
            infer_width = int(budget_lengths.max().item())

        infer_width = max(infer_width, 1)
        valid_lengths = valid_lengths.clamp(max=infer_width)
        return infer_width, valid_lengths

    def build_target_mask(
        self,
        batch_size: int,
        target_length: int,
        device: torch.device,
        valid_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = torch.zeros(batch_size, target_length, device=device, dtype=torch.long)
        if valid_lengths is None:
            mask.fill_(1)
        else:
            for idx, length in enumerate(valid_lengths.tolist()):
                mask[idx, : int(length)] = 1
        return mask

    def rescale_embeddings_for_decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Temporary inference patch: match per-position norm to typical token norms.
        """
        token_norms = self.shared.weight.norm(dim=-1)
        target_norm = token_norms.median().to(
            device=embeddings.device, dtype=embeddings.dtype
        )
        current_norm = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return embeddings * (target_norm / current_norm)

    def embeddings_to_token_ids(
        self,
        embeddings: torch.Tensor,
        method: str = "euclidean",
    ) -> torch.Tensor:
        weight = self.shared.weight
        if method == "cosine":
            emb_n = F.normalize(embeddings, dim=-1)
            weight_n = F.normalize(weight, dim=-1)
            return torch.matmul(emb_n, weight_n.T).argmax(dim=-1)

        emb_sq = (embeddings ** 2).sum(dim=-1, keepdim=True)
        weight_sq = (weight ** 2).sum(dim=-1)
        dots = torch.matmul(embeddings, weight.T)
        distances = emb_sq + weight_sq - 2.0 * dots
        return distances.argmin(dim=-1)

    def decode_embeddings(
        self,
        embeddings: torch.Tensor,
        use_decode_patch: bool = True,
        decode_method: str = "euclidean",
    ) -> torch.Tensor:
        if use_decode_patch:
            embeddings = self.rescale_embeddings_for_decode(embeddings)
            return self.embeddings_to_token_ids(embeddings, method=decode_method)
        return self.logits_from_embeddings(embeddings).argmax(dim=-1)

    def compute_losses(
        self,
        attack_input_ids: torch.Tensor,
        attack_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = attack_input_ids.size(0)
        device = attack_input_ids.device

        h = self.encode_attack(attack_input_ids, attack_attention_mask)
        x0 = self.embed_tokens(target_input_ids)
        t_max = self.num_diffusion_steps - 1

        if t is None:
            t = self.sample_timesteps(batch_size, device)

        pure_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if self.pure_noise_train_frac > 0.0 and self.training:
            pure_mask = torch.rand(batch_size, device=device) < self.pure_noise_train_frac
            if pure_mask.any():
                t = torch.where(
                    pure_mask,
                    torch.full_like(t, t_max),
                    t,
                )

        x_t, _ = self.add_noise(x0, t, target_attention_mask)
        if pure_mask.any():
            pure_noise = self.noise_scale * torch.randn_like(x0)
            pad_mask = target_attention_mask.unsqueeze(-1).float()
            pure_x_t = pad_mask * pure_noise + (1.0 - pad_mask) * x0
            x_t = torch.where(pure_mask.view(-1, 1, 1), pure_x_t, x_t)

        x0_recovered = self.predict_x0(
            z_t=x_t,
            t=t,
            encoder_hidden_states=h,
            target_attention_mask=target_attention_mask,
            attack_attention_mask=attack_attention_mask,
        )

        loss_emb = self.embedding_mse_loss(x0, x0_recovered, target_attention_mask)
        loss_ce, ce_frac = self.token_ce_loss(
            x0_recovered,
            target_input_ids,
            target_attention_mask,
            t=t,
            x0_reference=x0,
        )
        loss_cos = self.embedding_cosine_loss(x0, x0_recovered, target_attention_mask)
        loss = (
            self.emb_loss_weight * loss_emb
            + self.ce_loss_weight * loss_ce
            + self.cos_loss_weight * loss_cos
        )

        return {
            "loss": loss,
            "loss_emb": loss_emb,
            "loss_ce": loss_ce,
            "loss_cos": loss_cos,
            "ce_frac": ce_frac,
            "pure_noise_frac": pure_mask.float().mean(),
        }

    def forward(
        self,
        attack_input_ids: torch.Tensor,
        attack_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.compute_losses(
            attack_input_ids=attack_input_ids,
            attack_attention_mask=attack_attention_mask,
            target_input_ids=target_input_ids,
            target_attention_mask=target_attention_mask,
            t=t,
        )
        if return_dict:
            return losses
        return losses["loss"]

    @torch.no_grad()
    def denoise_from_z(
        self,
        z_t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        target_attention_mask: torch.Tensor,
        attack_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = z_t.size(0)
        device = z_t.device
        dtype = z_t.dtype

        for step in range(self.num_diffusion_steps - 1, -1, -1):
            t_tensor = torch.full(
                (batch_size,),
                fill_value=step,
                device=device,
                dtype=torch.long,
            )
            x0_recovered = self.predict_x0(
                z_t=z_t,
                t=t_tensor,
                encoder_hidden_states=encoder_hidden_states,
                target_attention_mask=target_attention_mask,
                attack_attention_mask=attack_attention_mask,
            )

            alpha_bar_t = self.alpha_bar[step].to(device=device, dtype=dtype)
            if step > 0:
                alpha_bar_prev = self.alpha_bar[step - 1].to(device=device, dtype=dtype)
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device, dtype=dtype)

            denom = torch.sqrt(1.0 - alpha_bar_t).clamp(min=1e-8) * self.noise_scale
            eps_hat = (z_t - torch.sqrt(alpha_bar_t) * x0_recovered) / denom
            z_t = (
                torch.sqrt(alpha_bar_prev) * x0_recovered
                + torch.sqrt(1.0 - alpha_bar_prev) * self.noise_scale * eps_hat
            )
        return z_t

    @torch.no_grad()
    def recover(
        self,
        attack_input_ids: torch.Tensor,
        attack_attention_mask: torch.Tensor,
        max_target_length: Optional[int] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        wm_attention_mask: Optional[torch.Tensor] = None,
        infer_length_strategy: str = "attack_length",
        use_decode_patch: bool = True,
        decode_method: str = "euclidean",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM-style reverse diffusion in embedding space.

        infer_length_strategy controls denoising width:
          attack_length — min(attack_len, max_target_length)
          max_target_length — always max_target_length
          target_length_if_available — wm length when wm_attention_mask is given
        """
        batch_size = attack_input_ids.size(0)
        device = attack_input_ids.device
        dtype = torch.float32

        cap = max_target_length or self.max_target_length
        h = self.encode_attack(attack_input_ids, attack_attention_mask)

        if target_attention_mask is None:
            infer_width, valid_lengths = self.resolve_infer_target_length(
                attack_attention_mask=attack_attention_mask,
                max_target_length=cap,
                wm_attention_mask=wm_attention_mask,
                infer_length_strategy=infer_length_strategy,
            )
            target_attention_mask = self.build_target_mask(
                batch_size=batch_size,
                target_length=infer_width,
                device=device,
                valid_lengths=valid_lengths,
            )
        else:
            infer_width = target_attention_mask.size(1)

        z_t = self.noise_scale * torch.randn(
            batch_size,
            infer_width,
            self.d_model,
            device=device,
            dtype=dtype,
        )
        z_t = self.denoise_from_z(
            z_t=z_t,
            encoder_hidden_states=h,
            target_attention_mask=target_attention_mask,
            attack_attention_mask=attack_attention_mask,
        )
        pred_ids = self.decode_embeddings(
            z_t,
            use_decode_patch=use_decode_patch,
            decode_method=decode_method,
        )
        return z_t, pred_ids
