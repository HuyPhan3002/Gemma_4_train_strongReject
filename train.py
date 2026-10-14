"""
Train a Gemma 4 (E4B) conditional embedding diffusion model for watermark recovery.

This mirrors the BART pipeline (model.py / train.py) as closely as possible, but
replaces the encoder-decoder + cross-attention denoiser with a decoder-only
Gemma backbone conditioned via PREFIX CONCATENATION instead of cross-attention:

    full_sequence = [ Embed(T_attack) ; noisy_target_embeddings ]
    hidden        = GemmaModel(inputs_embeds=full_sequence, attention_mask=...)
    predicted_x0  = output_proj(hidden[:, attack_len:, :])   # target positions only

Diffusion schedule, noise addition, DDIM update, and emb+CE+cos+norm loss.
CE uses temperature-scaled cosine similarity against the normalized embedding table
(not lm_head, which expects hidden states). Decoding uses nearest-neighbor in
embedding space.

LoRA is enabled by default so only adapters + diffusion heads are trained.

Example:
    python train.py \\
        --output_dir ./gemma4_embed_diffusion \\
        --model_name google/gemma-4-E4B \\
        --num_diffusion_steps 1000 --bf16 --gradient_checkpointing

Example (2 GPUs):
    torchrun --standalone --nproc_per_node=2 train_llama2.py \\
        --output_dir ./gemma4_embed_diffusion --bf16

Resume from a Trainer checkpoint (optimizer / step / RNG restored):
    python train.py \\
        --output_dir ./gemma4_embed_diffusion \\
        --resume_from_checkpoint ./gemma4_embed_diffusion/checkpoint-1500 \\
        --bf16 --gradient_checkpointing
"""

from __future__ import annotations

import argparse
import json
import os

os.environ["HF_TOKEN"]= ""
os.environ.pop("SSL_CERT_FILE", None)  
os.environ.pop("REQUESTS_CA_BUNDLE", None) 
os.environ.pop("CURL_CA_BUNDLE", None)  

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from strong_reject.evaluate import evaluate
from utils import get_response, extract_scores

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError as _peft_err:  # pragma: no cover
    LoraConfig = None  # type: ignore[misc, assignment]
    PeftModel = None  # type: ignore[misc, assignment]
    get_peft_model = None  # type: ignore[misc, assignment]
    _PEFT_IMPORT_ERROR = _peft_err
else:
    _PEFT_IMPORT_ERROR = None

from dataset import BartDiffusionCollator, WatermarkRecoveryDataset
from model import (
    build_valid_token_mask,
    compute_embedding_scale,
    make_alpha_bar,
    print_alpha_bar_summary,
    print_noise_scale_summary,
    recommend_noise_scale,
)


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_NAME = "google/gemma-4-E2B"
_DEFAULT_TRAIN_JSON = str(_SCRIPT_DIR / "train_dataset/training_prompts_train.jsonl")
_DEFAULT_EVAL_JSON = str(_SCRIPT_DIR / "train_dataset/training_prompts_eval.jsonl")
# Gemma 4 ships multimodal towers (vision/audio) that use Gemma4ClippableLinear.
# Plain LoRA target lists walk those towers and crash; scope to language_model only.
_DEFAULT_LORA_TARGET_MODULES = (
    r".*\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)

EVALUATOR = "strongreject_finetuned"


def _allow_trusted_local_torch_load() -> None:
    """
    Newer transformers refuse torch.load unless torch>=2.6 (CVE-2025-32434).
    Resume from our own local checkpoints still needs optimizer.pt / scheduler.pt /
    rng_state_*.pth, so disable that gate for this process only.
    """
    try:
        import transformers.trainer as trainer_mod
        import transformers.utils.import_utils as import_utils
    except Exception:
        return

    def _noop_check_torch_load_is_safe() -> None:
        return None

    import_utils.check_torch_load_is_safe = _noop_check_torch_load_is_safe
    if hasattr(trainer_mod, "check_torch_load_is_safe"):
        trainer_mod.check_torch_load_is_safe = _noop_check_torch_load_is_safe


# ======================================================================================
# Model
# ======================================================================================
def _resolve_lora_target_modules(
    model: nn.Module,
    target_modules: Optional[str],
) -> Union[str, List[str]]:
    """
    Pick LoRA targets that are safe for Gemma 4 (language tower only).

    If the user passes an explicit regex/list string, honor it. Otherwise prefer
    language_model-scoped regex when that submodule exists; fall back to common
    causal-LM projection names for plain Gemma/Llama backbones.
    """
    if target_modules:
        return target_modules
    module_names = [name for name, _ in model.named_modules()]
    if any(".language_model." in name for name in module_names):
        return _DEFAULT_LORA_TARGET_MODULES
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def _get_causal_lm_backbone(causal_lm: nn.Module) -> nn.Module:
    """Return the transformer stack used for inputs_embeds forward passes."""
    root = causal_lm
    if hasattr(root, "get_base_model"):
        try:
            root = root.get_base_model()
        except Exception:
            pass
    if hasattr(root, "language_model") and root.language_model is not None:
        return root.language_model
    if hasattr(root, "model") and hasattr(root.model, "language_model"):
        return root.model.language_model
    if hasattr(root, "model"):
        return root.model
    return root


class GemmaEmbeddingDiffusion(nn.Module):
    """
    Conditional embedding diffusion in (normalized) Gemma embedding space.

    Decoder-only conditioning: the attacked text embeddings are prepended as a
    prefix to the noisy target embeddings, and a single causal forward pass
    denoises the target positions. There is NO encoder and NO cross-attention.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        num_diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        pad_token_id: int = 0,
        max_target_length: int = 256,
        target_length_margin: int = 20,
        emb_loss_weight: float = 1.0,
        ce_loss_weight: float = 2.0,
        cos_loss_weight: float = 0.1,
        norm_loss_weight: float = 0.1,
        sec_loss_weight: float = 0.1,
        ce_min_alpha_bar: float = 0.9,
        ce_max_alpha_bar: Optional[float] = 0.05,
        pure_noise_train_frac: float = 0.25,
        use_residual: bool = True,
        residual_mode: str = "gated",
        noise_scale: Optional[float] = None,
        schedule_type: str = "linear",
        timestep_sampling: str = "uniform",
        ce_use_cosine: bool = True,
        ce_temperature: float = 0.05,
        ce_all_timesteps: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        diagnostics_interval: int = 500,
        disable_train_bucket_metrics: bool = True,
    ):
        super().__init__()
        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        self.llama = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.base_model_name = model_name
        self.use_lora = bool(use_lora)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = lora_target_modules

        if self.use_lora:
            self._apply_lora()

        cfg = getattr(self.llama, "config", None)
        hidden = getattr(cfg, "hidden_size", None)
        if hidden is None and cfg is not None:
            text_cfg = getattr(cfg, "text_config", None)
            hidden = getattr(text_cfg, "hidden_size", None)
        if hidden is None:
            hidden = self.embed_matrix.embedding_dim
        self.d_model = int(hidden)
        self.num_diffusion_steps = num_diffusion_steps
        self.pad_token_id = pad_token_id
        self.max_target_length = max_target_length
        self.target_length_margin = int(target_length_margin)
        self.emb_loss_weight = emb_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.cos_loss_weight = cos_loss_weight
        self.norm_loss_weight = norm_loss_weight
        self.sec_loss_weight = sec_loss_weight
        self.ce_min_alpha_bar = ce_min_alpha_bar
        self.ce_max_alpha_bar = ce_max_alpha_bar
        self.pure_noise_train_frac = pure_noise_train_frac
        # Decoder-only backbone: conditioning is prefix concatenation, not cross-attn.
        self.denoiser_backbone = "gemma_prefix"
        if not use_residual:
            residual_mode = "none"
        self.residual_mode = residual_mode
        self.use_residual = residual_mode != "none"
        self.schedule_type = schedule_type
        self.timestep_sampling = timestep_sampling
        self.ce_use_cosine = ce_use_cosine
        self.ce_temperature = ce_temperature
        self.ce_all_timesteps = ce_all_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.diagnostics_interval = int(diagnostics_interval)
        self.disable_train_bucket_metrics = bool(disable_train_bucket_metrics)

        self.time_embed = nn.Embedding(num_diffusion_steps, self.d_model)
        nn.init.zeros_(self.time_embed.weight)
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        if self.residual_mode in ("full", "gated"):
            nn.init.zeros_(self.output_proj.weight)
        else:
            nn.init.eye_(self.output_proj.weight)

        alpha_bar = make_alpha_bar(
            num_diffusion_steps, beta_start, beta_end, schedule_type=schedule_type
        )
        self.register_buffer("alpha_bar", alpha_bar, persistent=True)
        print_alpha_bar_summary(alpha_bar, schedule_type, num_diffusion_steps)

        # Unit-median normalize using content-token norms (ignore near-zero unused rows).
        with torch.no_grad():
            scale = compute_embedding_scale(self.embed_matrix.weight)
        self.register_buffer("embedding_scale", scale, persistent=True)

        # Match BART pure-noise L2 in the normalized space unless the user overrides.
        if noise_scale is None:
            noise_scale = recommend_noise_scale(self.d_model)
        self.noise_scale = float(noise_scale)
        print_noise_scale_summary(
            noise_scale=self.noise_scale,
            d_model=self.d_model,
            embedding_scale=float(self.embedding_scale),
            typical_x0_norm=1.0,
        )

        # Exclude near-zero / <unused*> rows from CE and NN decode by default.
        with torch.no_grad():
            valid = build_valid_token_mask(self.embed_matrix.weight)
        self.register_buffer("valid_token_mask", valid, persistent=True)
        self.mask_unused_tokens = True
        n_blocked = int((~valid).sum().item())
        print(
            f"valid_token_mask: blocked {n_blocked}/{valid.numel()} tokens "
            f"(near-zero rows; refine with configure_token_mask(tokenizer))"
        )

    def _apply_lora(self) -> None:
        if get_peft_model is None or LoraConfig is None:
            raise ImportError(
                "peft is required for --use_lora. Install with `pip install peft`.\n"
                f"Original import error: {_PEFT_IMPORT_ERROR}"
            )
        targets = _resolve_lora_target_modules(self.llama, self.lora_target_modules)
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        self.llama = get_peft_model(self.llama, lora_config)
        self.lora_target_modules = (
            targets if isinstance(targets, str) else ",".join(targets)
        )
        if hasattr(self.llama, "print_trainable_parameters"):
            self.llama.print_trainable_parameters()

    # -- backbone accessors --------------------------------------------------------
    @property
    def backbone(self):
        """Bare transformer stack (no lm_head), used to get hidden states."""
        return _get_causal_lm_backbone(self.llama)

    @property
    def embed_matrix(self) -> nn.Embedding:
        return self.llama.get_input_embeddings()

    @property
    def lm_head(self) -> nn.Module:
        return self.llama.get_output_embeddings()

    @property
    def backbone_dtype(self) -> torch.dtype:
        return self.embed_matrix.weight.dtype

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llama.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.llama.gradient_checkpointing_disable()

    # -- embedding helpers ---------------------------------------------------------
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token ids -> normalized diffusion-space embeddings (float32)."""
        emb = self.embed_matrix(input_ids)
        scale = self.embedding_scale.to(device=emb.device, dtype=torch.float32)
        return emb.float() / scale

    def unscale_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings * self.embedding_scale

    def encode_attack(
        self,
        attack_input_ids: torch.Tensor,
        attack_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prefix condition = normalized embeddings of the attacked text.

        Unlike BART/T5 there is no encoder pass here; the conditioning is fed as a
        prefix into the same self-attention stack inside predict_x0.
        """
        return self.embed_tokens(attack_input_ids)

    # -- forward diffusion ---------------------------------------------------------
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

    def _build_per_layer_inputs(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        prefix_input_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Gemma 4 Per-Layer Embeddings (PLE).

        Calling the text model with only continuous ``inputs_embeds`` triggers an
        O(B·S·V·D) reverse embedding lookup (hundreds of GiB). Pass explicit
        ``per_layer_inputs`` instead:

        - prefix tokens: real PLE from ``embed_tokens_per_layer`` when ids are known
        - noisy target positions: zeros (context projection only)
        """
        backbone = self.backbone
        ple_dim = getattr(backbone, "hidden_size_per_layer_input", None)
        if not ple_dim:
            cfg = getattr(backbone, "config", None)
            ple_dim = getattr(cfg, "hidden_size_per_layer_input", 0) if cfg is not None else 0
        if not ple_dim:
            return None

        n_layers = int(backbone.config.num_hidden_layers)
        per_layer = torch.zeros(
            batch_size,
            seq_len,
            n_layers,
            int(ple_dim),
            device=device,
            dtype=dtype,
        )
        if prefix_input_ids is None or not hasattr(backbone, "embed_tokens_per_layer"):
            return per_layer

        prefix_len = prefix_input_ids.size(1)
        prefix_ple = backbone.embed_tokens_per_layer(prefix_input_ids).reshape(
            batch_size, prefix_len, n_layers, int(ple_dim)
        )
        per_layer[:, :prefix_len] = prefix_ple.to(dtype=dtype)
        return per_layer

    # -- denoiser (prefix-concat, decoder-only) ------------------------------------
    def predict_x0(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        target_attention_mask: torch.Tensor,
        attack_attention_mask: torch.Tensor,
        attack_input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        encoder_hidden_states holds the attack PREFIX embeddings (B, La, D).

        Sequence fed to the backbone = [ attack_prefix ; (z_t + t_emb) ].
        Only the target slice of the hidden states is projected to predict x0.
        """
        t_emb = self.time_embed(t).unsqueeze(1).to(dtype=z_t.dtype)
        target_inputs = z_t + t_emb

        prefix = encoder_hidden_states.to(dtype=z_t.dtype)
        prefix_len = prefix.size(1)

        full_embeds = torch.cat([prefix, target_inputs], dim=1).to(dtype=self.backbone_dtype)
        full_mask = torch.cat([attack_attention_mask, target_attention_mask], dim=1)

        forward_kwargs: Dict[str, Any] = {
            "inputs_embeds": full_embeds,
            "attention_mask": full_mask,
            "use_cache": False,
            "return_dict": True,
        }
        per_layer_inputs = self._build_per_layer_inputs(
            batch_size=full_embeds.size(0),
            seq_len=full_embeds.size(1),
            device=full_embeds.device,
            dtype=full_embeds.dtype,
            prefix_input_ids=attack_input_ids,
        )
        if per_layer_inputs is not None:
            forward_kwargs["per_layer_inputs"] = per_layer_inputs

        outputs = self.backbone(**forward_kwargs)
        hidden = outputs.last_hidden_state[:, prefix_len:, :].to(dtype=z_t.dtype)

        delta = self.output_proj(hidden)
        if self.residual_mode == "full":
            return z_t + delta
        if self.residual_mode == "gated":
            gate = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1).to(
                device=z_t.device,
                dtype=z_t.dtype,
            )
            return gate * z_t + delta
        return delta

    # -- decoding / logits ---------------------------------------------------------
    def configure_token_mask(self, tokenizer) -> None:
        """
        Refine valid_token_mask using tokenizer names (blocks ``<unused*>`` etc.).

        Call after loading the tokenizer (train / sanity / inference).
        Kept for NN decode; CE never masks a gold target class (see token_ce_loss).
        """
        self._decode_tokenizer = tokenizer
        with torch.no_grad():
            valid = build_valid_token_mask(self.embed_matrix.weight, tokenizer=tokenizer)
        if self.valid_token_mask.shape != valid.shape:
            self.register_buffer("valid_token_mask", valid, persistent=True)
        else:
            self.valid_token_mask.copy_(valid.to(device=self.valid_token_mask.device))
        n_blocked = int((~self.valid_token_mask).sum().item())
        print(
            f"[info] token mask: blocked {n_blocked}/{self.valid_token_mask.numel()} "
            f"unused/near-zero vocab rows (decode always; CE unmasks gold targets)"
        )

    def _apply_valid_token_mask_to_logits(
        self,
        logits: torch.Tensor,
        *,
        unmask_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mask unused vocab rows for decoding / CE.

        If ``unmask_token_ids`` is set (gold CE labels), those vocab ids are NOT
        masked so the true class logit is never -inf.
        """
        if not getattr(self, "mask_unused_tokens", True):
            return logits
        mask = getattr(self, "valid_token_mask", None)
        if mask is None:
            return logits
        blocked = ~mask.to(device=logits.device)
        if unmask_token_ids is not None and unmask_token_ids.numel() > 0:
            blocked = blocked.clone()
            uniq = torch.unique(unmask_token_ids.reshape(-1))
            uniq = uniq[(uniq >= 0) & (uniq < blocked.numel())]
            if self.pad_token_id is not None:
                uniq = uniq[uniq != int(self.pad_token_id)]
            if uniq.numel() > 0:
                blocked[uniq] = False
        return logits.masked_fill(blocked, torch.finfo(logits.dtype).min)

    def _apply_valid_token_mask_to_distances(self, distances: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "mask_unused_tokens", True):
            return distances
        mask = getattr(self, "valid_token_mask", None)
        if mask is None:
            return distances
        blocked = ~mask.to(device=distances.device)
        return distances.masked_fill(blocked, torch.finfo(distances.dtype).max)

    def logits_from_embeddings(
        self,
        embeddings: torch.Tensor,
        *,
        use_cosine: Optional[bool] = None,
        unmask_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_cosine is None:
            use_cosine = self.ce_use_cosine
        if use_cosine:
            weight = self.embed_matrix.weight / self.embedding_scale
            # Avoid NaN from normalize(0): zero rows stay zero after normalize in
            # some builds, but clamp eps keeps logits finite.
            emb_n = F.normalize(embeddings.float(), dim=-1, eps=1e-8)
            weight_n = F.normalize(weight.float(), dim=-1, eps=1e-8)
            logits = torch.matmul(emb_n, weight_n.T) / self.ce_temperature
        else:
            head_dtype = self.lm_head.weight.dtype
            unscaled = self.unscale_embeddings(embeddings).to(dtype=head_dtype)
            logits = self.lm_head(unscaled)
        return self._apply_valid_token_mask_to_logits(
            logits, unmask_token_ids=unmask_token_ids
        )

    def _log_invalid_ce_targets(
        self,
        target_input_ids: torch.Tensor,
        invalid_target_mask: torch.Tensor,
        *,
        max_examples: int = 5,
    ) -> None:
        """Print a few gold tokens that fall in the unused/blocked vocab mask."""
        if not bool(invalid_target_mask.any()):
            return
        n_logged = int(getattr(self, "_invalid_ce_log_count", 0))
        if n_logged >= 20:
            return
        ids = target_input_ids[invalid_target_mask].detach().cpu().tolist()
        uniq: List[int] = []
        for tid in ids:
            if tid not in uniq:
                uniq.append(int(tid))
            if len(uniq) >= max_examples:
                break
        tok = getattr(self, "_decode_tokenizer", None)
        examples: List[str] = []
        for tid in uniq:
            if tok is not None:
                try:
                    piece = tok.decode([tid], skip_special_tokens=False)
                except Exception:
                    piece = tok.convert_ids_to_tokens(tid)
                examples.append(f"{tid}:{piece!r}")
            else:
                examples.append(str(tid))
        print(
            f"[ce] invalid gold targets (blocked by valid_token_mask): "
            f"count={int(invalid_target_mask.sum().item())} "
            f"examples=[{', '.join(examples)}]",
            flush=True,
        )
        self._invalid_ce_log_count = n_logged + 1

    def match_reference_embedding_scale(
        self,
        recovered: torch.Tensor,
        reference: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        ref_norm = reference.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        rec_norm = recovered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scaled = recovered * (ref_norm / rec_norm)
        mask = attention_mask.unsqueeze(-1).float()
        return mask * scaled + (1.0 - mask) * recovered

    # -- losses --------------------------------------------------------------------
    @torch.no_grad()
    def security_score(
        self,
        x0_recovered: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode recovered embeddings, generate responses, and score with StrongREJECT."""
        tokenizer = getattr(self, "_decode_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "security_score requires configure_token_mask(tokenizer) before use."
            )

        token_ids = self.decode_embeddings(
            x0_recovered.detach(),
            use_decode_patch=True,
            decode_method="euclidean",
        ).detach().cpu()

        active_mask = target_attention_mask.detach().cpu().bool()
        scores: List[float] = []

        for ids, mask in zip(token_ids, active_mask):
            # Exclude padding before converting IDs to text.
            prompt = tokenizer.decode(
                ids[mask].tolist(),
                skip_special_tokens=True,
            )

            response = get_response(prompt)

            results = evaluate(
                forbidden_prompt=prompt,
                response=response,
                evaluators=[EVALUATOR],
            )

            scores.append(extract_scores(results, key="score").item())

        mean_score = 1 - (sum(scores) / len(scores)) if scores else 0.0

        # Keeps the score compatible with the training device/dtype.
        return x0_recovered.new_tensor(mean_score)
        
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

    def embedding_norm_loss(
        self,
        x0: torch.Tensor,
        x0_recovered: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize norm drift between predicted and reference token embeddings."""
        ref_norm = x0.norm(dim=-1)
        rec_norm = x0_recovered.norm(dim=-1)
        per_token = (rec_norm - ref_norm) ** 2
        mask = target_attention_mask.float()
        return (per_token * mask).sum() / mask.sum().clamp(min=1.0)

    def token_ce_loss(
        self,
        x0_recovered: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        t: torch.Tensor,
        x0_reference: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute token CE. Returns (loss_ce, ce_frac, logits, ce_stats).

        Unused/blocked vocab rows stay masked for decoding, but:
        - gold target ids in this batch are unmasked so the true class is never -inf
        - positions whose gold id is still blocked are excluded from CE (weight=0)
        """
        device = target_input_ids.device
        active = target_attention_mask.bool()
        n_active = active.float().sum().clamp(min=1.0)

        valid_mask = getattr(self, "valid_token_mask", None)
        if valid_mask is None:
            bad_token_mask = torch.zeros(
                int(self.embed_matrix.weight.size(0)),
                device=device,
                dtype=torch.bool,
            )
        else:
            bad_token_mask = ~valid_mask.to(device=device)

        # Gold labels that fall in the unused/blocked set.
        clamped_ids = target_input_ids.clamp(min=0, max=max(bad_token_mask.numel() - 1, 0))
        invalid_target_mask = active & bad_token_mask[clamped_ids]
        invalid_target_count = invalid_target_mask.float().sum()
        invalid_target_fraction = invalid_target_count / n_active
        if bool(invalid_target_mask.any()):
            self._log_invalid_ce_targets(target_input_ids, invalid_target_mask)

        embeddings = x0_recovered
        if x0_reference is not None and not self.ce_use_cosine:
            embeddings = self.match_reference_embedding_scale(
                x0_recovered, x0_reference, target_attention_mask
            )

        # Unmask gold target vocab ids so CE never sees true-class logit = -inf.
        gold_for_unmask = target_input_ids[active]
        logits = self.logits_from_embeddings(
            embeddings, unmask_token_ids=gold_for_unmask
        )

        nonfinite_logit = ~torch.isfinite(logits)
        nonfinite_logit_fraction = nonfinite_logit.float().mean()
        if bool(nonfinite_logit.any()):
            logits = torch.nan_to_num(
                logits,
                nan=0.0,
                posinf=50.0,
                neginf=-50.0,
            )

        per_token_ce = F.cross_entropy(
            logits.transpose(1, 2),
            target_input_ids,
            ignore_index=self.pad_token_id,
            reduction="none",
        )

        nonfinite_ce = ~torch.isfinite(per_token_ce)
        # Only count active (non-pad) positions toward the fraction.
        nonfinite_ce_fraction = (nonfinite_ce & active).float().sum() / n_active
        if bool(nonfinite_ce.any()):
            # Fall through to nan_to_num / clamp below.
            pass

        # Exclude gold-blocked positions from CE (do not supervise -inf / unused labels).
        if self.ce_all_timesteps:
            token_weights = target_attention_mask.float()
        else:
            alpha_bar_t = self.alpha_bar[t]
            ce_gate = alpha_bar_t >= self.ce_min_alpha_bar
            if self.ce_max_alpha_bar is not None:
                ce_gate = ce_gate | (alpha_bar_t <= self.ce_max_alpha_bar)
            ce_gate = ce_gate.to(per_token_ce.dtype)
            token_weights = ce_gate.unsqueeze(1) * target_attention_mask.float()
        token_weights = token_weights * (~invalid_target_mask).float()

        # Overflow before safety clamp (nonfinite or extreme CE).
        with torch.no_grad():
            raw_overflow = nonfinite_ce | (torch.isfinite(per_token_ce) & (per_token_ce > 50.0))
            ce_overflow_count = (
                raw_overflow & active & (~invalid_target_mask)
            ).float().sum()

        # Safety clamp only after invalid-target handling.
        per_token_ce = torch.nan_to_num(per_token_ce, nan=0.0, posinf=50.0, neginf=0.0)
        per_token_ce = per_token_ce.clamp(max=50.0)

        denom = token_weights.sum().clamp(min=1.0)
        loss_ce = (per_token_ce * token_weights).sum() / denom
        ce_frac = token_weights.sum() / n_active

        ce_stats: Dict[str, torch.Tensor] = {
            "invalid_target_count": invalid_target_count.detach(),
            "invalid_target_fraction": invalid_target_fraction.detach(),
            "nonfinite_logit_fraction": nonfinite_logit_fraction.detach(),
            "nonfinite_ce_fraction": nonfinite_ce_fraction.detach(),
            "ce_overflow_count": ce_overflow_count.detach(),
        }
        return loss_ce, ce_frac, logits, ce_stats

    # -- timestep sampling ---------------------------------------------------------
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        t_max = self.num_diffusion_steps - 1
        if self.timestep_sampling == "uniform":
            return torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)

        u = torch.rand(batch_size, device=device)
        t_low = (u * u * t_max).long()
        if self.timestep_sampling == "low_bias":
            return t_low

        mask = torch.rand(batch_size, device=device) < 0.5
        t_uni = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        return torch.where(mask, t_low, t_uni)

    # -- inference-length helpers --------------------------------------------------
    def resolve_infer_target_length(
        self,
        attack_attention_mask: torch.Tensor,
        max_target_length: int,
        wm_attention_mask: Optional[torch.Tensor] = None,
        infer_length_strategy: str = "attack_length",
        target_length_margin: Optional[int] = None,
    ) -> Tuple[int, torch.Tensor]:
        margin = (
            self.target_length_margin
            if target_length_margin is None
            else int(target_length_margin)
        )
        attack_lengths = attack_attention_mask.sum(dim=1)
        # Default: room for wm slightly longer than attack.
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
        """Normalized diffusion embeddings -> raw space with median token norm."""
        raw = self.unscale_embeddings(embeddings)
        target_norm = self.embedding_scale.to(dtype=raw.dtype, device=raw.device)
        current_norm = raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return raw * (target_norm / current_norm)

    def embeddings_to_token_ids_raw(
        self,
        raw_embeddings: torch.Tensor,
        method: str = "euclidean",
    ) -> torch.Tensor:
        weight = self.embed_matrix.weight.to(dtype=raw_embeddings.dtype)
        if method == "cosine":
            emb_n = F.normalize(raw_embeddings, dim=-1)
            weight_n = F.normalize(weight, dim=-1)
            scores = torch.matmul(emb_n, weight_n.T)
            scores = self._apply_valid_token_mask_to_logits(scores)
            return scores.argmax(dim=-1)

        emb_sq = (raw_embeddings ** 2).sum(dim=-1, keepdim=True)
        weight_sq = (weight ** 2).sum(dim=-1)
        dots = torch.matmul(raw_embeddings, weight.T)
        distances = emb_sq + weight_sq - 2.0 * dots
        distances = self._apply_valid_token_mask_to_distances(distances)
        return distances.argmin(dim=-1)

    def embeddings_to_token_ids(
        self,
        embeddings: torch.Tensor,
        method: str = "euclidean",
    ) -> torch.Tensor:
        raw_embeddings = self.unscale_embeddings(embeddings)
        return self.embeddings_to_token_ids_raw(raw_embeddings, method=method)

    def decode_embeddings(
        self,
        embeddings: torch.Tensor,
        use_decode_patch: bool = True,
        decode_method: str = "euclidean",
    ) -> torch.Tensor:
        if use_decode_patch:
            raw_embeddings = self.rescale_embeddings_for_decode(embeddings)
            return self.embeddings_to_token_ids_raw(raw_embeddings, method=decode_method)
        return self.logits_from_embeddings(embeddings).argmax(dim=-1)

    def _masked_mean_norm(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        norms = embeddings.norm(dim=-1, keepdim=True)
        return (norms * mask).sum() / mask.sum().clamp(min=1.0)

    def _token_accuracy(
        self,
        predicted_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = attention_mask.bool()
        if not valid.any():
            return torch.tensor(0.0, device=target_ids.device)
        return (predicted_ids[valid] == target_ids[valid]).float().mean()

    def _compute_bucket_metrics(
        self,
        *,
        pure_mask: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        x0_recovered: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        compute_nn: bool = True,
    ) -> Dict[str, torch.Tensor]:
        alpha = self.alpha_bar[t]
        bucket_defs = {
            "low": alpha >= 0.9,
            "mid": (alpha > 0.05) & (alpha < 0.9),
            "high": (alpha <= 0.05) & ~pure_mask,
            "pure": pure_mask,
        }
        metrics: Dict[str, torch.Tensor] = {}
        # Reuse CE logits when provided — never recompute the L×V matmul.
        if logits is None:
            logits = self.logits_from_embeddings(x0_recovered.detach())
        else:
            logits = logits.detach()
        ce_pred = logits.argmax(dim=-1)
        nn_pred = (
            self.embeddings_to_token_ids(x0_recovered.detach()) if compute_nn else None
        )

        for bucket_name, sample_mask in bucket_defs.items():
            if not sample_mask.any():
                continue
            token_mask = sample_mask.float().unsqueeze(1) * target_attention_mask.float()
            denom = token_mask.sum().clamp(min=1.0)

            mse = F.mse_loss(x0_recovered, x0, reduction="none").mean(dim=-1)
            metrics[f"bucket_{bucket_name}_mse"] = (mse * token_mask).sum() / denom

            cos = F.cosine_similarity(x0_recovered, x0, dim=-1)
            metrics[f"bucket_{bucket_name}_cos"] = (cos * token_mask).sum() / denom

            x0_hat_norm = x0_recovered.norm(dim=-1)
            metrics[f"bucket_{bucket_name}_x0_hat_norm"] = (x0_hat_norm * token_mask).sum() / denom

            ce_per = F.cross_entropy(
                logits.transpose(1, 2),
                target_input_ids,
                ignore_index=self.pad_token_id,
                reduction="none",
            )
            metrics[f"bucket_{bucket_name}_ce"] = (ce_per * token_mask).sum() / denom

            ce_acc = (ce_pred == target_input_ids).float() * token_mask
            metrics[f"bucket_{bucket_name}_ce_acc"] = ce_acc.sum() / denom

            if nn_pred is not None:
                nn_acc = (nn_pred == target_input_ids).float() * token_mask
                metrics[f"bucket_{bucket_name}_nn_acc"] = nn_acc.sum() / denom

        return metrics

    # -- training objective --------------------------------------------------------
    def compute_losses(
        self,
        attack_input_ids: torch.Tensor,
        attack_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        compute_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size = attack_input_ids.size(0)
        device = attack_input_ids.device

        prefix = self.encode_attack(attack_input_ids, attack_attention_mask)
        x0 = self.embed_tokens(target_input_ids)
        t_max = self.num_diffusion_steps - 1

        if t is None:
            t = self.sample_timesteps(batch_size, device)

        pure_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if self.pure_noise_train_frac > 0.0 and self.training:
            pure_mask = torch.rand(batch_size, device=device) < self.pure_noise_train_frac
            if pure_mask.any():
                t = torch.where(pure_mask, torch.full_like(t, t_max), t)

        x_t, _ = self.add_noise(x0, t, target_attention_mask)
        if pure_mask.any():
            pure_noise = self.noise_scale * torch.randn_like(x0)
            pad_mask = target_attention_mask.unsqueeze(-1).float()
            pure_x_t = pad_mask * pure_noise + (1.0 - pad_mask) * x0
            x_t = torch.where(pure_mask.view(-1, 1, 1), pure_x_t, x_t)

        x0_recovered = self.predict_x0(
            z_t=x_t,
            t=t,
            encoder_hidden_states=prefix,
            target_attention_mask=target_attention_mask,
            attack_attention_mask=attack_attention_mask,
            attack_input_ids=attack_input_ids,
        )

        loss_emb = self.embedding_mse_loss(x0, x0_recovered, target_attention_mask)
        loss_ce, ce_frac, ce_logits, ce_stats = self.token_ce_loss(
            x0_recovered,
            target_input_ids,
            target_attention_mask,
            t=t,
            x0_reference=x0,
        )
        loss_cos = self.embedding_cosine_loss(x0, x0_recovered, target_attention_mask)
        loss_norm = self.embedding_norm_loss(x0, x0_recovered, target_attention_mask)
        # StrongREJECT operates on decoded text, hence it has no gradient path
        # back through nearest-neighbor decoding, response generation, or the
        # evaluator.  Run it only as an explicitly infrequent diagnostic.
        security_score = x0_recovered.new_zeros(())
        if compute_diagnostics:
            security_score = self.security_score(
                x0_recovered, target_attention_mask
            )
        loss = (
            self.emb_loss_weight * loss_emb
            + self.ce_loss_weight * loss_ce
            + self.cos_loss_weight * loss_cos
            + self.norm_loss_weight * loss_norm
        )

        # Normal training path: light scalars only (one L×V CE matmul above).
        result: Dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_emb": loss_emb,
            "loss_ce": loss_ce,
            "loss_cos": loss_cos,
            "loss_norm": loss_norm,
            "security_score": security_score,
            "ce_frac": ce_frac,
            "pure_noise_frac": pure_mask.float().mean(),
            **ce_stats,
        }

        # Heavy diagnostics only on an explicit interval (never every microbatch).
        # Reuse ce_logits — do not call logits_from_embeddings again.
        if compute_diagnostics:
            with torch.no_grad():
                ce_pred = ce_logits.argmax(dim=-1)
                result["token_acc_embed_ce"] = self._token_accuracy(
                    ce_pred, target_input_ids, target_attention_mask
                )
                result["x0_norm_mean"] = self._masked_mean_norm(x0, target_attention_mask)
                result["x_t_norm_mean"] = self._masked_mean_norm(x_t, target_attention_mask)
                result["x0_hat_norm_mean"] = self._masked_mean_norm(
                    x0_recovered, target_attention_mask
                )
                if not (self.training and self.disable_train_bucket_metrics):
                    result.update(
                        self._compute_bucket_metrics(
                            pure_mask=pure_mask,
                            t=t,
                            x0=x0,
                            x_t=x_t,
                            x0_recovered=x0_recovered,
                            target_input_ids=target_input_ids,
                            target_attention_mask=target_attention_mask,
                            logits=ce_logits,
                            compute_nn=True,
                        )
                    )
        return result

    def forward(
        self,
        attack_input_ids: torch.Tensor,
        attack_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        compute_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.compute_losses(
            attack_input_ids=attack_input_ids,
            attack_attention_mask=attack_attention_mask,
            target_input_ids=target_input_ids,
            target_attention_mask=target_attention_mask,
            t=t,
            compute_diagnostics=compute_diagnostics,
        )
        if return_dict:
            return losses
        return losses["loss"]

    # -- reverse diffusion (DDIM) --------------------------------------------------
    @torch.no_grad()
    def denoise_from_z(
        self,
        z_t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        target_attention_mask: torch.Tensor,
        attack_attention_mask: torch.Tensor,
        attack_input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = z_t.size(0)
        device = z_t.device
        dtype = z_t.dtype

        for step in range(self.num_diffusion_steps - 1, -1, -1):
            t_tensor = torch.full(
                (batch_size,), fill_value=step, device=device, dtype=torch.long
            )
            x0_recovered = self.predict_x0(
                z_t=z_t,
                t=t_tensor,
                encoder_hidden_states=encoder_hidden_states,
                target_attention_mask=target_attention_mask,
                attack_attention_mask=attack_attention_mask,
                attack_input_ids=attack_input_ids,
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
        target_length_margin: Optional[int] = None,
        use_decode_patch: bool = True,
        decode_method: str = "euclidean",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = attack_input_ids.size(0)
        device = attack_input_ids.device
        dtype = torch.float32

        cap = max_target_length or self.max_target_length
        prefix = self.encode_attack(attack_input_ids, attack_attention_mask)

        if target_attention_mask is None:
            infer_width, valid_lengths = self.resolve_infer_target_length(
                attack_attention_mask=attack_attention_mask,
                max_target_length=cap,
                wm_attention_mask=wm_attention_mask,
                infer_length_strategy=infer_length_strategy,
                target_length_margin=target_length_margin,
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
            batch_size, infer_width, self.d_model, device=device, dtype=dtype
        )
        z_t = self.denoise_from_z(
            z_t=z_t,
            encoder_hidden_states=prefix,
            target_attention_mask=target_attention_mask,
            attack_attention_mask=attack_attention_mask,
            attack_input_ids=attack_input_ids,
        )
        pred_ids = self.decode_embeddings(
            z_t, use_decode_patch=use_decode_patch, decode_method=decode_method
        )
        return z_t, pred_ids


# Backwards-compatible alias (this folder previously used the LLaMA2 name).
Llama2EmbeddingDiffusion = GemmaEmbeddingDiffusion


# ======================================================================================
# Checkpoint save / load
# ======================================================================================
def save_gemma_diffusion_checkpoint(model: GemmaEmbeddingDiffusion, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PeftModel.save_pretrained writes adapter weights only (cheap + fast).
    model.llama.save_pretrained(str(output_dir))
    torch.save(
        {
            "backbone": "gemma4",
            "denoiser_backbone": model.denoiser_backbone,
            "base_model_name": model.base_model_name,
            "use_lora": bool(model.use_lora),
            "lora_r": int(model.lora_r),
            "lora_alpha": int(model.lora_alpha),
            "lora_dropout": float(model.lora_dropout),
            "lora_target_modules": model.lora_target_modules,
            "time_embed": model.time_embed.state_dict(),
            "output_proj": model.output_proj.state_dict(),
            "alpha_bar": model.alpha_bar,
            "embedding_scale": model.embedding_scale,
            "num_diffusion_steps": model.num_diffusion_steps,
            "d_model": model.d_model,
            "pad_token_id": model.pad_token_id,
            "max_target_length": model.max_target_length,
            "target_length_margin": model.target_length_margin,
            "emb_loss_weight": model.emb_loss_weight,
            "ce_loss_weight": model.ce_loss_weight,
            "cos_loss_weight": model.cos_loss_weight,
            "norm_loss_weight": model.norm_loss_weight,
            "sec_loss_weight": model.sec_loss_weight,
            "ce_min_alpha_bar": model.ce_min_alpha_bar,
            "ce_max_alpha_bar": model.ce_max_alpha_bar,
            "pure_noise_train_frac": model.pure_noise_train_frac,
            "use_residual": model.use_residual,
            "residual_mode": model.residual_mode,
            "noise_scale": model.noise_scale,
            "schedule_type": model.schedule_type,
            "timestep_sampling": model.timestep_sampling,
            "ce_use_cosine": model.ce_use_cosine,
            "ce_temperature": model.ce_temperature,
            "ce_all_timesteps": model.ce_all_timesteps,
            "beta_start": model.beta_start,
            "beta_end": model.beta_end,
        },
        output_dir / "diffusion.pt",
    )


def save_llama2_diffusion_checkpoint(model: GemmaEmbeddingDiffusion, output_dir: Path) -> None:
    save_gemma_diffusion_checkpoint(model, output_dir)


def load_gemma_diffusion_checkpoint(
    model_dir: str,
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> GemmaEmbeddingDiffusion:
    model_dir = Path(model_dir)
    ckpt_path = model_dir / "diffusion.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing diffusion checkpoint: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    use_lora = bool(payload.get("use_lora", (model_dir / "adapter_config.json").is_file()))
    backbone = (
        model_name
        or payload.get("base_model_name")
        or (None if use_lora else str(model_dir))
        or _DEFAULT_MODEL_NAME
    )
    ce_max = payload.get("ce_max_alpha_bar")
    use_residual = bool(payload.get("use_residual", True))
    residual_mode = str(payload.get("residual_mode", "gated" if use_residual else "none"))

    model = GemmaEmbeddingDiffusion(
        model_name=backbone,
        num_diffusion_steps=int(payload["num_diffusion_steps"]),
        beta_start=float(payload.get("beta_start", 1e-4)),
        beta_end=float(payload.get("beta_end", 0.02)),
        pad_token_id=int(payload.get("pad_token_id", 0)),
        max_target_length=int(payload.get("max_target_length", 256)),
        target_length_margin=int(payload.get("target_length_margin", 20)),
        emb_loss_weight=float(payload.get("emb_loss_weight", 1.0)),
        ce_loss_weight=float(payload.get("ce_loss_weight", 2.0)),
        cos_loss_weight=float(payload.get("cos_loss_weight", 0.1)),
        norm_loss_weight=float(payload.get("norm_loss_weight", 0.1)),
        sec_loss_weight=float(payload.get("sec_loss_weight", 0.1)),
        ce_min_alpha_bar=float(payload.get("ce_min_alpha_bar", 0.9)),
        ce_max_alpha_bar=float(ce_max) if ce_max is not None else None,
        pure_noise_train_frac=float(payload.get("pure_noise_train_frac", 0.0)),
        use_residual=use_residual,
        residual_mode=residual_mode,
        noise_scale=(
            float(payload["noise_scale"]) if "noise_scale" in payload else None
        ),
        schedule_type=str(payload.get("schedule_type", "linear")),
        timestep_sampling=str(payload.get("timestep_sampling", "low_bias")),
        ce_use_cosine=bool(payload.get("ce_use_cosine", True)),
        ce_temperature=float(payload.get("ce_temperature", 0.05)),
        ce_all_timesteps=bool(payload.get("ce_all_timesteps", True)),
        # Build base LM first; attach saved adapter below if present.
        use_lora=False,
        lora_r=int(payload.get("lora_r", 16)),
        lora_alpha=int(payload.get("lora_alpha", 32)),
        lora_dropout=float(payload.get("lora_dropout", 0.05)),
        lora_target_modules=payload.get("lora_target_modules"),
    )

    adapter_config = model_dir / "adapter_config.json"
    if use_lora and adapter_config.is_file():
        if PeftModel is None:
            raise ImportError(
                "Checkpoint contains a LoRA adapter but peft is not installed.\n"
                f"Original import error: {_PEFT_IMPORT_ERROR}"
            )
        model.llama = PeftModel.from_pretrained(model.llama, str(model_dir))
        model.use_lora = True
    else:
        # Full fine-tune checkpoint (or merged weights) stored via save_pretrained.
        model.llama = AutoModelForCausalLM.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        model.use_lora = False

    model.time_embed.load_state_dict(payload["time_embed"])
    model.output_proj.load_state_dict(payload["output_proj"])
    model.alpha_bar.copy_(payload["alpha_bar"])
    if "embedding_scale" in payload:
        model.embedding_scale.copy_(payload["embedding_scale"])

    model.pad_token_id = int(payload.get("pad_token_id", model.pad_token_id))
    model.max_target_length = int(payload.get("max_target_length", model.max_target_length))
    model.target_length_margin = int(
        payload.get("target_length_margin", model.target_length_margin)
    )
    model.noise_scale = float(payload.get("noise_scale", model.noise_scale))
    model.schedule_type = str(payload.get("schedule_type", model.schedule_type))
    model.base_model_name = str(payload.get("base_model_name", backbone))

    if device is not None:
        model.to(device)
    return model


def load_llama2_diffusion_checkpoint(
    model_dir: str,
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> GemmaEmbeddingDiffusion:
    return load_gemma_diffusion_checkpoint(model_dir, model_name=model_name, device=device)


# ======================================================================================
# Training harness (mirrors train.py)
# ======================================================================================
def _ensure_local_temp_dir(preferred: Optional[str] = None) -> Path:
    if preferred:
        path = Path(preferred).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    """uid = os.getuid()
    for path in (
        Path("/dev/shm") / f"gemma4-embed-diffusion-{uid}",
        Path("/tmp") / f"gemma4-embed-diffusion-{uid}","""
    for path in (
        Path("/dev/shm") / "gemma4-embed-diffusion",
        Path("/tmp") / "gemma4-embed-diffusion",
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            return path
        except OSError:
            continue
    return Path(tempfile.gettempdir())


def _configure_multiprocessing_temp(tmp_dir: Optional[str] = None) -> Path:
    local = _ensure_local_temp_dir(tmp_dir)
    os.environ["TMPDIR"] = str(local)
    os.environ["TEMP"] = str(local)
    os.environ["TMP"] = str(local)
    return local


def _resolve_dataloader_num_workers(requested: int) -> int:
    if requested <= 0:
        return 0
    return min(requested, 2)


def _moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < window:
        return list(values)
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def save_training_results(
    output_dir: Path,
    log_history: List[Dict[str, Any]],
    smooth_window: int = 5,
    *,
    loss_clip: float = 1e4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)
    print(f"Wrote training log: {results_path}")

    if not log_history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped training_curves.png.")
        return

    train_points = [
        h for h in log_history if h.get("loss") is not None and h.get("eval_loss") is None
    ]
    eval_points = [h for h in log_history if h.get("eval_loss") is not None]
    train_steps = [float(h.get("step", i)) for i, h in enumerate(train_points)]
    train_epochs = [float(h.get("epoch", i)) for i, h in enumerate(train_points)]
    train_losses = [float(h["loss"]) for h in train_points]
    eval_steps = [float(h.get("step", i)) for i, h in enumerate(eval_points)]
    eval_epochs = [float(h.get("epoch", i)) for i, h in enumerate(eval_points)]
    eval_losses = [float(h["eval_loss"]) for h in eval_points]

    clipped = [min(v, loss_clip) for v in train_losses]
    keep = [v < loss_clip for v in train_losses]
    filtered_steps = [s for s, k in zip(train_steps, keep) if k]
    filtered_losses = [v for v, k in zip(train_losses, keep) if k]

    def _series(key: str) -> Tuple[List[float], List[float]]:
        pts = [h for h in log_history if h.get(key) is not None and h.get("eval_loss") is None]
        return (
            [float(h.get("step", i)) for i, h in enumerate(pts)],
            [float(h[key]) for h in pts],
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (0,0) raw + clipped + filtered train loss + eval
    ax = axes[0, 0]
    if train_steps:
        ax.plot(train_steps, train_losses, alpha=0.25, color="tab:red", linewidth=0.8, label="train raw")
        ax.plot(train_steps, clipped, color="tab:blue", linewidth=1.2, label=f"train clipped@{loss_clip:g}")
        if filtered_steps:
            ax.plot(
                filtered_steps,
                filtered_losses,
                color="tab:cyan",
                linewidth=1.3,
                label="train (CE overflow removed)",
            )
            if smooth_window > 1 and len(filtered_losses) >= smooth_window:
                ax.plot(
                    filtered_steps,
                    _moving_average(filtered_losses, smooth_window),
                    color="navy",
                    linewidth=1.6,
                    label=f"filtered MA-{smooth_window}",
                )
    if eval_steps:
        ax.plot(
            eval_steps,
            eval_losses,
            color="tab:green",
            marker="o",
            linestyle="--",
            linewidth=1.2,
            markersize=4,
            label="eval loss",
        )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Train/eval loss (raw vs clipped vs filtered)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)

    # (0,1) same by epoch with clipped y for readability
    ax = axes[0, 1]
    if train_epochs:
        ax.plot(train_epochs, clipped, color="tab:blue", linewidth=1.2, label=f"train clipped@{loss_clip:g}")
    if eval_epochs:
        ax.plot(
            eval_epochs,
            eval_losses,
            color="tab:green",
            marker="o",
            linestyle="--",
            linewidth=1.2,
            markersize=4,
            label="eval loss",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Clipped train + eval by epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # (1,0) component losses
    ax = axes[1, 0]
    for key, color in (
        ("loss_emb", "tab:blue"),
        ("loss_ce", "tab:red"),
        ("loss_cos", "tab:orange"),
        ("loss_norm", "tab:purple"),
    ):
        xs, ys = _series(key)
        if not xs:
            continue
        # Clip CE for display if historical overflows exist
        ys_plot = [min(v, loss_clip) if key == "loss_ce" else v for v in ys]
        ax.plot(xs, ys_plot, color=color, alpha=0.75, linewidth=1.0, label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("component loss")
    ax.set_title("loss_emb / loss_ce / loss_cos / loss_norm")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # (1,1) CE diagnostics if present
    ax = axes[1, 1]
    plotted = False
    for key, color in (
        ("invalid_target_fraction", "tab:red"),
        ("nonfinite_logit_fraction", "tab:orange"),
        ("nonfinite_ce_fraction", "tab:purple"),
        ("pure_noise_frac", "tab:blue"),
        ("ce_overflow_count", "tab:green"),
    ):
        xs, ys = _series(key)
        if not xs:
            continue
        plotted = True
        if key == "ce_overflow_count":
            ax2 = ax.twinx()
            ax2.plot(xs, ys, color=color, alpha=0.7, linewidth=1.0, label=key)
            ax2.set_ylabel("ce_overflow_count", color=color)
            ax2.legend(loc="upper right", fontsize=7)
        else:
            ax.plot(xs, ys, color=color, alpha=0.75, linewidth=1.0, label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("fraction")
    ax.set_title("CE diagnostics")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="upper left", fontsize=7)
    else:
        ax.text(
            0.5,
            0.5,
            "No CE diagnostic keys in log_history yet\n(retrain with updated train.py)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )

    n_huge = sum(1 for v in train_losses if v >= loss_clip)
    fig.suptitle(
        f"Gemma 4 embed diffusion — train/eval "
        f"(overflow batches: {n_huge}/{len(train_losses)})",
        fontsize=12,
    )
    fig.tight_layout()
    plot_path = output_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote plot: {plot_path}")


class GemmaDiffusionTrainer(Trainer):
    """HF Trainer with embedding + CE + cosine loss from GemmaEmbeddingDiffusion."""

    # Always logged (cheap scalars from the training objective).
    _LIGHT_LOG_KEYS = (
        "loss_emb",
        "loss_ce",
        "loss_cos",
        "loss_norm",
        "ce_frac",
        "pure_noise_frac",
        "ce_overflow_count",
        "invalid_target_fraction",
        "invalid_target_count",
        "nonfinite_logit_fraction",
        "nonfinite_ce_fraction",
    )
    # Only present on diagnostics_interval steps (or sanity/overfit paths).
    _DIAG_LOG_KEYS = (
        "security_score",
        "x0_norm_mean",
        "x_t_norm_mean",
        "x0_hat_norm_mean",
        "token_acc_embed_ce",
        "bucket_low_mse",
        "bucket_low_ce",
        "bucket_low_cos",
        "bucket_low_nn_acc",
        "bucket_low_ce_acc",
        "bucket_low_x0_hat_norm",
        "bucket_mid_mse",
        "bucket_mid_ce",
        "bucket_mid_cos",
        "bucket_mid_nn_acc",
        "bucket_mid_ce_acc",
        "bucket_mid_x0_hat_norm",
        "bucket_high_mse",
        "bucket_high_ce",
        "bucket_high_cos",
        "bucket_high_nn_acc",
        "bucket_high_ce_acc",
        "bucket_high_x0_hat_norm",
        "bucket_pure_mse",
        "bucket_pure_ce",
        "bucket_pure_cos",
        "bucket_pure_nn_acc",
        "bucket_pure_ce_acc",
        "bucket_pure_x0_hat_norm",
    )
    _SCALAR_LOG_KEYS = _LIGHT_LOG_KEYS + _DIAG_LOG_KEYS

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Resolve wrapper attrs under DDP.
        core = model.module if hasattr(model, "module") else model
        compute_diagnostics = False
        if model.training:
            interval = int(getattr(core, "diagnostics_interval", 0) or 0)
            if interval > 0:
                step = int(getattr(self.state, "global_step", 0))
                # At most once per due optimizer step (not every microbatch / grad-accum).
                if step % interval == 0 and step != getattr(self, "_last_diagnostics_step", -1):
                    compute_diagnostics = True
                    self._last_diagnostics_step = step

        losses = model(
            attack_input_ids=inputs["attack_input_ids"],
            attack_attention_mask=inputs["attack_attention_mask"],
            target_input_ids=inputs["target_input_ids"],
            target_attention_mask=inputs["target_attention_mask"],
            return_dict=True,
            compute_diagnostics=compute_diagnostics,
        )
        # Only log on logging_steps (and at most once per optimizer step).
        # Logging every microbatch floods stdout and slows training.
        if hasattr(self, "log") and model.training:
            logging_steps = max(1, int(getattr(self.args, "logging_steps", 50) or 50))
            step = int(getattr(self.state, "global_step", 0))
            should_log = (
                step % logging_steps == 0
                and step != getattr(self, "_last_metric_log_step", -1)
            )
            if should_log:
                keys = self._LIGHT_LOG_KEYS
                if compute_diagnostics:
                    keys = self._SCALAR_LOG_KEYS
                log_payload = {}
                for key in keys:
                    if key in losses:
                        log_payload[key] = losses[key].detach().item()
                if log_payload:
                    self.log(log_payload)
                    self._last_metric_log_step = step
        return (losses["loss"], None) if return_outputs else losses["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss.detach().mean(), None, None)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = Path(output_dir if output_dir is not None else self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        save_gemma_diffusion_checkpoint(self.model, output_dir)
        # With LoRA, avoid dumping the full frozen backbone into wrapper_state.bin.
        if getattr(self.model, "use_lora", False):
            slim = {
                k: v.detach().cpu()
                for k, v in state_dict.items()
                if ("lora_" in k)
                or k.startswith("time_embed.")
                or k.startswith("output_proj.")
                or k in ("alpha_bar", "embedding_scale")
            }
            torch.save(slim, output_dir / "wrapper_state.bin")
        else:
            torch.save(state_dict, output_dir / "wrapper_state.bin")
        if self.processing_class is not None and self.args.should_save:
            self.processing_class.save_pretrained(str(output_dir))
        torch.save(self.args, output_dir / "training_args.bin")

    def _load_wrapper_state(self, checkpoint_dir: str, model=None) -> bool:
        if not checkpoint_dir:
            return False
        checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir, self.args.output_dir)
        wrapper_path = os.path.join(checkpoint_dir, "wrapper_state.bin")
        if not os.path.isfile(wrapper_path):
            return False
        if model is None:
            model = self.model
        state_dict = torch.load(wrapper_path, map_location="cpu", weights_only=False)
        strict = not getattr(model, "use_lora", False)
        model.load_state_dict(state_dict, strict=strict)
        return True

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        resume_from_checkpoint = resolve_checkpoint_dir(
            resume_from_checkpoint, self.args.output_dir
        )
        if self._load_wrapper_state(resume_from_checkpoint, model=model):
            return
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _torch_load_checkpoint(self, path: str, map_location=None):
        """Load a trusted local .pt checkpoint under torch<2.6."""
        kwargs = {"map_location": map_location} if map_location is not None else {}
        try:
            return torch.load(path, weights_only=False, **kwargs)
        except TypeError:
            # Older torch without weights_only=
            return torch.load(path, **kwargs)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Resume optimizer/scheduler without the torch>=2.6 transformers gate."""
        if checkpoint is None:
            return
        opt_path = os.path.join(checkpoint, "optimizer.pt")
        sched_path = os.path.join(checkpoint, "scheduler.pt")
        if not (os.path.isfile(opt_path) and os.path.isfile(sched_path)):
            _allow_trusted_local_torch_load()
            super()._load_optimizer_and_scheduler(checkpoint)
            return

        map_location = self.args.device if self.args.world_size > 1 else "cpu"
        self.optimizer.load_state_dict(
            self._torch_load_checkpoint(opt_path, map_location=map_location)
        )
        self.lr_scheduler.load_state_dict(self._torch_load_checkpoint(sched_path))
        print(f"[info] restored optimizer/scheduler from {checkpoint}")

    def _load_rng_state(self, checkpoint):
        """Restore RNG states from local checkpoint without the torch>=2.6 gate."""
        if checkpoint is None:
            return
        if self.args.world_size > 1:
            rng_file = os.path.join(checkpoint, f"rng_state_{self.args.process_index}.pth")
            if not os.path.isfile(rng_file):
                print(
                    f"[warn] missing {rng_file}; reproducibility not guaranteed when resuming DDP"
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                print(
                    "[warn] missing rng_state.pth; reproducibility not guaranteed when resuming"
                )
                return

        import random

        import numpy as np

        checkpoint_rng_state = self._torch_load_checkpoint(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available() and "cuda" in checkpoint_rng_state:
            cuda_state = checkpoint_rng_state["cuda"]
            # DDP checkpoints typically store a list of per-device CUDA RNG states.
            if self.args.world_size > 1 or isinstance(cuda_state, (list, tuple)):
                torch.cuda.random.set_rng_state_all(cuda_state)
            else:
                torch.cuda.random.set_rng_state(cuda_state)

    def _load_best_model(self):
        best = getattr(self.state, "best_model_checkpoint", None)
        if best:
            remapped = resolve_checkpoint_dir(best, self.args.output_dir)
            if remapped != best:
                print(
                    f"[info] remapped best_model_checkpoint:\n"
                    f"  {best}\n  -> {remapped}"
                )
                self.state.best_model_checkpoint = remapped
        if self._load_wrapper_state(self.state.best_model_checkpoint):
            return
        super()._load_best_model()


# Backwards-compatible alias
Llama2DiffusionTrainer = GemmaDiffusionTrainer

# NFS lab path used when this run was originally launched on the cluster.
_CLUSTER_OUTPUT_PREFIXES = (
    "/network/rit/lab/Lai_ReSecureAI/kiel/Translator/Gemma/",
)


def resolve_checkpoint_dir(
    checkpoint_dir: Optional[str],
    output_dir: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Resolve a checkpoint path, remapping old cluster mounts if needed."""
    if checkpoint_dir is None:
        return None
    path = Path(str(checkpoint_dir)).expanduser()
    if path.is_dir():
        return str(path)

    name = path.name
    if name.startswith("checkpoint-") and output_dir is not None:
        local = Path(output_dir).expanduser().resolve() / name
        if local.is_dir():
            return str(local)

    text = str(path)
    for prefix in _CLUSTER_OUTPUT_PREFIXES:
        if text.startswith(prefix):
            remapped = Path(__file__).resolve().parent / text[len(prefix) :]
            if remapped.is_dir():
                return str(remapped)

    return str(path)


def remap_trainer_state_paths(checkpoint_dir: str, output_dir: Path) -> None:
    """Rewrite best_model_checkpoint in trainer_state.json onto the local mount."""
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.is_file():
        return
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    best = state.get("best_model_checkpoint")
    if not best:
        return
    remapped = resolve_checkpoint_dir(best, output_dir)
    if remapped is None or remapped == best:
        return
    if not Path(remapped).is_dir():
        return
    state["best_model_checkpoint"] = remapped
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.write("\n")
    print(f"[info] updated best_model_checkpoint in {state_path} -> {remapped}")


def run_debug_overfit_one_batch(
    model: GemmaEmbeddingDiffusion,
    batch: Dict[str, torch.Tensor],
    *,
    steps: int = 500,
    lr: float = 1e-4,
    freeze_llama: bool = True,
    overfit_timestep: int = 0,
    overfit_schedule: str = "fixed",
    grad_clip: float = 1.0,
) -> None:
    """Train on a single batch and report metrics at fixed timesteps."""
    device = next(model.parameters()).device
    saved_pure_noise_frac = model.pure_noise_train_frac
    model.pure_noise_train_frac = 0.0
    print(
        f"[info] overfit: pure_noise_train_frac disabled | schedule={overfit_schedule} | "
        f"timestep={overfit_timestep if overfit_schedule == 'fixed' else 'cycle 0..T-1'} | "
        f"lr={lr} | grad_clip={grad_clip}",
        flush=True,
    )
    if freeze_llama and not model.use_lora:
        for param in model.llama.parameters():
            param.requires_grad = False
    trainable = [
        p for p in list(model.time_embed.parameters()) + list(model.output_proj.parameters())
        if p.requires_grad
    ]
    if model.use_lora or not freeze_llama:
        trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    infer_width = int(batch["target_attention_mask"].sum(dim=1).max().item())
    eval_mask = model.build_target_mask(
        batch_size=1,
        target_length=infer_width,
        device=device,
        valid_lengths=batch["target_attention_mask"].sum(dim=1).clamp(max=infer_width),
    )
    target_ids = batch["target_input_ids"][:, :infer_width]
    attack_mask = batch["attack_attention_mask"]
    x0 = model.embed_tokens(target_ids)
    prefix = model.encode_attack(batch["attack_input_ids"], attack_mask)
    t_max = model.num_diffusion_steps - 1

    with torch.no_grad():
        nn_true = model.embeddings_to_token_ids(x0)
        lm_true = model.logits_from_embeddings(x0, use_cosine=False).argmax(dim=-1)
        nn_agree = model._token_accuracy(nn_true[0], target_ids[0], eval_mask[0]).item()
        lm_agree = model._token_accuracy(lm_true[0], target_ids[0], eval_mask[0]).item()
    print(f"\n=== debug_overfit_one_batch ===")
    print(f"NN(true x0) token agreement: {nn_agree:.4f}")
    print(f"lm_head(true x0) token agreement: {lm_agree:.4f} (expected ~0)")

    def evaluate(label: str, t_val: int, *, pure: bool = False) -> None:
        seq_len = int(attack_mask.sum()) + int(eval_mask.sum())
        print(f"  evaluating {label} (t={t_val}, seq_len≈{seq_len})...", flush=True)
        model.eval()
        with torch.inference_mode():
            t = torch.tensor([t_val], device=device)
            if pure:
                x_t = model.noise_scale * torch.randn_like(x0)
            else:
                x_t, _ = model.add_noise(x0, t, eval_mask)
            x0_hat = model.predict_x0(
                z_t=x_t,
                t=t,
                encoder_hidden_states=prefix,
                target_attention_mask=eval_mask,
                attack_attention_mask=attack_mask,
                attack_input_ids=batch["attack_input_ids"],
            )
            nn_pred = model.embeddings_to_token_ids(x0_hat)
            ce_pred = model.logits_from_embeddings(x0_hat).argmax(dim=-1)
            mse = model.embedding_mse_loss(x0, x0_hat, eval_mask).item()
            nn_acc = model._token_accuracy(nn_pred[0], target_ids[0], eval_mask[0]).item()
            ce_acc = model._token_accuracy(ce_pred[0], target_ids[0], eval_mask[0]).item()
            x0_n = model._masked_mean_norm(x0, eval_mask).item()
            xt_n = model._masked_mean_norm(x_t, eval_mask).item()
            xh_n = model._masked_mean_norm(x0_hat, eval_mask).item()
        print(
            f"{label:>20}: MSE={mse:.4f} | NN acc={nn_acc:.3f} | CE acc={ce_acc:.3f} | "
            f"||x0||={x0_n:.3f} ||xt||={xt_n:.3f} ||x0_hat||={xh_n:.3f}",
            flush=True,
        )
        model.train()

    evaluate("initial t=0", 0)
    evaluate("initial t=5", min(5, t_max))
    evaluate("initial t=50", min(50, t_max))
    evaluate("initial t=99", min(99, t_max))
    evaluate("initial pure-noise", t_max, pure=True)

    print(f"\n--- overfit training ({steps} steps) ---", flush=True)
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        if overfit_schedule == "fixed":
            t = torch.tensor([overfit_timestep], device=device)
        else:
            t = torch.tensor([step % (t_max + 1)], device=device)
        losses = model.compute_losses(
            attack_input_ids=batch["attack_input_ids"],
            attack_attention_mask=attack_mask,
            target_input_ids=target_ids,
            target_attention_mask=eval_mask,
            t=t,
            compute_diagnostics=True,
        )
        losses["loss"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        if step % max(steps // 5, 1) == 0 or step == steps - 1:
            print(
                f"step {step:4d}: loss={losses['loss'].item():.4f} "
                f"emb={losses['loss_emb'].item():.4f} ce={losses['loss_ce'].item():.4f} "
                f"nn_acc={losses['token_acc_embed_ce'].item():.3f}",
                flush=True,
            )

    print("\n--- after overfit ---")
    evaluate("final t=0", 0)
    evaluate("final t=5", min(5, t_max))
    evaluate("final t=50", min(50, t_max))
    evaluate("final t=99", min(99, t_max))
    evaluate("final pure-noise", t_max, pure=True)
    model.pure_noise_train_frac = saved_pure_noise_frac


def _distributed_world_size() -> int:
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def train(args):
    set_seed(args.seed)

    world_size = _distributed_world_size()
    if world_size > 1 and (args.freeze_llama or args.use_lora):
        if not args.ddp_find_unused_parameters:
            args.ddp_find_unused_parameters = True
            print(
                "[info] Multi-GPU + frozen/LoRA backbone: enabled "
                "ddp_find_unused_parameters (required for DDP)."
            )

    tmp_dir = _configure_multiprocessing_temp(args.tmp_dir)
    num_workers = _resolve_dataloader_num_workers(args.dataloader_num_workers)
    print(f"multiprocessing temp dir: {tmp_dir}")
    print(f"dataloader_num_workers: {num_workers}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"output_dir: {output_dir}")
    print(f"train_json: {args.train_json}")
    if args.number_training_data is not None:
        print(f"number_training_data: {args.number_training_data}")
    print(f"eval_json: {args.eval_json}")
    print(f"model_name: {args.model_name}")
    print(f"max_steps: {args.max_steps}")
    print(f"num_diffusion_steps: {args.num_diffusion_steps}")
    print(f"max_target_length: {args.max_target_length}")
    print(f"target_length_margin: {args.target_length_margin}")
    print(
        f"loss = {args.emb_loss_weight}*emb + "
        f"{args.ce_loss_weight}*ce + {args.cos_loss_weight}*cos + "
        f"{args.norm_loss_weight}*norm"
    )
    print(
        f"ce_min_alpha_bar: {args.ce_min_alpha_bar} | "
        f"ce_max_alpha_bar: {args.ce_max_alpha_bar} | ce_all_timesteps: {args.ce_all_timesteps}"
    )
    print(f"pure_noise_train_frac: {args.pure_noise_train_frac}")
    print(
        f"diagnostics_interval: {args.diagnostics_interval} | "
        f"disable_train_bucket_metrics: {args.disable_train_bucket_metrics}"
    )
    print(f"denoiser_backbone: gemma_prefix (decoder-only prefix concat)")
    residual_mode = args.residual_mode
    if not args.use_residual:
        residual_mode = "none"
    print(
        f"residual_mode: {residual_mode} | noise_scale: "
        f"{'auto' if args.noise_scale is None else args.noise_scale} | "
        f"ce_use_cosine: {args.ce_use_cosine} | ce_temperature: {args.ce_temperature}"
    )
    print(f"schedule_type: {args.schedule_type}, timestep_sampling: {args.timestep_sampling}")
    print(
        f"use_lora: {args.use_lora} | lora_r={args.lora_r} | "
        f"lora_alpha={args.lora_alpha} | lora_dropout={args.lora_dropout}"
    )
    effective_batch = args.batch_size * args.grad_accum * world_size
    print(
        f"distributed: world_size={world_size} | "
        f"per_device_batch={args.batch_size} | grad_accum={args.grad_accum} | "
        f"effective_batch_per_optimizer_step={effective_batch}"
    )
    print(f"ddp_find_unused_parameters: {args.ddp_find_unused_parameters}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if args.bf16
        else (torch.float16 if args.fp16 else torch.float32)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[warn] CUDA not available; Gemma 4 E4B will be extremely slow on CPU.")

    model = GemmaEmbeddingDiffusion(
        model_name=args.model_name,
        num_diffusion_steps=args.num_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        pad_token_id=tokenizer.pad_token_id,
        max_target_length=args.max_target_length,
        target_length_margin=args.target_length_margin,
        emb_loss_weight=args.emb_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        cos_loss_weight=args.cos_loss_weight,
        norm_loss_weight=args.norm_loss_weight,
        ce_min_alpha_bar=args.ce_min_alpha_bar,
        ce_max_alpha_bar=args.ce_max_alpha_bar,
        pure_noise_train_frac=args.pure_noise_train_frac,
        use_residual=args.use_residual,
        residual_mode=residual_mode,
        noise_scale=args.noise_scale,
        schedule_type=args.schedule_type,
        timestep_sampling=args.timestep_sampling,
        ce_use_cosine=args.ce_use_cosine,
        ce_temperature=args.ce_temperature,
        ce_all_timesteps=args.ce_all_timesteps,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        torch_dtype=dtype if device.type == "cuda" else None,
        diagnostics_interval=args.diagnostics_interval,
        disable_train_bucket_metrics=args.disable_train_bucket_metrics,
    )
    model.configure_token_mask(tokenizer)
    model.to(device=device, dtype=dtype)
    # Diffusion heads stay fp32; Gemma backbone uses bf16/fp16.
    model.time_embed.float()
    model.output_proj.float()
    print(f"device: {device} | dtype: {dtype}")
    print(
        f"embedding_scale (raw→unit-median diffusion): {float(model.embedding_scale):.4f} | "
        f"noise_scale={model.noise_scale:.6f}"
    )

    if args.gradient_checkpointing and not args.debug_overfit_one_batch:
        model.llama.gradient_checkpointing_enable()
        if hasattr(model.llama, "config"):
            model.llama.config.use_cache = False

    if args.freeze_llama and not args.use_lora:
        for param in model.llama.parameters():
            param.requires_grad = False
        for param in model.time_embed.parameters():
            param.requires_grad = True
        for param in model.output_proj.parameters():
            param.requires_grad = True
        print("[info] backbone frozen; only time_embed + output_proj train.")
    elif args.use_lora:
        for param in model.time_embed.parameters():
            param.requires_grad = True
        for param in model.output_proj.parameters():
            param.requires_grad = True
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"[info] LoRA enabled; trainable params={n_train:,} / {n_total:,} "
            f"({100.0 * n_train / max(n_total, 1):.2f}%)."
        )

    # The paired JSONL schema is {"vanilla": input, "prompt": expected output}.
    # WatermarkRecoveryDataset maps those fields to the model's condition/target tensors.
    train_dataset = WatermarkRecoveryDataset(
        json_path=args.train_json,
        max_samples=1 if args.debug_overfit_one_batch else args.number_training_data,
    )
    if args.debug_overfit_one_batch:
        print("[info] debug_overfit_one_batch: using 1 training example only.")
    print(f"train examples: {len(train_dataset)}")

    eval_dataset: Optional[Dataset] = None
    if args.eval_json:
        eval_dataset = WatermarkRecoveryDataset(json_path=args.eval_json)
        print(f"eval examples: {len(eval_dataset)}")

    data_collator = BartDiffusionCollator(
        tokenizer=tokenizer,
        max_attack_length=args.max_attack_length,
        max_target_length=args.max_target_length,
        target_length_margin=args.target_length_margin,
        pad_token_id=tokenizer.pad_token_id,
    )

    if args.debug_overfit_one_batch:
        device = next(model.parameters()).device
        batch = data_collator([train_dataset[0]])
        for key, value in batch.items():
            batch[key] = value.to(device)
        run_debug_overfit_one_batch(
            model,
            batch,
            steps=args.overfit_steps,
            lr=args.overfit_lr,
            freeze_llama=args.freeze_llama,
            overfit_timestep=args.overfit_timestep,
            overfit_schedule=args.overfit_schedule,
            grad_clip=args.overfit_grad_clip,
        )
        return

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        resume_from_checkpoint = resolve_checkpoint_dir(
            resume_from_checkpoint, output_dir
        )
        if not Path(resume_from_checkpoint).is_dir():
            raise FileNotFoundError(
                f"resume_from_checkpoint does not exist or is not a directory: {resume_from_checkpoint}"
            )
        remap_trainer_state_paths(resume_from_checkpoint, output_dir)
        _allow_trusted_local_torch_load()
        print(f"[info] resuming training from checkpoint: {resume_from_checkpoint}")

    gc_kwargs = {"use_reentrant": False} if args.gradient_checkpointing else None
    use_eval = eval_dataset is not None
    load_best = use_eval and args.load_best_model_at_end

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps" if use_eval or args.max_steps > 0 else "epoch",
        eval_strategy="steps" if use_eval else "no",
        eval_steps=args.eval_steps if use_eval else None,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=num_workers > 0,
        remove_unused_columns=False,
        report_to="none",
    )

    callbacks = []
    if use_eval and args.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer = GemmaDiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    save_gemma_diffusion_checkpoint(model, output_dir)
    tokenizer.save_pretrained(str(output_dir))

    if load_best and trainer.state.best_model_checkpoint:
        print(f"Best checkpoint (eval_loss): {trainer.state.best_model_checkpoint}")
        print(f"Best eval_loss: {trainer.state.best_metric}")

    save_training_results(
        output_dir=output_dir,
        log_history=list(trainer.state.log_history),
        smooth_window=args.plot_smooth_window,
    )
    print(f"Saved model to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gemma 4 embedding diffusion attack training script"
    )

    parser.add_argument("--train_json", type=str, default=_DEFAULT_TRAIN_JSON)
    parser.add_argument("--number_training_data", type=int, default=100000)
    parser.add_argument("--eval_json", type=str, default=_DEFAULT_EVAL_JSON)
    parser.add_argument("--output_dir", type=str, default="./save_model_gemma4_diffusion")
    parser.add_argument(
        "--model_name",
        type=str,
        default=_DEFAULT_MODEL_NAME,
        help="HF id or local path for the Gemma 4 E4B backbone",
    )

    parser.add_argument("--max_attack_length", type=int, default=256)
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=200,
        help="Hard cap on target/wm sequence length.",
    )
    parser.add_argument(
        "--target_length_margin",
        type=int,
        default=20,
        help=(
            "Target width per sample is min(max_target_length, attack_len + margin). "
            "Gives room because watermarked text is usually longer than the attack."
        ),
    )
    parser.add_argument("--num_diffusion_steps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.1)
    parser.add_argument("--emb_loss_weight", type=float, default=1.0)
    parser.add_argument("--ce_loss_weight", type=float, default=2.0)
    parser.add_argument("--cos_loss_weight", type=float, default=0.1)
    parser.add_argument("--norm_loss_weight", type=float, default=0.1)
    parser.add_argument(
        "--sec_loss_weight",
        type=float,
        default=0.1,
        help=(
            "Deprecated compatibility option. StrongREJECT scoring is a "
            "non-differentiable diagnostic and is not added to training loss."
        ),
    )
    parser.add_argument("--ce_min_alpha_bar", type=float, default=0.9)
    parser.add_argument("--ce_max_alpha_bar", type=float, default=0.05)
    parser.add_argument(
        "--ce_all_timesteps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply CE at every diffusion timestep (recommended for debugging).",
    )
    parser.add_argument("--pure_noise_train_frac", type=float, default=0.25)
    parser.add_argument(
        "--use_residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy flag; if false, forces residual_mode=none.",
    )
    parser.add_argument(
        "--residual_mode",
        type=str,
        choices=("full", "gated", "none"),
        default="gated",
        help="full: x0_hat=z_t+delta; gated: sqrt(alpha_bar)*z_t+delta; none: direct delta.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=None,
        help=(
            "Std of Gaussian noise in unit-median-normalized embedding space. "
            "Default: auto = BART-matched L2 (≈0.033 for Gemma-4 D≈2560). "
            "The old 0.012 default was ~3x too small once embeddings are scaled "
            "correctly, and ~100x too small when unused-token rows collapsed the "
            "vocab median (x0_norm≈50 with noise L2≈0.6)."
        ),
    )
    parser.add_argument(
        "--schedule_type",
        type=str,
        choices=("linear", "cosine"),
        default="linear",
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        choices=("low_bias", "uniform", "mixed"),
        default="uniform",
    )
    parser.add_argument(
        "--ce_use_cosine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use normalized embedding-table cosine logits for CE (recommended).",
    )
    parser.add_argument(
        "--ce_temperature",
        type=float,
        default=0.05,
        help="Temperature for embedding-table cosine CE logits.",
    )
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train Gemma backbone with LoRA adapters (default: on).",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help=(
            "LoRA target regex/CSV. Default auto-scopes to language_model.* for Gemma 4 "
            "to avoid multimodal ClippableLinear modules."
        ),
    )
    parser.add_argument(
        "--freeze_llama",
        action="store_true",
        help="Freeze the backbone fully; train only time_embed + output_proj (ignored when LoRA on).",
    )
    parser.add_argument(
        "--debug_overfit_one_batch",
        action="store_true",
        help="Train on 1 batch only, print fixed-t metrics, then exit.",
    )
    parser.add_argument("--overfit_steps", type=int, default=500)
    parser.add_argument("--overfit_lr", type=float, default=1e-4)
    parser.add_argument(
        "--overfit_timestep",
        type=int,
        default=0,
        help="Fixed diffusion timestep for overfit when --overfit_schedule=fixed.",
    )
    parser.add_argument(
        "--overfit_schedule",
        type=str,
        choices=("fixed", "cycle"),
        default="fixed",
        help="fixed: train one t only (recommended). cycle: rotate t each step.",
    )
    parser.add_argument("--overfit_grad_clip", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Optimizer steps. Default -1 uses --epochs instead.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--tmp_dir", type=str, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument(
        "--diagnostics_interval",
        type=int,
        default=500,
        help=(
            "Compute heavy train diagnostics (token accuracy, norms, optional bucket "
            "metrics) every N optimizer steps. Set 0 to disable. Light loss scalars "
            "are logged only every --logging_steps."
        ),
    )
    parser.add_argument(
        "--disable_train_bucket_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip bucket MSE/CE/NN accuracy metrics during training even on "
            "diagnostics steps. Use --no-disable_train_bucket_metrics to re-enable."
        ),
    )

    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    parser.add_argument(
        "--load_best_model_at_end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="./gemma4_embed_diffusion/checkpoint-1500",
        help=(
            "Path to a Trainer checkpoint directory to resume optimizer/model state from. "
            "Pass an empty string to start fresh."
        ),
    )

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_smooth_window", type=int, default=5)

    args = parser.parse_args()

    # Allow --resume_from_checkpoint "" (or "none"/"false") to disable the default.
    if isinstance(args.resume_from_checkpoint, str):
        stripped = args.resume_from_checkpoint.strip()
        if stripped == "" or stripped.lower() in {"none", "false", "0"}:
            args.resume_from_checkpoint = None
        else:
            args.resume_from_checkpoint = stripped

    # <=0 means auto-calibrate to BART-matched noise L2 for this d_model.
    if args.noise_scale is not None and args.noise_scale <= 0:
        args.noise_scale = None

    if args.max_steps is not None and args.max_steps <= 0:
        args.max_steps = -1

    if not args.bf16 and not args.fp16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("[info] No --bf16/--fp16 given; enabling --bf16 automatically.")
            args.bf16 = True

    if not args.eval_json:
        args.eval_json = None
    if not args.eval_json:
        if args.early_stopping:
            print("[warn] --eval_json not set; disabling early stopping.")
        args.early_stopping = False
        args.load_best_model_at_end = False

    if args.eval_json and args.save_steps % args.eval_steps != 0:
        print(
            f"[warn] save_steps ({args.save_steps}) != eval_steps ({args.eval_steps}); "
            "aligning save_steps to eval_steps for load_best_model_at_end."
        )
        args.save_steps = args.eval_steps

    train(args)
