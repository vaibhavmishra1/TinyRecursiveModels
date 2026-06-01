from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class GenerativeRecursiveReasoningModelInnerCarry:
    h: torch.Tensor
    l: torch.Tensor


@dataclass
class GenerativeRecursiveReasoningModelCarry:
    inner_carry: GenerativeRecursiveReasoningModelInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class GenerativeRecursiveReasoningModelConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Paper notation: K low-level refinements and T high-level stochastic transitions.
    K_steps: int = 4
    T_steps: int = 3
    N_sup: int = 16

    H_layers: int = 2
    L_layers: int = 2

    hidden_size: int = 512
    expansion: float = 4
    num_heads: int = 8
    pos_encodings: str = "rope"

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"

    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.0

    puzzle_emb_len: int = 16
    mlp_t: bool = False

    # Stochastic guidance.
    min_log_std: float = -10.0
    max_log_std: float = 2.0
    posterior_target_conditioning: str = "add"
    decoder_swiglu: bool = True

    # Compatibility aliases used by the older TRM configs.
    H_cycles: Optional[int] = None
    L_cycles: Optional[int] = None

    def resolved_T(self) -> int:
        return self.T_steps if self.H_cycles is None else self.H_cycles

    def resolved_K(self) -> int:
        return self.K_steps if self.L_cycles is None else self.L_cycles


class GRAMBlock(nn.Module):
    def __init__(self, config: GenerativeRecursiveReasoningModelConfig) -> None:
        super().__init__()
        self.config = config

        if config.mlp_t:
            puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size) if config.puzzle_emb_len == 0 else config.puzzle_emb_len
            self.mlp_t = SwiGLU(hidden_size=config.seq_len + puzzle_emb_len, expansion=config.expansion)
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )

        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states_t = hidden_states.transpose(1, 2)
            hidden_states_t = rms_norm(hidden_states_t + self.mlp_t(hidden_states_t), variance_epsilon=self.norm_eps)
            hidden_states = hidden_states_t.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )

        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class GRAMReasoningModule(nn.Module):
    def __init__(self, layers: List[GRAMBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class GRAMGuidanceHead(nn.Module):
    """SwiGLU MLP parameterizing one diagonal-Gaussian parameter."""

    def __init__(self, config: GenerativeRecursiveReasoningModelConfig):
        super().__init__()
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.out = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
        self.norm_eps = config.rms_norm_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(rms_norm(x + self.mlp(x), variance_epsilon=self.norm_eps))


def gaussian_kl_diag(
    q_mu: torch.Tensor,
    q_log_std: torch.Tensor,
    p_mu: torch.Tensor,
    p_log_std: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    q_mu = q_mu.to(torch.float32)
    q_log_std = q_log_std.to(torch.float32)
    p_mu = p_mu.to(torch.float32)
    p_log_std = p_log_std.to(torch.float32)
    q_var = torch.exp(2.0 * q_log_std)
    p_var = torch.exp(2.0 * p_log_std)
    kl = p_log_std - q_log_std + (q_var + (q_mu - p_mu).square()) / (2.0 * p_var.clamp_min(1e-12)) - 0.5
    kl = kl.clamp_min(0)
    if reduction == "sum":
        return kl.flatten(1).sum(-1)
    if reduction == "mean":
        return kl.flatten(1).mean(-1)
    raise ValueError(f"Unsupported KL reduction: {reduction}")


class GenerativeRecursiveReasoningModelInner(nn.Module):
    def __init__(self, config: GenerativeRecursiveReasoningModelConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.decoder_mlp = SwiGLU(config.hidden_size, config.expansion) if config.decoder_swiglu else None
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        self.v_head = CastedLinear(config.hidden_size, 1, bias=True)

        self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size) if config.puzzle_emb_len == 0 else config.puzzle_emb_len
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers,
                config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len + self.puzzle_emb_len,
                config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        elif config.pos_encodings == "none":
            pass
        else:
            raise ValueError(f"Unsupported pos_encodings={config.pos_encodings}")

        self.L_level = GRAMReasoningModule([GRAMBlock(config) for _ in range(config.L_layers)])
        self.H_level = GRAMReasoningModule([GRAMBlock(config) for _ in range(config.H_layers)])

        self.prior_mu = GRAMGuidanceHead(config)
        self.prior_log_std = GRAMGuidanceHead(config)
        self.posterior_mu = GRAMGuidanceHead(config)
        self.posterior_log_std = GRAMGuidanceHead(config)

        self.h_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.l_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore[union-attr]
            self.v_head.weight.zero_()
            self.v_head.bias.zero_()  # type: ignore[union-attr]

    def _sequence_embeddings(self, tokens: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(tokens.to(torch.int32))

        if self.puzzle_emb_len > 0:
            if self.config.puzzle_emb_ndim > 0:
                puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
                pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
                if pad_count > 0:
                    puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
                puzzle_embedding = puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size)
            else:
                puzzle_embedding = torch.zeros(
                    tokens.shape[0],
                    self.puzzle_emb_len,
                    self.config.hidden_size,
                    dtype=self.forward_dtype,
                    device=tokens.device,
                )
            embedding = torch.cat(
                (puzzle_embedding, embedding),
                dim=-2,
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def _target_embeddings(self, labels: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
        return self._sequence_embeddings(labels, puzzle_identifiers)

    def empty_carry(self, batch_size: int):
        return GenerativeRecursiveReasoningModelInnerCarry(
            h=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            l=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: GenerativeRecursiveReasoningModelInnerCarry):
        return GenerativeRecursiveReasoningModelInnerCarry(
            h=torch.where(reset_flag.view(-1, 1, 1), self.h_init, carry.h),
            l=torch.where(reset_flag.view(-1, 1, 1), self.l_init, carry.l),
        )

    def _prior_params(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.prior_mu(u)
        log_std = self.prior_log_std(u).clamp(self.config.min_log_std, self.config.max_log_std)
        return mu, log_std

    def _posterior_params(self, u: torch.Tensor, target_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.posterior_target_conditioning == "add":
            x = u + target_embeddings
        else:
            raise ValueError(f"Unsupported posterior_target_conditioning={self.config.posterior_target_conditioning}")
        mu = self.posterior_mu(x)
        log_std = self.posterior_log_std(x).clamp(self.config.min_log_std, self.config.max_log_std)
        return mu, log_std

    def _sample_transition(
        self,
        h: torch.Tensor,
        l: torch.Tensor,
        input_embeddings: torch.Tensor,
        target_embeddings: Optional[torch.Tensor],
        seq_info: Dict[str, torch.Tensor],
        sample_from_posterior: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        for _ in range(self.config.resolved_K()):
            l = self.L_level(l, h + input_embeddings, **seq_info)

        u = self.H_level(h, l, **seq_info)
        p_mu, p_log_std = self._prior_params(u)

        q_mu = q_log_std = None
        if sample_from_posterior:
            assert target_embeddings is not None
            q_mu, q_log_std = self._posterior_params(u, target_embeddings)
            sample_mu, sample_log_std = q_mu, q_log_std
        else:
            sample_mu, sample_log_std = p_mu, p_log_std

        eps = sample_mu + torch.exp(sample_log_std) * torch.randn_like(sample_mu)
        h = u + eps

        info = {
            "prior_mu": p_mu,
            "prior_log_std": p_log_std,
            "sample_mu": sample_mu,
            "sample_log_std": sample_log_std,
            "v_logits": self.v_head(h[:, 0]).squeeze(-1).to(torch.float32),
        }
        if q_mu is not None and q_log_std is not None:
            info.update(
                {
                    "posterior_mu": q_mu,
                    "posterior_log_std": q_log_std,
                    "kl": gaussian_kl_diag(q_mu, q_log_std, p_mu, p_log_std),
                    "kl_prior_grad": gaussian_kl_diag(q_mu.detach(), q_log_std.detach(), p_mu, p_log_std),
                    "kl_posterior_grad": gaussian_kl_diag(q_mu, q_log_std, p_mu.detach(), p_log_std.detach()),
                }
            )
        return h, l, info

    def forward(
        self,
        carry: GenerativeRecursiveReasoningModelInnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[GenerativeRecursiveReasoningModelInnerCarry, Dict[str, torch.Tensor]]:
        seq_info = {"cos_sin": self.rotary_emb() if hasattr(self, "rotary_emb") else None}
        input_embeddings = self._sequence_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        labels = batch.get("labels")
        sample_from_posterior = self.training and labels is not None
        target_embeddings = self._target_embeddings(labels, batch["puzzle_identifiers"]) if sample_from_posterior else None

        h, l = carry.h, carry.l
        early_h_states = []

        with torch.no_grad():
            for _ in range(max(self.config.resolved_T() - 1, 0)):
                h, l, info = self._sample_transition(
                    h,
                    l,
                    input_embeddings,
                    target_embeddings,
                    seq_info,
                    sample_from_posterior=sample_from_posterior,
                )
                early_h_states.append(h.detach())

        transition_v_logits = [
            self.v_head(state[:, 0]).squeeze(-1).to(torch.float32)
            for state in early_h_states
        ]

        h, l, final_info = self._sample_transition(
            h,
            l,
            input_embeddings,
            target_embeddings,
            seq_info,
            sample_from_posterior=sample_from_posterior,
        )
        transition_v_logits.append(final_info["v_logits"])

        decoder_state = h
        if self.decoder_mlp is not None:
            decoder_state = rms_norm(decoder_state + self.decoder_mlp(decoder_state), variance_epsilon=self.config.rms_norm_eps)
        logits = self.lm_head(decoder_state)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(h[:, 0].detach()).to(torch.float32)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_logits[..., 0],
            "q_continue_logits": q_logits[..., 1],
            "v_logits": final_info["v_logits"],
            "transition_v_logits": transition_v_logits,
            "prior_std": torch.exp(final_info["prior_log_std"]).detach().to(torch.float32).mean(),
            "sample_std": torch.exp(final_info["sample_log_std"]).detach().to(torch.float32).mean(),
        }
        for key in ("kl", "kl_prior_grad", "kl_posterior_grad"):
            if key in final_info:
                outputs[key] = final_info[key]

        new_carry = GenerativeRecursiveReasoningModelInnerCarry(h=h.detach(), l=l.detach())
        return new_carry, outputs


class GenerativeRecursiveReasoningModel_ACTV1(nn.Module):
    """GRAM with stochastic high-level transitions and ACT-compatible carry."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = GenerativeRecursiveReasoningModelConfig(**config_dict)
        self.inner = GenerativeRecursiveReasoningModelInner(self.config)

    @property
    def puzzle_emb(self):
        return getattr(self.inner, "puzzle_emb", None)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return GenerativeRecursiveReasoningModelCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: GenerativeRecursiveReasoningModelCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[GenerativeRecursiveReasoningModelCarry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        new_inner_carry, outputs = self.inner(new_inner_carry, new_current_data)

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step
            if not self.training and self.config.halt_max_steps > 1:
                halted = halted | (outputs["q_halt_logits"] > 0)

        return GenerativeRecursiveReasoningModelCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
