from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


class GRAMLossHead(nn.Module):
    """Paper-faithful GRAM surrogate objective with deep supervision.

    The trainer keeps the recurrent carry between batches, so each optimizer
    step advances one supervision step. The model's halt_max_steps controls the
    N_sup reset cadence, matching the ACT-style training loop in this codebase.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_type: str,
        deep_supervision_steps: int = 16,
        beta: float = 0.1,
        kl_balance: float = 0.8,
        act_loss_weight: float = 1.0,
        lprm_loss_weight: float = 1.0,
        lprm_reward_type: str = "token_accuracy",
        prior_lprm_loss_weight: float = 0.0,
        prior_aux_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.deep_supervision_steps = deep_supervision_steps
        self.beta = beta
        self.kl_balance = kl_balance
        self.act_loss_weight = act_loss_weight
        self.lprm_loss_weight = lprm_loss_weight
        self.lprm_reward_type = lprm_reward_type
        self.prior_lprm_loss_weight = prior_lprm_loss_weight
        self.prior_aux_loss_weight = prior_aux_loss_weight

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def _reconstruction_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
        loss = (self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            token_accuracy = (is_correct.to(torch.float32) / loss_divisor).sum(-1)

        return loss, preds, token_accuracy, seq_is_correct, loss_counts

    def _nqueens_stats(self, preds: torch.Tensor, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq_len = preds.shape[-1]
        n = math.isqrt(seq_len)
        if n * n != seq_len:
            raise ValueError(f"N-Queens reward expects a square board, got seq_len={seq_len}")

        queens = (preds == 2).view(preds.shape[0], n, n)
        queen_count = queens.sum(dim=(1, 2)).to(torch.float32)

        row_counts = queens.sum(dim=2)
        col_counts = queens.sum(dim=1)
        conflicts = (
            (row_counts * (row_counts - 1) // 2).sum(dim=1)
            + (col_counts * (col_counts - 1) // 2).sum(dim=1)
        )
        for offset in range(-(n - 1), n):
            diag_count = queens.diagonal(offset=offset, dim1=1, dim2=2).sum(dim=1)
            anti_diag_count = queens.flip(2).diagonal(offset=offset, dim1=1, dim2=2).sum(dim=1)
            conflicts = conflicts + diag_count * (diag_count - 1) // 2
            conflicts = conflicts + anti_diag_count * (anti_diag_count - 1) // 2

        clue_violations = ((inputs == 2) & (preds != 2)).sum(dim=1).to(torch.float32)
        conflicts = conflicts.to(torch.float32)
        return {
            "queen_count": queen_count,
            "conflicts": conflicts,
            "clue_violations": clue_violations,
        }

    def _nqueens_reward(self, preds: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        stats = self._nqueens_stats(preds, inputs)
        seq_len = preds.shape[-1]
        n = math.isqrt(seq_len)
        max_conflicts = max(n * (n - 1) / 2.0, 1.0)

        queen_score = (1.0 - (stats["queen_count"] - n).abs() / max(n, 1)).clamp(0.0, 1.0)
        conflict_score = (1.0 - stats["conflicts"] / max_conflicts).clamp(0.0, 1.0)
        clue_score = (1.0 - stats["clue_violations"] / max(n, 1)).clamp(0.0, 1.0)
        valid = (stats["queen_count"] == n) & (stats["conflicts"] == 0) & (stats["clue_violations"] == 0)

        dense_reward = 0.45 * queen_score + 0.45 * conflict_score + 0.10 * clue_score
        return torch.where(valid, torch.ones_like(dense_reward), 0.5 * dense_reward)

    def _lprm_reward(self, preds: torch.Tensor, token_accuracy: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        if self.lprm_reward_type == "token_accuracy":
            return token_accuracy.detach()
        if self.lprm_reward_type == "nqueens":
            return self._nqueens_reward(preds.detach(), inputs).detach()
        raise ValueError(f"Unsupported lprm_reward_type={self.lprm_reward_type}")

    def _lprm_loss(self, outputs: Dict[str, torch.Tensor], reward: torch.Tensor) -> torch.Tensor:
        lprm_loss = torch.zeros((), dtype=outputs["v_logits"].dtype, device=outputs["v_logits"].device)
        for v_logits in outputs.get("transition_v_logits", [outputs["v_logits"]]):
            v_pred = torch.sigmoid(v_logits)
            lprm_loss = lprm_loss + F.mse_loss(v_pred, reward.to(v_pred.dtype), reduction="sum")
        return lprm_loss

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        carry, outputs = self.model(**model_kwargs)
        labels = carry.current_data["labels"]
        inputs = carry.current_data["inputs"]
        recon_loss, preds, token_accuracy, seq_is_correct, loss_counts = self._reconstruction_loss(outputs["logits"], labels)

        kl_loss = torch.zeros((), dtype=recon_loss.dtype, device=recon_loss.device)
        if "kl_prior_grad" in outputs and "kl_posterior_grad" in outputs:
            balanced_kl = self.kl_balance * outputs["kl_prior_grad"] + (1.0 - self.kl_balance) * outputs["kl_posterior_grad"]
            kl_loss = balanced_kl.sum()

        reward = self._lprm_reward(preds, token_accuracy, inputs)
        lprm_loss = self._lprm_loss(outputs, reward).to(recon_loss.dtype)

        prior_lprm_loss = torch.zeros((), dtype=recon_loss.dtype, device=recon_loss.device)
        prior_recon_loss = torch.zeros((), dtype=recon_loss.dtype, device=recon_loss.device)
        prior_reward = torch.zeros_like(reward)
        prior_seq_is_correct = torch.zeros_like(seq_is_correct)
        prior_stats = None
        if self.prior_lprm_loss_weight > 0 or self.prior_aux_loss_weight > 0:
            prior_carry, prior_outputs = self.model(
                carry=model_kwargs["carry"],
                batch=model_kwargs["batch"],
                force_prior=True,
            )
            prior_labels = prior_carry.current_data["labels"]
            prior_inputs = prior_carry.current_data["inputs"]
            prior_recon_loss, prior_preds, prior_token_accuracy, prior_seq_is_correct, _prior_loss_counts = self._reconstruction_loss(
                prior_outputs["logits"],
                prior_labels,
            )
            prior_reward = self._lprm_reward(prior_preds, prior_token_accuracy, prior_inputs)
            prior_lprm_loss = self._lprm_loss(prior_outputs, prior_reward).to(recon_loss.dtype)
            if self.lprm_reward_type == "nqueens":
                prior_stats = self._nqueens_stats(prior_preds.detach(), prior_inputs)

        halt_target = seq_is_correct.to(outputs["q_halt_logits"].dtype).detach()
        act_loss = F.mse_loss(outputs["q_halt_logits"], halt_target, reduction="sum")

        total_loss = (
            recon_loss
            + self.beta * kl_loss
            + self.act_loss_weight * act_loss
            + self.lprm_loss_weight * lprm_loss
            + self.prior_lprm_loss_weight * prior_lprm_loss
            + self.prior_aux_loss_weight * prior_recon_loss
        )

        with torch.no_grad():
            valid_metrics = loss_counts > 0
            count = valid_metrics.sum()
            exact = seq_is_correct.detach()
            metrics = {
                "count": count,
                "accuracy": torch.where(valid_metrics, token_accuracy, 0).sum(),
                "exact_accuracy": (valid_metrics & exact.to(torch.bool)).sum(),
                "lm_loss": recon_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "act_loss": torch.as_tensor(act_loss).detach(),
                "lprm_loss": torch.as_tensor(lprm_loss).detach(),
                "lprm_reward": reward.sum(),
                "prior_lprm_loss": torch.as_tensor(prior_lprm_loss).detach(),
                "prior_lm_loss": torch.as_tensor(prior_recon_loss).detach(),
                "prior_lprm_reward": prior_reward.sum(),
                "prior_exact_accuracy": (valid_metrics & prior_seq_is_correct.to(torch.bool)).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == exact.to(torch.bool))).sum(),
                "steps": torch.where(valid_metrics, carry.steps, 0).sum(),
                "prior_std": outputs["prior_std"].to(torch.float32).mean(),
                "sample_std": outputs["sample_std"].to(torch.float32).mean(),
                "prior_log_std_mean": outputs["prior_log_std_mean"].to(torch.float32).mean(),
                "prior_log_std_max": outputs["prior_log_std_max"].to(torch.float32).mean(),
                "sample_log_std_mean": outputs["sample_log_std_mean"].to(torch.float32).mean(),
                "sample_log_std_max": outputs["sample_log_std_max"].to(torch.float32).mean(),
            }
            if prior_stats is not None:
                metrics.update(
                    {
                        "prior_queen_count": prior_stats["queen_count"].sum(),
                        "prior_conflicts": prior_stats["conflicts"].sum(),
                        "prior_clue_violations": prior_stats["clue_violations"].sum(),
                    }
                )
            outputs["preds"] = preds

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return carry, total_loss, metrics, detached_outputs, carry.halted.all()
