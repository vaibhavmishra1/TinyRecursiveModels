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

    The model forward performs one supervision step containing T stochastic transitions.
    This loss head repeats it N_sup times, applies reconstruction and the final-step KL
    of each supervision step, and adds the ACT and LPRM auxiliary losses from Appendix A.
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
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.deep_supervision_steps = deep_supervision_steps
        self.beta = beta
        self.kl_balance = kl_balance
        self.act_loss_weight = act_loss_weight
        self.lprm_loss_weight = lprm_loss_weight

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

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        carry = model_kwargs["carry"]
        batch = model_kwargs["batch"]

        recon_loss = 0
        kl_loss = 0
        lprm_loss = 0
        act_q_halt_logits = []
        act_q_continue_logits = []
        seq_correct_by_step = []
        token_accuracy_by_step = []
        last_outputs = None
        last_preds = None
        last_loss_counts = None

        for _step in range(self.deep_supervision_steps):
            carry, outputs = self.model(carry=carry, batch=batch)
            labels = carry.current_data["labels"]
            step_recon, preds, token_accuracy, seq_is_correct, loss_counts = self._reconstruction_loss(outputs["logits"], labels)
            recon_loss = recon_loss + step_recon

            if "kl_prior_grad" in outputs and "kl_posterior_grad" in outputs:
                balanced_kl = self.kl_balance * outputs["kl_prior_grad"] + (1.0 - self.kl_balance) * outputs["kl_posterior_grad"]
                kl_loss = kl_loss + balanced_kl.sum()

            reward = token_accuracy.detach()
            for v_logits in outputs.get("transition_v_logits", [outputs["v_logits"]]):
                v_pred = torch.sigmoid(v_logits)
                lprm_loss = lprm_loss + F.mse_loss(v_pred, reward.to(v_pred.dtype), reduction="sum")

            act_q_halt_logits.append(outputs["q_halt_logits"])
            act_q_continue_logits.append(outputs["q_continue_logits"])
            seq_correct_by_step.append(seq_is_correct.to(outputs["q_halt_logits"].dtype).detach())
            token_accuracy_by_step.append(token_accuracy.detach())
            last_outputs = outputs
            last_preds = preds
            last_loss_counts = loss_counts

        assert last_outputs is not None and last_preds is not None and last_loss_counts is not None

        recon_loss = recon_loss / self.deep_supervision_steps
        kl_loss = kl_loss / self.deep_supervision_steps
        lprm_loss = lprm_loss / self.deep_supervision_steps

        act_loss = 0
        for i, (q_halt, q_continue) in enumerate(zip(act_q_halt_logits, act_q_continue_logits)):
            halt_target = seq_correct_by_step[i]
            if i + 1 < len(act_q_halt_logits):
                continue_target = torch.maximum(act_q_halt_logits[i + 1].detach(), act_q_continue_logits[i + 1].detach())
            else:
                continue_target = halt_target
            act_loss = act_loss + F.mse_loss(q_halt, halt_target, reduction="sum")
            act_loss = act_loss + F.mse_loss(q_continue, continue_target.to(q_continue.dtype), reduction="sum")
        act_loss = act_loss / self.deep_supervision_steps

        total_loss = recon_loss + self.beta * kl_loss + self.act_loss_weight * act_loss + self.lprm_loss_weight * lprm_loss

        with torch.no_grad():
            valid_metrics = last_loss_counts > 0
            count = valid_metrics.sum()
            exact = seq_correct_by_step[-1]
            metrics = {
                "count": count,
                "accuracy": torch.where(valid_metrics, token_accuracy_by_step[-1], 0).sum(),
                "exact_accuracy": (valid_metrics & exact.to(torch.bool)).sum(),
                "lm_loss": recon_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "act_loss": torch.as_tensor(act_loss).detach(),
                "lprm_loss": torch.as_tensor(lprm_loss).detach(),
                "q_halt_accuracy": (valid_metrics & ((last_outputs["q_halt_logits"] >= 0) == exact.to(torch.bool))).sum(),
                "steps": torch.where(valid_metrics, carry.steps, 0).sum(),
                "prior_std": last_outputs["prior_std"].to(torch.float32).mean(),
                "sample_std": last_outputs["sample_std"].to(torch.float32).mean(),
            }
            last_outputs["preds"] = last_preds

        detached_outputs = {k: last_outputs[k].detach() for k in return_keys if k in last_outputs}
        return carry, total_loss, metrics, detached_outputs, carry.halted.all()
