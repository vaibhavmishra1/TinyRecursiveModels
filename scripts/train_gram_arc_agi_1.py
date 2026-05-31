from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_gram_common import GRAMTrainConfig, train


def main():
    # GRAM paper Appendix B.1/B.2:
    # D=512, 8 heads, two fL/fH blocks, T=3 high-level transitions,
    # K=4 low-level refinements for ARC-AGI, N_sup=16, beta=0.04 for
    # ARC-AGI-1, KL balance=0.8, AdamW lr=1e-4, weight decay=1.0,
    # grad clip=1.0, global batch size=768, EMA decay=0.9999, 200K epochs.
    config = GRAMTrainConfig(
        run_name="gram_arc_agi_1",
        data_paths=["data/arc1concept-aug-1000"],
        checkpoint_path="checkpoints/GRAM-ARC-AGI-1/gram_arc_agi_1",
        epochs=200000,
        eval_interval=10000,
        global_batch_size=768,
        lr=1e-4,
        puzzle_emb_lr=1e-4,
        weight_decay=1.0,
        puzzle_emb_weight_decay=0.1,
        grad_clip=1.0,
        ema=True,
        ema_rate=0.9999,
        hidden_size=512,
        num_heads=8,
        expansion=4,
        H_layers=2,
        L_layers=2,
        T_steps=3,
        K_steps=4,
        N_sup=16,
        puzzle_emb_ndim=512,
        puzzle_emb_len=16,
        pos_encodings="rope",
        forward_dtype="bfloat16",
        mlp_t=False,
        decoder_swiglu=True,
        beta=0.04,
        kl_balance=0.8,
        act_loss_weight=1.0,
        lprm_loss_weight=1.0,
        loss_type="stablemax_cross_entropy",
        seed=0,
        device="auto",
        num_workers=1,
        log_every=20,
        save_every_eval=True,
        compile=False,
    )
    train(config)


if __name__ == "__main__":
    main()
