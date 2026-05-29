import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim

from training import create_dataloaders, Trainer
from hybrid_model import create_model, model_summary


# =========================================================
# Loss
# =========================================================
class BoundaryAwareHybridLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, pred_change, pred_boundary, target):
        return self.bce(pred_change, target)


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Main
# =========================================================
def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser(description="VMambaChangeFFT Training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-GPU batch size (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective_bs = batch_size × grad_accum)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Mixed precision training (default: ON)")
    parser.add_argument("--no-amp", action="store_true", default=False,
                        help="Disable mixed precision")
    parser.add_argument("--save-features-every", type=int, default=10,
                        help="Save intermediate features every N epochs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pos-weight", type=float, default=5.0,
                        help="Positive class weight for BCE loss (handles class imbalance)")
    args = parser.parse_args()

    if args.no_amp:
        args.amp = False

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # ---- Config ----
    config = {
        "batch_size": args.batch_size,
        "in_channels": 3,
        "img_size": 256,
        "fft_channels": 48,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "checkpoint_vmamba": True,
    }

    # ---- Data ----
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # ---- Model ----
    model, _ = create_model(config)
    model.to(device)
    model_summary(model)

    # Initialize LazyConv2d layers with a dummy forward pass
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256, device=device)
        model(dummy, dummy)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ---- Optimizer ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Loss ----
    loss_fn = BoundaryAwareHybridLoss(
        pos_weight=torch.tensor([args.pos_weight], device=device)
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir="./checkpoints",
        exp_name="vmamba_changefft",
        use_amp=args.amp,
        amp_dtype=torch.float16,
        grad_accum_steps=args.grad_accum,
        save_features_every=args.save_features_every,
        log_mem=True,
    )

    # ---- Train + Evaluate ----
    print(f"\n{'='*60}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  LR: {args.lr}")
    print(f"  AMP: {args.amp}")
    print(f"  Pos weight: {args.pos_weight}")
    print(f"  Save features every: {args.save_features_every} epochs")
    print(f"{'='*60}\n")

    test_metrics = trainer.train(args.epochs)

    print(f"\n{'='*60}")
    print("🏁 ALL DONE!")
    print(f"   Best Val IoU: {trainer.best_iou:.4f}")
    print(f"   Test IoU:     {test_metrics['iou']:.4f}")
    print(f"   Test F1:      {test_metrics['f1']:.4f}")
    print(f"{'='*60}")
    print(f"\n📁 All outputs saved to: ./checkpoints/")
    print(f"   checkpoints/vmamba_changefft_best_model.pth  — best model")
    print(f"   checkpoints/training_curves.png              — loss & IoU curves")
    print(f"   checkpoints/training_log.json                — per-epoch metrics")
    print(f"   checkpoints/predictions/test_evaluation/     — TP/FP/TN/FN figures")
    print(f"   checkpoints/features/                        — intermediate features")


if __name__ == "__main__":
    main()
