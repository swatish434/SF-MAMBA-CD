import argparse
import torch
import random
import numpy as np
import torch.optim as optim
from tqdm import tqdm

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
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Using device: {device}")

    config = {
        "batch_size": args.batch_size,
        "in_channels": 3,
        "embed_dims": [96, 192, 384, 768]
    }

    train_loader, val_loader, test_loader = create_dataloaders(config)

    model, _ = create_model(config)
    model.to(device)
    model_summary(model)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = BoundaryAwareHybridLoss(
        pos_weight=torch.tensor([2.0], device=device)
    )

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        loss_fn,
        optimizer,
        scheduler,
        device,
        "./checkpoints",
        "vmamba_changefft"
    )

    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
