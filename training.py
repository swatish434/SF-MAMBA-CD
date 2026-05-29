import os
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
# Use torch.cuda.amp — works on ALL PyTorch versions (1.6+)
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score


# =========================================================
# Dataset
# =========================================================
class MineNetCDDataset(Dataset):
    """MineNetCD256 (Hugging Face) Dataset."""

    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        pre_img = np.array(row["imageA"], dtype=np.float32) / 255.0
        post_img = np.array(row["imageB"], dtype=np.float32) / 255.0

        pre = torch.from_numpy(pre_img).permute(2, 0, 1).contiguous()
        post = torch.from_numpy(post_img).permute(2, 0, 1).contiguous()

        label = row["label"]
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        mask = torch.from_numpy(label).float().unsqueeze(0).contiguous()
        if mask.max() > 1.0:
            mask = mask / 255.0

        return {"pre_image": pre, "post_image": post, "mask": mask}


# =========================================================
# Dataloaders
# =========================================================
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(config):
    print("📦 Loading MineNetCD256 from Hugging Face...")
    ds = load_dataset("HZDR-FWGEL/MineNetCD256")

    train_dataset = MineNetCDDataset(ds["train"])
    val_dataset = MineNetCDDataset(ds["val"])
    test_dataset = MineNetCDDataset(ds["test"])

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    g = torch.Generator()
    g.manual_seed(42)
    nw = int(config.get("num_workers", 4))
    pm = bool(config.get("pin_memory", True))
    bs = config["batch_size"]

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=pm, drop_last=True,
        persistent_workers=(nw > 0), worker_init_fn=_seed_worker, generator=g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pm,
        persistent_workers=(nw > 0), worker_init_fn=_seed_worker, generator=g,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pm,
        persistent_workers=(nw > 0), worker_init_fn=_seed_worker, generator=g,
    )

    return train_loader, val_loader, test_loader


# =========================================================
# Confusion map visualization
# =========================================================
def make_confusion_overlay(gt, pred):
    """
    Color-coded confusion map:
      TP = GREEN, TN = WHITE, FP = RED, FN = BLUE
    """
    H, W = gt.shape
    overlay = np.ones((H, W, 3), dtype=np.float32)

    overlay[(gt == 1) & (pred == 1)] = [0.0, 0.8, 0.0]   # TP green
    overlay[(gt == 0) & (pred == 0)] = [1.0, 1.0, 1.0]   # TN white
    overlay[(gt == 0) & (pred == 1)] = [0.9, 0.0, 0.0]   # FP red
    overlay[(gt == 1) & (pred == 0)] = [0.0, 0.0, 0.9]   # FN blue

    return overlay


# =========================================================
# Trainer
# =========================================================
class Trainer:
    def __init__(
        self, model, train_loader, val_loader, test_loader,
        loss_fn, optimizer, scheduler, device,
        save_dir, exp_name,
        use_amp=True, amp_dtype=torch.float16,
        grad_accum_steps=1, log_mem=False,
        save_features_every=10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.use_amp = bool(use_amp) and (device.type == "cuda")
        self.amp_dtype = amp_dtype
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.log_mem = bool(log_mem)
        self.save_features_every = save_features_every
        self.scaler = GradScaler(enabled=self.use_amp)

        self.best_iou = 0.0
        self.save_dir = save_dir
        self.exp_name = exp_name
        os.makedirs(save_dir, exist_ok=True)

        self.features_dir = os.path.join(save_dir, "features")
        self.predictions_dir = os.path.join(save_dir, "predictions")
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)

        # Metrics log
        self.metrics_log = []

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------
    def validate(self, epoch):
        self.model.eval()
        all_preds, all_targets, val_losses = [], [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"  Val Ep{epoch}", leave=False):
                pre = batch["pre_image"].to(self.device, non_blocking=True)
                post = batch["post_image"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    pred_change, pred_boundary = self.model(pre, post)
                    loss = self.loss_fn(pred_change, pred_boundary, mask)

                val_losses.append(loss.item())
                probs = torch.sigmoid(pred_change)
                preds = (probs > 0.5).long().cpu().numpy().flatten()
                targets = mask.cpu().numpy().flatten().astype(int)
                all_preds.append(preds)
                all_targets.append(targets)

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Pixel-level confusion matrix
        tp = int(((all_targets == 1) & (all_preds == 1)).sum())
        tn = int(((all_targets == 0) & (all_preds == 0)).sum())
        fp = int(((all_targets == 0) & (all_preds == 1)).sum())
        fn = int(((all_targets == 1) & (all_preds == 0)).sum())

        metrics = {
            'val_loss': float(np.mean(val_losses)),
            'iou': jaccard_score(all_targets, all_preds, zero_division=0),
            'f1': f1_score(all_targets, all_preds, zero_division=0),
            'precision': precision_score(all_targets, all_preds, zero_division=0),
            'recall': recall_score(all_targets, all_preds, zero_division=0),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        }

        print(f"  Val Loss: {metrics['val_loss']:.4f} | IoU: {metrics['iou']:.4f} | "
              f"F1: {metrics['f1']:.4f} | Prec: {metrics['precision']:.4f} | "
              f"Rec: {metrics['recall']:.4f}")
        print(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

        return metrics

    # -------------------------------------------------
    # Save intermediate features
    # -------------------------------------------------
    def save_intermediate_features(self, epoch):
        self.model.eval()
        epoch_dir = os.path.join(self.features_dir, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 3:
                    break

                pre = batch["pre_image"].to(self.device, non_blocking=True)
                post = batch["post_image"].to(self.device, non_blocking=True)
                mask = batch["mask"]

                with autocast(enabled=self.use_amp):
                    pred_change, pred_boundary, features = self.model.forward_with_features(pre, post)

                for feat_name, feat_tensor in features.items():
                    feat_np = feat_tensor[0].numpy()
                    np.save(os.path.join(epoch_dir, f"sample_{i}_{feat_name}.npy"), feat_np)

                    feat_mean = feat_np.mean(axis=0)
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    ax.imshow(feat_mean, cmap='viridis')
                    ax.set_title(f"{feat_name} (epoch {epoch})")
                    ax.axis('off')
                    fig.savefig(os.path.join(epoch_dir, f"sample_{i}_{feat_name}.png"),
                                bbox_inches='tight', dpi=100)
                    plt.close(fig)

                # Prediction visualization with confusion map
                probs = torch.sigmoid(pred_change[0, 0]).cpu().numpy()
                pred_binary = (probs > 0.5).astype(np.uint8)
                gt = (mask[0, 0].numpy() > 0.5).astype(np.uint8)

                pre_img = pre[0].cpu().permute(1, 2, 0).numpy()
                post_img = post[0].cpu().permute(1, 2, 0).numpy()
                confusion = make_confusion_overlay(gt, pred_binary)

                fig, axes = plt.subplots(1, 5, figsize=(25, 5))
                axes[0].imshow(pre_img); axes[0].set_title("Pre"); axes[0].axis('off')
                axes[1].imshow(post_img); axes[1].set_title("Post"); axes[1].axis('off')
                axes[2].imshow(gt, cmap='gray', vmin=0, vmax=1); axes[2].set_title("GT"); axes[2].axis('off')
                axes[3].imshow(pred_binary, cmap='gray', vmin=0, vmax=1); axes[3].set_title("Pred"); axes[3].axis('off')
                axes[4].imshow(confusion); axes[4].set_title("TP/FP/TN/FN"); axes[4].axis('off')

                legend = [
                    mpatches.Patch(color=[0,0.8,0], label='TP'),
                    mpatches.Patch(color=[1,1,1], label='TN', edgecolor='gray'),
                    mpatches.Patch(color=[0.9,0,0], label='FP'),
                    mpatches.Patch(color=[0,0,0.9], label='FN'),
                ]
                fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=10)
                fig.suptitle(f"Epoch {epoch} — Sample {i}")
                fig.savefig(os.path.join(epoch_dir, f"sample_{i}_prediction.png"),
                            bbox_inches='tight', dpi=100)
                plt.close(fig)

        print(f"  📸 Features saved to {epoch_dir}")

    # -------------------------------------------------
    # Final evaluation on test set with TP/FP/TN/FN
    # -------------------------------------------------
    def evaluate_test_set(self):
        """Full test set evaluation with metrics + confusion map visualizations."""
        print("\n" + "=" * 60)
        print("📊 FINAL TEST SET EVALUATION")
        print("=" * 60)

        # Load best model
        best_path = os.path.join(self.save_dir, f"{self.exp_name}_best_model.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded best model from: {best_path}")
        else:
            print("  ⚠️ No best model found, using current weights")

        self.model.eval()
        eval_dir = os.path.join(self.predictions_dir, "test_evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        all_preds, all_targets = [], []

        # --- Find 3 changed + 2 unchanged samples for visualization ---
        changed_viz, unchanged_viz = [], []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
                pre = batch["pre_image"].to(self.device, non_blocking=True)
                post = batch["post_image"].to(self.device, non_blocking=True)
                mask = batch["mask"]

                with autocast(enabled=self.use_amp):
                    pred_change, _ = self.model(pre, post)

                probs = torch.sigmoid(pred_change).cpu()

                for b in range(pre.shape[0]):
                    idx = i * self.test_loader.batch_size + b
                    prob = probs[b, 0].numpy()
                    pred_bin = (prob > 0.5).astype(np.uint8)
                    gt = (mask[b, 0].numpy() > 0.5).astype(np.uint8)

                    all_preds.append(pred_bin.flatten())
                    all_targets.append(gt.flatten())

                    change_pct = gt.sum() / gt.size * 100

                    # Collect samples for visualization
                    if change_pct > 5.0 and len(changed_viz) < 3:
                        changed_viz.append({
                            'idx': idx, 'pre': pre[b].cpu(), 'post': post[b].cpu(),
                            'gt': gt, 'pred': pred_bin, 'prob': prob, 'change_pct': change_pct
                        })
                    elif change_pct == 0.0 and len(unchanged_viz) < 2:
                        unchanged_viz.append({
                            'idx': idx, 'pre': pre[b].cpu(), 'post': post[b].cpu(),
                            'gt': gt, 'pred': pred_bin, 'prob': prob, 'change_pct': change_pct
                        })

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # --- Compute overall test metrics ---
        tp = int(((all_targets == 1) & (all_preds == 1)).sum())
        tn = int(((all_targets == 0) & (all_preds == 0)).sum())
        fp = int(((all_targets == 0) & (all_preds == 1)).sum())
        fn = int(((all_targets == 1) & (all_preds == 0)).sum())

        test_metrics = {
            'iou': jaccard_score(all_targets, all_preds, zero_division=0),
            'f1': f1_score(all_targets, all_preds, zero_division=0),
            'precision': precision_score(all_targets, all_preds, zero_division=0),
            'recall': recall_score(all_targets, all_preds, zero_division=0),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'overall_accuracy': (tp + tn) / (tp + tn + fp + fn) * 100,
        }

        print(f"\n  TEST RESULTS:")
        print(f"  IoU:       {test_metrics['iou']:.4f}")
        print(f"  F1:        {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  Accuracy:  {test_metrics['overall_accuracy']:.2f}%")
        print(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

        # Save metrics as JSON
        with open(os.path.join(eval_dir, "test_metrics.json"), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\n  Metrics saved to {eval_dir}/test_metrics.json")

        # --- Generate 5-sample visualization (3 changed + 2 unchanged) ---
        viz_samples = changed_viz + unchanged_viz
        if len(viz_samples) > 0:
            self._generate_5sample_figure(viz_samples, eval_dir)
            self._generate_individual_figures(viz_samples, eval_dir)

        return test_metrics

    def _generate_5sample_figure(self, viz_samples, eval_dir):
        """Generate combined 5×4 grid figure like the MineNetCD paper."""
        n = len(viz_samples)
        fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, s in enumerate(viz_samples):
            pre_img = s['pre'].permute(1, 2, 0).numpy()
            post_img = s['post'].permute(1, 2, 0).numpy()
            confusion = make_confusion_overlay(s['gt'], s['pred'])

            tp = int(((s['gt'] == 1) & (s['pred'] == 1)).sum())
            tn = int(((s['gt'] == 0) & (s['pred'] == 0)).sum())
            fp = int(((s['gt'] == 0) & (s['pred'] == 1)).sum())
            fn = int(((s['gt'] == 1) & (s['pred'] == 0)).sum())

            is_changed = s['change_pct'] > 0
            tag = "CHANGED" if is_changed else "NO CHANGE"

            axes[row, 0].imshow(pre_img); axes[row, 0].axis('off')
            axes[row, 1].imshow(post_img); axes[row, 1].axis('off')
            axes[row, 2].imshow(s['gt'], cmap='gray', vmin=0, vmax=1); axes[row, 2].axis('off')
            axes[row, 3].imshow(confusion); axes[row, 3].axis('off')

            axes[row, 0].set_ylabel(f"#{s['idx']}\n[{tag}]",
                                     fontsize=11, fontweight='bold',
                                     rotation=0, labelpad=80, va='center')

            stats = f"TP:{tp} FP:{fp}\nTN:{tn} FN:{fn}"
            axes[row, 3].text(5, s['gt'].shape[0] - 5, stats, fontsize=9,
                               color='black', backgroundcolor='white',
                               verticalalignment='bottom', fontfamily='monospace',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        col_titles = ["Pre-Change", "Post-Change", "Ground Truth", "VMambaChangeFFT (Ours)"]
        for j, t in enumerate(col_titles):
            axes[0, j].set_title(t, fontsize=13, fontweight='bold', pad=10)

        legend = [
            mpatches.Patch(color=[0, 0.8, 0], label='TP (True Positive)'),
            mpatches.Patch(color=[1, 1, 1], label='TN (True Negative)', edgecolor='gray'),
            mpatches.Patch(color=[0.9, 0, 0], label='FP (False Positive)'),
            mpatches.Patch(color=[0, 0, 0.9], label='FN (False Negative)'),
        ]
        fig.legend(handles=legend, loc='lower center', ncol=4,
                   fontsize=12, frameon=True, edgecolor='black',
                   bbox_to_anchor=(0.5, 0.0))

        fig.suptitle("VMambaChangeFFT — Qualitative Results on MineNetCD256",
                     fontsize=16, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0.08, 0.03, 1, 0.96])

        path = os.path.join(eval_dir, "summary_5samples.png")
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  📊 Summary figure: {path}")

    def _generate_individual_figures(self, viz_samples, eval_dir):
        """Generate individual sample figures."""
        for s in viz_samples:
            pre_img = s['pre'].permute(1, 2, 0).numpy()
            post_img = s['post'].permute(1, 2, 0).numpy()
            confusion = make_confusion_overlay(s['gt'], s['pred'])

            tp = int(((s['gt'] == 1) & (s['pred'] == 1)).sum())
            tn = int(((s['gt'] == 0) & (s['pred'] == 0)).sum())
            fp = int(((s['gt'] == 0) & (s['pred'] == 1)).sum())
            fn = int(((s['gt'] == 1) & (s['pred'] == 0)).sum())
            total = s['gt'].size
            tag = "CHANGED" if s['change_pct'] > 0 else "NO_CHANGE"

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(pre_img); axes[0].set_title("(a) Pre-Change", fontsize=12); axes[0].axis('off')
            axes[1].imshow(post_img); axes[1].set_title("(b) Post-Change", fontsize=12); axes[1].axis('off')
            axes[2].imshow(s['gt'], cmap='gray', vmin=0, vmax=1)
            axes[2].set_title(f"(c) GT ({s['change_pct']:.1f}%)", fontsize=12); axes[2].axis('off')
            axes[3].imshow(confusion); axes[3].set_title("(d) Ours", fontsize=12); axes[3].axis('off')

            legend = [
                mpatches.Patch(color=[0,0.8,0], label=f'TP: {tp} ({tp/total*100:.1f}%)'),
                mpatches.Patch(color=[1,1,1], label=f'TN: {tn} ({tn/total*100:.1f}%)', edgecolor='gray'),
                mpatches.Patch(color=[0.9,0,0], label=f'FP: {fp} ({fp/total*100:.1f}%)'),
                mpatches.Patch(color=[0,0,0.9], label=f'FN: {fn} ({fn/total*100:.1f}%)'),
            ]
            fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=11,
                       frameon=True, edgecolor='black', bbox_to_anchor=(0.5, -0.02))
            fig.suptitle(f"Sample {s['idx']} [{tag}]", fontsize=14, fontweight='bold')
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])

            path = os.path.join(eval_dir, f"sample_{s['idx']}_{tag.lower()}.png")
            fig.savefig(path, bbox_inches='tight', dpi=150)
            plt.close(fig)

        print(f"  📸 Individual figures saved to {eval_dir}")

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    def train(self, epochs):
        print("🚀 Starting training...")
        if self.use_amp:
            print(f"⚡ AMP enabled: dtype={self.amp_dtype}, grad_accum={self.grad_accum_steps}")

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            self.optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for step, batch in enumerate(pbar):
                pre = batch["pre_image"].to(self.device, non_blocking=True)
                post = batch["post_image"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    pred_change, pred_boundary = self.model(pre, post)
                    loss = self.loss_fn(pred_change, pred_boundary, mask)
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                epoch_losses.append(loss.item() * self.grad_accum_steps)
                pbar.set_postfix(loss=float(loss.item() * self.grad_accum_steps))

                if self.log_mem and self.device.type == "cuda" and (step % 200 == 0):
                    torch.cuda.synchronize()
                    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    pbar.write(f"[mem] peak_alloc={peak_gb:.2f} GiB")

            # Handle leftover accumulation steps
            if len(self.train_loader) % self.grad_accum_steps != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            self.scheduler.step()

            avg_loss = float(np.mean(epoch_losses))
            print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_loss:.4f}")

            # Validation
            val_metrics = self.validate(epoch + 1)
            val_metrics['epoch'] = epoch + 1
            val_metrics['train_loss'] = avg_loss
            self.metrics_log.append(val_metrics)

            # Save best model (by validation IoU)
            if val_metrics['iou'] > self.best_iou:
                self.best_iou = val_metrics['iou']
                self.save_checkpoint(epoch + 1, avg_loss, val_metrics)

            # Save intermediate features periodically
            if (epoch + 1) % self.save_features_every == 0:
                self.save_intermediate_features(epoch + 1)

        # Save training metrics log
        log_path = os.path.join(self.save_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)

        # Save loss + IoU curves
        self._save_training_curves()

        # Final test set evaluation with TP/FP/TN/FN
        print("\n🏁 Training finished. Running final test evaluation...")
        test_metrics = self.evaluate_test_set()

        return test_metrics

    # -------------------------------------------------
    # Training curves
    # -------------------------------------------------
    def _save_training_curves(self):
        epochs = [m['epoch'] for m in self.metrics_log]
        train_losses = [m['train_loss'] for m in self.metrics_log]
        val_losses = [m['val_loss'] for m in self.metrics_log]
        ious = [m['iou'] for m in self.metrics_log]
        f1s = [m['f1'] for m in self.metrics_log]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, ious, 'g-', label='IoU')
        axes[1].plot(epochs, f1s, 'm-', label='F1')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Score')
        axes[1].set_title('Validation IoU & F1')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(self.save_dir, "training_curves.png")
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  📈 Training curves: {path}")

    # -------------------------------------------------
    # Checkpointing
    # -------------------------------------------------
    def save_checkpoint(self, epoch, loss, val_metrics=None):
        path = os.path.join(self.save_dir, f"{self.exp_name}_best_model.pth")
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        if val_metrics:
            ckpt["val_metrics"] = val_metrics
        torch.save(ckpt, path)
        print(f"  ✓ Saved best model (IoU={self.best_iou:.4f}) → {path}")
