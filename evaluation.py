import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def evaluate_dataset(self, dataloader, save_predictions=False, save_dir='./evaluation_results'):
        self.model.eval()
        all_preds = []
        all_targets = []
        os.makedirs(save_dir, exist_ok=True)
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
                pre_img = batch['pre_image'].to(self.device)
                post_img = batch['post_image'].to(self.device)
                mask = batch['mask'].to(self.device).float().unsqueeze(1)
                outputs_change, outputs_boundary = self.model(pre_img, post_img)  # Dual output, use change map for evaluation
                probs = torch.sigmoid(outputs_change)
                preds = (probs > 0.5).long()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(mask.cpu().numpy())
                if save_predictions:
                    for i in range(preds.shape[0]):  # Corrected indexing for batch dimension
                        plt.imsave(os.path.join(save_dir, f'pred_{idx*dataloader.batch_size+i}.png'), preds[i, 0].numpy(), cmap='gray')
                        plt.imsave(os.path.join(save_dir, f'boundary_{idx*dataloader.batch_size+i}.png'), torch.sigmoid(outputs_boundary[i, 0]).cpu().numpy() > 0.5, cmap='gray')
                        plt.imsave(os.path.join(save_dir, f'gt_{idx*dataloader.batch_size+i}.png'), mask[i, 0].cpu().numpy(), cmap='gray')
        all_preds = np.concatenate(all_preds, axis=0).reshape(-1)
        all_targets = np.concatenate(all_targets, axis=0).reshape(-1)
        metrics = {
            'precision': precision_score(all_targets, all_preds, zero_division=0),
            'recall': recall_score(all_targets, all_preds, zero_division=0),
            'f1': f1_score(all_targets, all_preds, zero_division=0),
            'accuracy': accuracy_score(all_targets, all_preds),
            'iou': jaccard_score(all_targets, all_preds, zero_division=0)
        }
        return metrics, all_preds, all_targets

    def analyze_predictions(self, preds, targets, save_dir='./evaluation_results'):
        # Compute confusion matrix components for detailed analysis
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        tp = np.sum((preds == 1) & (targets == 1))
        fp = np.sum((preds == 1) & (targets == 0))
        fn = np.sum((preds == 0) & (targets == 1))
        tn = np.sum((preds == 0) & (targets == 0))
        print('Evaluation Summary:')
        print(f'Total samples: {len(preds)}')
        print(f'True Positives (TP): {tp}')
        print(f'False Positives (FP): {fp}')
        print(f'False Negatives (FN): {fn}')
        print(f'True Negatives (TN): {tn}')
        # Optionally save confusion matrix visualization
        try:
            import seaborn as sns
            cm = np.array([[tn, fp], [fn, tp]])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
            plt.close()
            print("Confusion matrix saved to confusion_matrix.png")
        except ImportError:
            print("Seaborn not installed; skipping confusion matrix visualization.")
