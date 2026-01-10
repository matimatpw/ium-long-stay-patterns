import torch
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
from ..config import set_seed

class Trainer:
    def __init__(self, model, criterion, optimizer, epochs=100, device=None, seed: int = None):
        # Optionally seed RNGs for reproducible training runs
        if seed is not None:
            set_seed(seed)

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.history = {'train_loss': [], 'val_auc': []}

    def train(self, train_loader, val_loader):
        """Modified to iterate through DataLoaders."""
        logger.info(f"Training started on device: {self.device}")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            # Progress bar for batches within the epoch
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}", leave=False)

            for batch_X, batch_y in pbar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            self.history['train_loss'].append(avg_loss)

            # Every 10 epochs, log advanced metrics using the validation loader
            if epoch % 10 == 0 or epoch == 1:
                metrics = self._validate(val_loader)
                self.history['val_auc'].append(metrics['auc'])

                logger.info(
                    f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f} | "
                    f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | "
                    f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}"
                )

    def _validate(self, loader):
        """Modified to collect predictions across all batches in the loader."""
        self.model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                probs = self.model(batch_X).cpu().numpy()

                all_probs.extend(probs)
                all_targets.extend(batch_y.numpy())

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        preds = (all_probs > 0.5).astype(float)

        return {
            'auc': roc_auc_score(all_targets, all_probs),
            'f1': f1_score(all_targets, preds),
            'precision': precision_score(all_targets, preds, zero_division=0),
            'recall': recall_score(all_targets, preds, zero_division=0)
        }

    def save_model(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path):
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"No model found at {path}")
