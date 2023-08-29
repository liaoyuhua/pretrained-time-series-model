import time
import torch
from torch.optim import Adam
import torch.nn.functional as F


class Trainer:
    """
    TODO: support lr scheduler, auto mixed precision
    """

    def __init__(self, model, lr, max_epochs) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.lr = lr
        self.max_epochs = max_epochs

        self.optimizer = Adam(model.parameters(), lr=lr)

    def train(
        self,
        batch_size,
        train_dataset,
        val_dataset=None,
        num_workers=None,
        save_path=None,
        save_every=10,
    ):
        train_loader = train_dataset.to_dataloader(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        if val_dataset is not None:
            val_loader = val_dataset.to_dataloader(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        for epoch in range(self.max_epochs):
            start_time = time.time()
            train_loss = self.train_epoch(train_loader)

            if val_dataset is not None:
                val_loss = self.val_epoch(val_loader)
            else:
                val_loss = None

            if save_path is not None and (epoch + 1) % save_every == 0:
                torch.save(
                    self.model.state_dict(),
                    save_path + f"epoch_{epoch + 1}.pth",
                )

            if val_loss is not None:
                print(
                    f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time() - start_time:.2f}s"
                )
            else:
                print(
                    f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Time: {time.time() - start_time:.2f}s"
                )

    def train_epoch(self, train_loader):
        self.model.train()

        total_loss = 0
        for batch in train_loader:
            target = batch["target"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            pred, idx = self.model(target, mask)

            pred_idx = torch.zeros_like(target.squeeze(-1)).to(self.device)
            pred_idx[:, idx] = 1
            pred_idx = ~mask.bool() & pred_idx.bool()

            loss = F.l1_loss(pred[pred_idx], target[pred_idx])

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def val_epoch(self, val_loader):
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                target = batch["target"].to(self.device)
                mask = batch["mask"].to(self.device)

                pred, idx = self.model(target, mask)
                pred_idx = torch.zeros_like(target).to(self.device)
                pred_idx[:, idx] = 1
                pred_idx = ~mask.bool() & pred_idx.bool()

                loss = F.l1_loss(pred[pred_idx], target[pred_idx])

                total_loss += loss.item()

        return total_loss / len(val_loader)
