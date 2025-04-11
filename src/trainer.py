import os
import torch
import wandb
from src.utils import set_seed, plot_losses
from src.train import training_epoch, validation_epoch


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_epochs: int,
        optimizer,
        criterion,
        scheduler=None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        experiment_name: str = 'experiment',
        save_dir: str = 'checkpoints',
        use_wandb: bool = False,
        seed: int = 42,
        batch_augment_fn=None,
        scaler = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = torch.device(device)
        self.model.to(self.device)
        self.batch_augment_fn = batch_augment_fn

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.experiment_name = experiment_name

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        from torch.optim.lr_scheduler import OneCycleLR
        self.step_scheduler_per_batch = isinstance(scheduler, OneCycleLR)

        self.best_f1 = 0.0
        self.best_epoch = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': []
        }

        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=experiment_name,
                config={
                    'num_epochs': num_epochs,
                    'optimizer': str(optimizer),
                    'device': device,
                    'criterion': str(criterion),
                    'scheduler': str(scheduler),
                    'seed': seed
                }
            )

        set_seed(seed)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_f1': self.best_f1,
            'history': self.history
        }

        if is_best:
            path = os.path.join(self.save_dir, f'{self.experiment_name}_best.pth')
        else:
            path = os.path.join(self.save_dir, f'{self.experiment_name}_epoch{epoch}.pth')

        torch.save(checkpoint, path)

    def train(self, start_epoch: int = 1):
        for epoch in range(start_epoch, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")

            train_loss, train_f1 = training_epoch(
                self.model, self.optimizer, self.criterion,
                self.train_loader, self.device, f"Train {epoch}",
                batch_augment_fn=self.batch_augment_fn,
                scheduler=self.scheduler if self.step_scheduler_per_batch else None,
                scaler = self.scaler if self.scaler else None,
            )

            val_loss, val_f1 = validation_epoch(
                self.model, self.criterion,
                self.val_loader, self.device, f"Val {epoch}"
            )

            if self.scheduler is not None and not self.step_scheduler_per_batch:
                self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)

            current_lr = self.optimizer.param_groups[0]['lr']

            metrics = {
                'train/loss': train_loss,
                'train/f1': train_f1,
                'val/loss': val_loss,
                'val/f1': val_f1,
                'lr': current_lr,
                'epoch': epoch
            }

            if self.use_wandb:
                wandb.log(metrics)

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)

            self.save_checkpoint(epoch)

            plot_losses(
                self.history['train_loss'],
                self.history['val_loss'],
                self.history['train_f1'],
                self.history['val_f1'],
                clear=True
            )

        print(f"Training completed. Best Val F1: {self.best_f1:.4f} at epoch {self.best_epoch}")
        return self.history