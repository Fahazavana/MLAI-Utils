import json
import os
import time

from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
)

import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    SequentialLR,
    _LRScheduler,
)
from matplotlib.ticker import MaxNLocator, ScalarFormatter


class AETrainer:
    def __init__(
        self,
        encoder,
        decoder,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        name,
        device,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_size = len(self.train_loader.dataset)
        self.valid_size = len(self.valid_loader.dataset)
        self.train_loss = []
        self.valid_loss = []
        self.train_lr = []

    def _run_epoch(self, desc):
        lr = self.optimizer.param_groups[-1]["lr"]
        train_loss = 0.0
        valid_loss = 0.0
        self.encoder.train()
        self.decoder.train()
        for batch_idx, (inputs, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
            loss = self.criterion(decoded, inputs)
            loss.backward()
            nn.utils.clip_grad_value_(self.encoder.parameters(), 1, foreach=True)
            nn.utils.clip_grad_value_(self.decoder.parameters(), 1, foreach=True)
            self.optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            batch = f"{batch_idx+1}/{len(self.train_loader)}"
            msg = f"\r{desc:^20} batch:{batch:^10} | train_loss:{train_loss / self.train_size:>7.2e} | val_loss:{0.0:>7.2e} | lr:{lr:>7.1e}"
            print(msg, end="")
        train_loss /= self.train_size
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            for inputs, _ in self.valid_loader:
                inputs = inputs.to(self.device)
                encoded = self.encoder(inputs)
                decoded = self.decoder(encoded)
                loss = self.criterion(decoded, inputs)
                valid_loss += loss.item() * inputs.size(0)
                msg = f"\r{desc:^20} batch:{batch:^10} | train_loss:{train_loss:>7.2e} | val_loss:{valid_loss / self.valid_size:>7.2e} | lr:{lr:>7.1e}"
                print(msg, end="")

        print()
        return train_loss, valid_loss / self.valid_size

    def train(self, epochs: int, patience=10, delta=0.0, load_best=True):
        os.makedirs(self.name, exist_ok=True)
        no_improvement_counter = 0
        best_valid_loss = float("inf")

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=patience // 2, min_lr=1e-8
        )
        start_time = time.time()
        epochstr = str(epochs)
        nbdigit = len(epochstr)

        for epoch in range(epochs):
            self.train_lr.append(scheduler.get_last_lr())
            train_loss, valid_loss = self._run_epoch(
                desc=f"Epoch [{str(epoch +1).zfill(nbdigit)}/{epochstr}]"
            )
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)
            scheduler.step(valid_loss)
            if valid_loss < (best_valid_loss - delta):
                no_improvement_counter = 0
                best_valid_loss = valid_loss
                torch.save(self.encoder, f"{self.name}/{self.name}_encoder.pth")
                torch.save(self.decoder, f"{self.name}/{self.name}_decoder.pth")
            else:
                no_improvement_counter += 1
                if no_improvement_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        end_time = time.time()
        duration = end_time - start_time
        training_data = {
            "start": start_time,
            "end": end_time,
            "duration": duration,
            "validation_loss": self.valid_loss,
            "train_loss": self.train_loss,
            "train_lr": self.train_lr,
        }
        with open(f"{self.name}/ae_training_log.json", "w") as f:
            json.dump(training_data, f)
        if load_best:
            self.encoder.load_state_dict(
                torch.load(
                    f"{self.name}/{self.name}_encoder.pth", weights_only=False
                ).state_dict()
            )
            self.decoder.load_state_dict(
                torch.load(
                    f"{self.name}/{self.name}_decoder.pth", weights_only=False
                ).state_dict()
            )
            self.encoder.eval()
            self.decoder.eval()

    def plot_training_loss(self):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        fig, ax1 = plt.subplots()
        color = "tab:red"
        ax1.set_xlabel("Epochs")
        ax1.plot(self.train_loss, color="tab:blue", label="Training loss")
        ax1.plot(self.valid_loss, color="tab:orange", label="Validation loss")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.yaxis.set_major_formatter(formatter)
        plt.legend()
        ax2 = ax1.twinx()
        color = "tab:gray"
        ax2.set_ylabel("Learning Rate", color=color)
        ax2.plot(self.train_lr, "--", color=color, label="Learning Rate")
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.yaxis.set_major_formatter(formatter)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2

        ax1.legend(lines, labels)
        fig.tight_layout()
        plt.savefig(f"{self.name}/ae_training.pdf")
        plt.show()

class AESTrainer(AETrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def train(self, epochs: int, update_freq:int, patience=10, delta=0.0, load_best=True):
        os.makedirs(self.name, exist_ok=True)
        no_improvement_counter = 0
        best_valid_loss = float("inf")

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=patience // 2, min_lr=1e-8
        )
        start_time = time.time()
        epochstr = str(epochs)
        nbdigit = len(epochstr)

        for epoch in range(epochs):
            self.train_lr.append(scheduler.get_last_lr())
            train_loss, valid_loss = self._run_epoch(
                desc=f"Epoch [{str(epoch +1).zfill(nbdigit)}/{epochstr}]"
            )
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)
            scheduler.step(valid_loss)
            if (epochs+1)%update_freq ==0:
                self.criterion.feature_extractor.features.load_state_dict(self.encoder.convolutional_features.state_dict())
                for p in self.criterion.parameters():
                    p.requires_grad = False
                    
            if valid_loss < (best_valid_loss - delta):
                no_improvement_counter = 0
                best_valid_loss = valid_loss
                torch.save(self.encoder, f"{self.name}/{self.name}_encoder.pth")
                torch.save(self.decoder, f"{self.name}/{self.name}_decoder.pth")
            else:
                no_improvement_counter += 1
                if no_improvement_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        end_time = time.time()
        duration = end_time - start_time
        training_data = {
            "start": start_time,
            "end": end_time,
            "duration": duration,
            "validation_loss": self.valid_loss,
            "train_loss": self.train_loss,
            "train_lr": self.train_lr,
        }
        with open(f"{self.name}/ae_training_log.json", "w") as f:
            json.dump(training_data, f)
        if load_best:
            self.encoder.load_state_dict(
                torch.load(
                    f"{self.name}/{self.name}_encoder.pth", weights_only=False
                ).state_dict()
            )
            self.decoder.load_state_dict(
                torch.load(
                    f"{self.name}/{self.name}_decoder.pth", weights_only=False
                ).state_dict()
            )
            self.encoder.eval()
            self.decoder.eval()


class WarmupLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        final_lr: float,
        initial_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.final_lr = final_lr
        self.initial_lr = initial_lr
        super().__init__(optimizer, last_epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr
            param_group["initial_lr"] = self.final_lr

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                self.initial_lr
                + (self.final_lr - self.initial_lr)
                * self.last_epoch
                / self.warmup_steps
                for _ in self.optimizer.param_groups
            ]
        else:
            return [self.final_lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class CFMTrainer:
    def __init__(self, cfm, encoder, optimizer, train_loader, name, device):
        self.cfm = cfm
        self.name = name
        self.device = device
        self.encoder = encoder
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.train_size = len(self.train_loader.dataset)
        self.criterion = nn.MSELoss()
        self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.01)
        self.sampler = self.fm.sample_location_and_conditional_flow
        self.train_loss = []
        self.train_lr = []

    def train(
        self,
        epochs,
        min_lr=1e-6,
        high_lr=2e-4,
        warmup=10,
        patience=10,
        delta=0.0,
        load_best=True,
    ):
        os.makedirs(self.name, exist_ok=True)
        self.cfm.train()
        self.cfm = self.cfm.to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        best_loss = float("inf")
        epochs_no_improve = 0

        scheduler_lists = [
            WarmupLR(
                self.optimizer, warmup_steps=warmup, initial_lr=min_lr, final_lr=high_lr
            ),
            CosineAnnealingLR(self.optimizer, T_max=epochs - warmup, eta_min=min_lr),
        ]
        scheduler = SequentialLR(
            self.optimizer, milestones=[warmup], schedulers=scheduler_lists
        )

        start_time = time.time()
        epochstr = str(epochs)
        nbdigit = len(epochstr)
        for epoch in range(epochs):
            self.train_lr.append(scheduler.get_last_lr())
            desc = f"Epoch [{str(epoch +1).zfill(nbdigit)}/{epochstr}]"
            total_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    x1 = self.encoder(inputs)

                self.optimizer.zero_grad()
                x0 = torch.randn_like(x1)
                t, xt, ut = self.sampler(x0, x1)
                if self.cfm.nntype == "mlp":
                    vt = self.cfm((torch.cat([t.unsqueeze(-1), xt], dim=-1),))
                elif self.cfm.nntype == "unet":
                    vt = self.cfm((t, xt))
                else:
                    raise ValueError(f"Unknown model type: {self.cfm.type}")

                loss = self.criterion(ut, vt)
                loss.backward()
                nn.utils.clip_grad_value_(self.cfm.parameters(), 1, foreach=True)
                self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                batch = f"{batch_idx+1}/{len(self.train_loader)}"
                msg = f"\r{desc:^20} batch:{batch:^10} | train_loss:{total_loss / self.train_size:>7.2e} | lr:{self.train_lr[-1][0]:>7.1e}"
                print(msg, end="")
            total_loss /= self.train_size

            self.train_loss.append(total_loss)

            scheduler.step()
            print()
            if total_loss < (best_loss - delta):
                best_loss = total_loss
                epochs_no_improve = 0
                torch.save(self.cfm, f"{self.name}/{self.name}_cfm.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        end_time = time.time()
        duration = end_time - start_time
        training_data = {
            "start": start_time,
            "end": end_time,
            "duration": duration,
            "train_loss": self.train_loss,
        }
        with open(f"{self.name}/cfm_training_log.json", "w") as f:
            json.dump(training_data, f)
        if load_best:
            self.cfm.load_state_dict(
                torch.load(
                    f"{self.name}/{self.name}_cfm.pth", weights_only=False
                ).state_dict()
            )
        self.cfm.eval()

    def plot_training_loss(self):
        fig, ax1 = plt.subplots()
        color1 = "tab:red"
        ax1.plot(self.train_loss, color=color1, label="Loss")
        ax1.set_ylabel("Loss", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xlabel("Epochs")

        ax2 = ax1.twinx()
        color2 = "tab:gray"
        ax2.plot(self.train_lr, "--", color=color2, label="LR")
        ax2.set_ylabel("LR", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2

        ax1.legend(lines, labels)
        fig.tight_layout()
        plt.savefig(f"{self.name}/cfm_training.pdf")
        plt.show()
