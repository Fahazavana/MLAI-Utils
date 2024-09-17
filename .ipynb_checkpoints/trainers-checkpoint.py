import json
import os
import time

from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
)

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from tqdm import tqdm
from utils import WarmupLR


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

    def _run_epoch(self, is_train):
        if is_train:
            self.encoder.train()
            self.decoder.train()
            data_size = self.train_size
            loader = self.train_loader
        else:
            self.encoder.eval()
            self.decoder.eval()
            data_size = self.valid_size
            loader = self.valid_loader

        total_loss = 0.0
        for inputs, _ in loader:
            inputs = inputs.to(self.device)
            if is_train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                encoded = self.encoder(inputs)
                decoded = self.decoder(encoded)
                loss = self.criterion(decoded, inputs)

                if is_train:
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        return total_loss / data_size

    def train(self, epochs: int, patience=10, delta=0.0, load_best=True):
        os.makedirs(self.name, exist_ok=True)
        no_improvement_counter = 0
        best_valid_loss = float("inf")

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=10, min_lr=1e-8
        )
        start_time = time.time()
        for epoch in range(epochs):
            self.train_lr.append(scheduler.get_last_lr())
            train_loss = self._run_epoch(is_train=True)
            valid_loss = self._run_epoch(is_train=False)

            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)
            msg = f"Train Loss: {train_loss:.3e} | Val Loss: {valid_loss:.3e} | lr:{self.train_lr[-1][0]:.3e}"
            scheduler.step(valid_loss)
            scheduler.step(valid_loss)
            print(msg)

            if valid_loss < (best_valid_loss - delta):
                no_improvement_counter = 0
                best_valid_loss = valid_loss
                torch.save(
                    self.encoder.state_dict(), f"{self.name}/{self.name}_encoder.pth"
                )
                torch.save(
                    self.decoder.state_dict(), f"{self.name}/{self.name}_decoder.pth"
                )
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
        }
        with open(f"{self.name}/ae_training_log.json", "w") as f:
            json.dump(training_data, f)
        if load_best:
            self.encoder.load_state_dict(
                torch.load(f"{self.name}/{self.name}_encoder.pth")
            )
            self.decoder.load_state_dict(
                torch.load(f"{self.name}/{self.name}_decoder.pth")
            )
            self.encoder.eval()
            self.decoder.eval()


class CFMTrainer:
    def __init__(self, cfm, encoder, optimizer, train_loader, name, device):
        self.cfm = cfm
        self.name = name
        self.device = device
        self.encoder = encoder
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.train_size = len(self.train_loader)
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
        for epoch in range(epochs):
            self.train_lr.append(scheduler.get_last_lr())
            total_loss = 0.0
            for inputs, _ in self.train_loader:
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
                self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)
            total_loss /= self.train_size

            self.train_loss.append(total_loss)

            scheduler.step()
            msg = f"Epoch [{epoch + 1}/{epochs}]: Train Loss: {total_loss:.3e} | lr:{self.train_lr[-1][0]:.3e}"
            print(msg)

            if total_loss < (best_loss - delta):
                best_loss = total_loss
                epochs_no_improve = 0
                torch.save(self.cfm.state_dict(), f"{self.name}/{self.name}_cfm.pth")
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
            self.cfm.load_state_dict(torch.load(f"{self.name}/{self.name}_cfm.pth"))
        self.cfm.eval()
