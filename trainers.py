from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .utils import WarmupLR


class AETrainer:
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        model_name: str,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_size = len(self.train_loader.dataset)
        self.val_size = (
            len(self.val_loader.dataset) if self.val_loader is not None else 0
        )
        self.model_name = model_name
        self.train_loss = []
        self.val_loss = []
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=10, min_lr=1e-10
        )

    def _run_epoch(self, loader: DataLoader, is_train: bool) -> float:
        """
        Runs a single training or validation epoch.

        Args:
            loader (DataLoader): The data loader for the epoch.
            is_train (bool): Whether to train or validate.

        Returns:
            float: The average loss for the epoch.
        """
        if is_train:
            self.encoder.train()
            self.decoder.train()
            data_size = self.train_size
        else:
            self.encoder.eval()
            self.decoder.eval()
            data_size = self.val_size

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

    def train(self, epochs: int, patience=10, delta=1e-4):
        """
        Trains the autoencoder model.

        Args:
            epochs (int): The number of epochs to train for.
        """
        best_val_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(epochs):
            train_loss = self._run_epoch(loader=self.train_loader, is_train=True)
            self.train_loss.append(train_loss)
            msg = f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}"
            if self.val_loader is not None:
                val_loss = self._run_epoch(loader=self.val_loader, is_train=False)
                self.val_loss.append(val_loss)
                self.scheduler.step(val_loss)
                msg += f" Val Loss: {val_loss:.4f}"
            print(msg)
            # Early stopping
            if val_loss < (best_val_loss - delta):
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.encoder.state_dict(), f"{self.model_name}_encoder.pth")
                torch.save(self.decoder.state_dict(), f"{self.model_name}_decoder.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break


class AEETrainer:
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        model_name:str,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_size = len(self.train_loader.dataset)
        self.val_size = (
            len(self.val_loader.dataset) if self.val_loader is not None else 0
        )
        self.model_name = model_name
        self.train_loss = []
        self.val_loss = []
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=10, min_lr=1e-10
        )

    def _run_epoch(self, loader: DataLoader, is_train: bool) -> float:
        """
        Runs a single training or validation epoch.

        Args:
            loader (DataLoader): The data loader for the epoch.
            is_train (bool): Whether to train or validate.

        Returns:
            float: The average loss for the epoch.
        """
        if is_train:
            self.encoder.train()
            self.decoder.train()
            data_size = self.train_size
        else:
            self.encoder.eval()
            self.decoder.eval()
            data_size = self.val_size

        total_loss = 0.0
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            if is_train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                encoded = self.encoder(inputs, labels)
                decoded = self.decoder(encoded)
                loss = self.criterion(decoded, inputs)

                if is_train:
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        return total_loss / data_size

    def train(self, epochs: int, patience=10, delta=1e-4):
        """
        Trains the autoencoder model.

        Args:
            epochs (int): The number of epochs to train for.
        """
        best_val_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(epochs):
            train_loss = self._run_epoch(loader=self.train_loader, is_train=True)
            self.train_loss.append(train_loss)
            msg = f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}"
            if self.val_loader is not None:
                val_loss = self._run_epoch(loader=self.val_loader, is_train=False)
                self.val_loss.append(val_loss)
                self.scheduler.step(val_loss)
                msg += f" Val Loss: {val_loss:.4f}"
            print(msg)
            # Early stopping
            if val_loss < (best_val_loss - delta):
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.encoder.state_dict(), f"{self.model_name}_encoder.pth")
                torch.save(self.decoder.state_dict(), f"{self.model_name}_decoder.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break


class AECTrainer:
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        model_name:str,
        classifier: nn.Module,
        optimizer: optim.Optimizer,
        criterion1: nn.Module,
        criterion2: nn.Module,
        train_loader,
        val_loader=None,
        device: str = "cpu",
    ):
        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.classifier = classifier.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.train_size = len(self.train_loader.dataset)
        self.val_size = (
            len(self.val_loader.dataset) if self.val_loader is not None else 0
        )
        self.model_name = model.name
        self.train_loss = []
        self.val_loss = []

        self.val_accuracy = []
        self.val_cls_loss = []
        self.val_rec_loss = []
        self.val_accuracy = []

        self.train_accuracy = []
        self.train_cls_loss = []
        self.train_rec_loss = []
        self.train_accuracy = []

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=10, min_lr=1e-10
        )

    def _run_epoch(self, loader, is_train: bool) -> float:
        """
        Runs a single training or validation epoch.

        Args:
            loader (DataLoader): The data loader for the epoch.
            is_train (bool): Whether to train or validate.

        Returns:
            float: The average loss for the epoch.
        """
        if is_train:
            self.encoder.train()
            self.decoder.train()
            self.classifier.train()
            data_size = self.train_size
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.classifier.eval()
            data_size = self.val_size

        total_loss = 0.0
        total_recloss = 0.0
        total_clsloss = 0.0
        correct = 0
        total = 0
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            if is_train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                encoded = self.encoder(inputs)
                decoded = self.decoder(encoded)
                classified = self.classifier(encoded)

                loss1 = self.criterion1(decoded, inputs)
                loss2 = self.criterion2(classified, labels)
                loss = loss2 + 10 * loss1
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                _, predicted = torch.max(classified.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            total_recloss += loss1.item() * inputs.size(0)
            total_clsloss += loss2.item() * inputs.size(0)
            total_loss += loss.item() * inputs.size(0)

        accuracy = 100 * correct / total
        return (
            total_loss / data_size,
            total_clsloss / data_size,
            total_recloss / data_size,
            accuracy,
        )

    def train(self, epochs: int, patience=10, delta=1e-4):
        """
        Trains the autoencoder model.

        Args:
            epochs (int): The number of epochs to train for.
        """
        best_val_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(epochs):
            train_loss, cls, rec, acc = self._run_epoch(
                loader=self.train_loader, is_train=True
            )
            self.train_loss.append(train_loss)
            self.train_cls_loss.append(cls)
            self.train_rec_loss.append(rec)
            self.train_accuracy.append(acc)
            msg = f"Epoch [{epoch + 1}/{epochs}], TL: {train_loss:.4f}, TC:{cls:.4f}, TA:{acc:.4f},  TR={rec:.4f}"
            if self.val_loader is not None:
                val_loss, vcls, vrec, vacc = self._run_epoch(
                    loader=self.val_loader, is_train=False
                )
                self.val_loss.append(val_loss)
                self.val_cls_loss.append(vcls)
                self.val_rec_loss.append(vrec)
                self.val_accuracy.append(vacc)
                self.scheduler.step(val_loss)
                self.scheduler.step(val_loss)
                msg += (
                    f" | VL: {val_loss:.4f}, VC:{vcls:.4f}, VA:{vacc:.4f} VR:{vrec:.4f}"
                )
            print(msg)
            # Early stopping
            if val_loss < (best_val_loss - delta):
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.encoder.state_dict(), f"{self.model_name}_encoder.pth")
                torch.save(self.decoder.state_dict(), f"{self.model_name}_decoder.pth")
                torch.save(self.classifier.state_dict(), f"{self.model_name}_classifier.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break



class CFMTrainer:
    def __init__(
        self,
        cfm_model: nn.Module,
        sampler: Any,
        encoder: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        model_name:str,
        device: str = "cpu",
    ):
        self.device = device
        self.cfm_model = cfm_model.to(device)
        self.sampler = sampler
        self.encoder = encoder.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.data_size = len(self.train_loader.dataset)
        self.train_loss = []
        self.train_lr = []
        self.model_name = model_name
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(
        self,
        epochs: int,
        lr_scheduler: Optional[WarmupLR] = None,
        patience: int = 10,
        delta: float = 1e-4,
    ):
        """
        Trains the CFM model.

        Args:
            epochs (int): The number of epochs to train for.
            lr_scheduler (Optional[WarmupLR], optional): The learning rate scheduler to use. Defaults to None.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 10.
            delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 1e-4.
        """

        self.cfm_model.train()
        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            if lr_scheduler is not None:
                self.train_lr.append(lr_scheduler.get_last_lr())
            total_loss = 0.0
            for inputs, _ in self.train_loader:
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    x1 = self.encoder(inputs)

                self.optimizer.zero_grad()
                x0 = torch.randn_like(x1)
                t, xt, ut = self.sampler(x0, x1)

                if self.cfm_model.model_type == "mlp":
                    vt = self.cfm_model((torch.cat([t.unsqueeze(-1), xt], dim=-1),))
                elif self.cfm_model.model_type == "unet":
                    vt = self.cfm_model((t, xt))
                else:
                    raise ValueError(f"Unknown model type: {self.cfm_model.model_type}")

                loss = self.criterion(ut, vt)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)
            total_loss /= self.data_size

            self.train_loss.append(total_loss)

            if lr_scheduler is not None:
                lr_scheduler.step()
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss:.4f}, lr={self.train_lr[-1][0]:.4f}"
                )
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss:.4f}")

            # Early stopping
            if total_loss < (best_loss - delta):
                best_loss = total_loss
                epochs_no_improve = 0
                torch.save(self.encoder.state_dict(), f"{self.model_name}_cfm.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break


class CFMETrainer:
    def __init__(
        self,
        cfm_model: nn.Module,
        sampler: Any,
        encoder: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        model_name:str,
        device: str = "cpu",
    ):
        self.device = device
        self.cfm_model = cfm_model.to(device)
        self.sampler = sampler
        self.encoder = encoder.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.data_size = len(self.train_loader.dataset)
        self.train_loss = []
        self.train_lr = []
        self.model_name = model_name
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(
        self,
        epochs: int,
        lr_scheduler: Optional[WarmupLR] = None,
        patience: int = 10,
        delta: float = 1e-4,
    ):
        """
        Trains the CFM model.

        Args:
            epochs (int): The number of epochs to train for.
            lr_scheduler (Optional[WarmupLR], optional): The learning rate scheduler to use. Defaults to None.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 10.
            delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 1e-4.
        """

        self.cfm_model.train()
        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            if lr_scheduler is not None:
                self.train_lr.append(lr_scheduler.get_last_lr())
            total_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    x1 = self.encoder(inputs, labels)

                self.optimizer.zero_grad()
                x0 = torch.randn_like(x1)
                t, xt, ut = self.sampler(x0, x1)

                if self.cfm_model.model_type == "mlp":
                    vt = self.cfm_model((torch.cat([t.unsqueeze(-1), xt], dim=-1),))
                elif self.cfm_model.model_type == "unet":
                    vt = self.cfm_model((t, xt))
                else:
                    raise ValueError(f"Unknown model type: {self.cfm_model.model_type}")

                loss = self.criterion(ut, vt)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)
            total_loss /= self.data_size

            self.train_loss.append(total_loss)

            if lr_scheduler is not None:
                lr_scheduler.step()
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss:.4f}, lr={self.train_lr[-1][0]:.4f}"
                )
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss:.4f}")

            # Early stopping
            if total_loss < (best_loss - delta):
                best_loss = total_loss
                epochs_no_improve = 0
                torch.save(self.encoder.state_dict(), f"{self.model_name}_cfm.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

