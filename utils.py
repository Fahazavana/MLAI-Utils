# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import platform
from glob import glob
import numpy as np
import torch
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from pytorch_fid.inception import InceptionV3
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T


# %%
def get_device():
    if platform.platform().lower().startswith("mac"):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    else:  # Linux, Windows
        return "cuda" if torch.cuda.is_available() else "cpu"


# %%
class FIDDataSet(Dataset):
    def __init__(self, real, generated):
        if len(real) != len(generated):
            raise ValueError("The two dataset must have the same size")
        self.real = real
        self.generated = generated

    def __getitem__(self, idx):
        real, _ = self.real[idx]
        gen = self.generated[idx]
        return real, gen

    def __len__(self):
        return len(self.real)


# %%
class GeneratedData(torch.utils.data.Dataset):

    def __init__(self, root_dir, resize=False):
        self.root_dir = root_dir
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npy")
        ]
        self.resize = resize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        images = np.load(self.files[idx])
        images = torch.from_numpy(images)
        if self.resize:
            images = T.functional.resize(images, size=(299, 299))
        if images.shape[0] == 1:
            images = images.repeat(3, 1, 1)
        return images


# %%
class WrapperInceptionV3(nn.Module):
    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y


# %%
@torch.no_grad()
def generate(nbr_sample, cfm, decoder, device):
    t, s = cfm.sample(nbr_sample, device)
    return decoder(s)
# %%
def save_image(image, output_dir, index):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"image_{index}.npy"), image.numpy())
# %%
def generate_and_save(total_samples, batch_size, cfm, decoder, output_dir, device):
    num_full_batches = total_samples // batch_size
    remaining_samples = total_samples % batch_size
    saved = 0

    for _ in range(num_full_batches):
        generated_images = generate(batch_size, cfm, decoder, device).cpu()
        for i in range(generated_images.shape[0]):
            save_image(generated_images[i], output_dir, saved)
            saved += 1
            print(f"\rGenerated images saved at '{output_dir}': {saved}", end="")

    if remaining_samples > 0:
        generated_images = generate(remaining_samples, cfm, decoder, device).cpu()
        for _ in range(generated_images.shape[0]):
            save_image(generated_images[i], output_dir, saved)
            saved += 1
            print(f"\rGenerated images saved at '{output_dir}': {saved}", end="")
    print()

# %%
def compute_fid(dims, data_loader, device):
    def evaluation_step(engine, batch):
        real, fake = batch
        return real.squeeze(0), fake.squeeze(0)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx])

    wrapper_model = WrapperInceptionV3(inception_model)
    wrapper_model.eval()
    pytorch_fid_metric = FID(
        num_features=dims, feature_extractor=wrapper_model, device=device
    )

    evaluator = Engine(evaluation_step)
    pytorch_fid_metric.attach(evaluator, "fid")

    evaluator.run(data_loader, max_epochs=1)
    metrics = evaluator.state.metrics

    return metrics
