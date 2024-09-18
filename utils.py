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
class FIDDataset(Dataset):
    def __init__(self, path, real_data, fake_batch):
        self.fake_batch = fake_batch
        self.fake_data = glob(f"{path}/*.pt")
        self.real_data = iter(
            DataLoader(real_data, batch_size=fake_batch, shuffle=True)
        )
        self.__test()

    def __test(self):
        fake_data = torch.load(
            self.fake_data[0], map_location=torch.device("cpu"), weights_only=True
        )
        if fake_data.shape[0] != self.fake_batch:
            raise ValueError(
                f"batch size must be equale to the batch_size of the saved images"
            )

    def __len__(self):
        return len(self.fake_data)

    def __getitem__(self, idx):
        real_data, _ = next(self.real_data)
        fake_data = torch.load(
            self.fake_data[idx], map_location=torch.device("cpu"), weights_only=True
        )

        # Resize first
        real_data = T.functional.resize(real_data, size=(299, 299))
        fake_data = T.functional.resize(fake_data, size=(299, 299))

        # Then repeat channels if needed
        if real_data.shape[1] != 3:
            real_data = real_data.repeat(1, 3, 1, 1)
        if fake_data.shape[1] != 3:
            fake_data = fake_data.repeat(1, 3, 1, 1)
        return real_data, fake_data


# %%
class FIDRecGen(Dataset):
    def __init__(self, gen_path, rec_path):
        self.gen_data = glob(f"{gen_path}/*.pt")
        self.rec_data = glob(f"{rec_path}/*.pt")
        self.__test()

    def __test(self):
        gen_data = torch.load(
            self.gen_data[0], map_location=torch.device("cpu"), weights_only=True
        )
        rec_data = torch.load(
            self.rec_data[0], map_location=torch.device("cpu"), weights_only=True
        )
        if gen_data.shape[0] != rec_data.shape[0]:
            raise ValueError(
                f"batch size must be equale to the batch_size of the saved images"
            )

    def __len__(self):
        return len(self.gen_data)

    def __getitem__(self, idx):
        gen_data = torch.load(
            self.gen_data[idx], map_location=torch.device("cpu"), weights_only=True
        )
        rec_data = torch.load(
            self.rec_data[idx], map_location=torch.device("cpu"), weights_only=True
        )

        # Resize first
        real_data = T.functional.resize(gen_data, size=(299, 299))
        fake_data = T.functional.resize(rec_data, size=(299, 299))

        # Then repeat channels if needed
        if gen_data.shape[1] != 3:
            gen_data = gen_data.repeat(1, 3, 1, 1)
        if rec_data.shape[1] != 3:
            rec_data = rec_data.repeat(1, 3, 1, 1)
        return rec_data, gen_data


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
def cfm_gen_and_save(
    cfm_model, decoder, path, device, batch_size=128, total=2048, final_size=(299, 299)
):
    cfm_model = cfm_model.to(device)
    decoder = decoder.to(device)
    decoder.eval()
    os.makedirs(path, exist_ok=True)
    for batch in range(total // batch_size):
        sample = cfm_model.sample(batch_size, device)
        with torch.no_grad():
            fake = decoder(sample)
        torch.save(fake, os.path.join(path, f"batch_{batch}.pt"))
        print(f"Batch {batch+1}: saved")


# %%
def ae_gen_and_save(
    encoder, decoder, path, data_set, device, batch_size=128, total=2048, final_size=(299, 299)
):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    os.makedirs(path, exist_ok=True)
    counter = 0
    for batch, (images, _) in enumerate(data_loader):
        images = images.to(device)
        code = encoder(images)
        recon = decoder(code)
        counter += images.shape[0]
        torch.save(recon, os.path.join(path, f"batch_{batch}.pt"))
        print(f"Batch {batch+1}: saved")
        if counter == total:
            break


# %%
def compute_fid(dims, data_loader, device):

    def evaluation_step(engine, batch):
        real, fake = batch
        return real.squeeze(0), fake.squeeze(0)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(device)

    wrapper_model = WrapperInceptionV3(inception_model)
    wrapper_model.eval()
    pytorch_fid_metric = FID(
        num_features=dims, feature_extractor=wrapper_model, device="cpu"
    )

    evaluator = Engine(evaluation_step)
    pytorch_fid_metric.attach(evaluator, "fid")

    evaluator.run(data_loader, max_epochs=1)
    metrics = evaluator.state.metrics

    return metrics