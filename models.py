import torch
from torch import nn
from torchcfm.utils import torch_wrapper
from torchdyn.core import NeuralODE


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            if "Conv" in classname:
                nn.init.normal_(m.weight.data, 0.0, 0.02)

            elif "BatchNorm" in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

            elif "Linear" in classname:
                nn.init.normal_(m.weight.data, 0, 0.02)


class BaseEncoder(Net):
    def __init__(self):
        super().__init__()
        self.net = self.build_encoder()

    def build_encoder(self):
        raise NotImplementedError("Build encoder method must be implemented")

    def forward(self, x):
        return self.net(x)


class BaseDecoder(Net):
    def __init__(self):
        super().__init__()
        self.net = self.build_decoder()

    def build_decoder(self):
        raise NotImplementedError("Build decoder method must be implemented")

    def forward(self, x):
        return self.net(x)

class Encoder28(nn.Module):
    def __init__(self, c_in=1, c_hid=16, c_out=1, dim=16):
        super(Encoder28, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(c_in, c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(4 * c_hid, 8 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(8 * c_hid, 16 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Flatten(1, -1),
            nn.Linear(16 * c_hid, c_out * dim * dim),
            nn.Unflatten(1, (c_out, dim, dim)),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder28(nn.Module):
    def __init__(self, c_out=1, c_hid=16, c_in=1, dim=16):
        super(Decoder28, self).__init__()
        self.decoder = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(c_in * dim * dim, 16 * c_hid),
            nn.GELU(),
            nn.Unflatten(1, (16 * c_hid, 1, 1)),
            nn.ConvTranspose2d(
                16 * c_hid,
                8 * c_hid,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                8 * c_hid,
                4 * c_hid,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                4 * c_hid,
                2 * c_hid,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                2 * c_hid, c_hid, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                c_hid, c_out, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class CFMModel(Net):
    def __init__(self, feature_size, model, model_type, init_weights, node_kwargs):
        super().__init__()
        self.net = model
        self.feature_size = feature_size
        self.model_type = model_type
        self.node = self._create_node(model, model_type, node_kwargs)
        if init_weights:
            self.weights_init()

    def _create_node(self, model, model_type, node_kwargs):
        if model_type == "mlp":
            node_model = torch_wrapper(model)
        elif model_type == "unet":
            node_model = model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return NeuralODE(node_model, **node_kwargs)

    def forward(self, inputs):
        return self.net(*inputs)

    @torch.no_grad()
    def sample(self, nbr_sample, device):
        self.net.train()
        self.node = self.node.to(device)
        noise = self._generate_noise(nbr_sample, device)

        trajectory = self.node.trajectory(
            noise,
            t_span=torch.linspace(0, 1, 2, device=device),
        )
        return trajectory[1]

    def _generate_noise(self, nbr_sample, device):
        if self.model_type == "unet":
            return torch.randn((nbr_sample, *self.feature_size), device=device)
        else:
            return torch.randn((nbr_sample, self.feature_size), device=device)
