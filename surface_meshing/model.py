# import necessary libraries
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from einops import rearrange
from gem_cnn.nn import GemResNetBlock
from gem_cnn.transform import ScaleMask, GemPrecomp
from pretraining.model import MAEncoder
from torch.nn import *
from .blocks import ConvEncoder, ParallelTransportPool


class GEMCNN(nn.Module):
    def __init__(self, config, dim_in):
        super().__init__()
        self.dim = [dim_in] + config["dim"]
        self.max_order = config["max_order"]
        self.n_classes = config["n_classes"]

        # set transform
        self.transform = T.Compose([ScaleMask(0), GemPrecomp(n_rings=config["n_rings"], max_order=config["max_order"])])

        # set conv layers
        self.conv_dict = {
            "batch_norm": True,
            "checkpoint": True,
            "num_samples": 7,
            "n_rings": config["n_rings"]
        }

        self.conv = nn.ModuleList()
        self.make_probe()

    def make_probe(self):
        for i in range(len(self.dim)):
            if i == 0:
                self.conv.append(GemResNetBlock(self.dim[i], self.dim[i + 1], 0, self.max_order,
                                                **self.conv_dict))
            elif i == len(self.dim) - 1:
                self.conv.append(GemResNetBlock(self.dim[i], self.n_classes, self.max_order, 0,
                                                last_layer=True, **self.conv_dict))
            else:
                self.conv.append(GemResNetBlock(self.dim[i], self.dim[i + 1], self.max_order, self.max_order,
                                                **self.conv_dict))

    def forward(self, x, data):
        data = self.transform(data)
        attr = (data.edge_index, data.precomp, data.connection)

        for i in range(len(self.conv)):
            x = self.conv[i](x, *attr)

        return x


class GEMUNet(nn.Module):
    def __init__(self, config, dim_in):
        super().__init__()
        self.max_order = config["max_order"]
        self.n_classes = config["n_classes"]
        self.conv_dict = {
            "batch_norm": True,
            "checkpoint": True,
            "num_samples": 7,
            "n_rings": config["n_rings"]
        }

        # pre-compute each forward pass
        self.scale_transforms = [T.Compose([ScaleMask(i),
                                            GemPrecomp(n_rings=config["n_rings"], max_order=config["max_order"])])
                                 for i in range(3)]
        dim = config["dim"]

        # Encoder
        self.conv01 = GemResNetBlock(dim_in, dim[0], 0, self.max_order, **self.conv_dict)
        self.conv02 = GemResNetBlock(dim[0], dim[0], self.max_order, self.max_order, **self.conv_dict)

        # Downstream
        self.pool1 = ParallelTransportPool(1, unpool=False)
        self.conv11 = GemResNetBlock(dim[0], dim[1], self.max_order, self.max_order, **self.conv_dict)
        self.conv12 = GemResNetBlock(dim[1], dim[1], self.max_order, self.max_order, **self.conv_dict)

        self.pool2 = ParallelTransportPool(2, unpool=False)
        self.conv21 = GemResNetBlock(dim[1], dim[2], self.max_order, self.max_order, **self.conv_dict)
        self.conv22 = GemResNetBlock(dim[2], dim[2], self.max_order, self.max_order, **self.conv_dict)

        # Up-stream
        self.unpool2 = ParallelTransportPool(2, unpool=True)
        self.conv13 = GemResNetBlock(dim[1] + dim[2], dim[1], self.max_order, self.max_order, **self.conv_dict)
        self.conv14 = GemResNetBlock(dim[1], dim[1], self.max_order, self.max_order, **self.conv_dict)

        # Decoder
        self.unpool1 = ParallelTransportPool(1, unpool=True)
        self.conv03 = GemResNetBlock(dim[0] + dim[1], dim[0], self.max_order, self.max_order, **self.conv_dict)
        self.conv04 = GemResNetBlock(dim[0], dim[0], self.max_order, self.max_order, **self.conv_dict)
        self.conv05 = GemResNetBlock(dim[0], self.n_classes, self.max_order, 0, last_layer=True, **self.conv_dict)

    def forward(self, x, data):
        # scale graphs
        scale_data = [s(data) for s in self.scale_transforms]
        scale_attr = [(d.edge_index, d.precomp, d.connection) for d in scale_data]

        # encoder
        x = self.conv01(x, *scale_attr[0])
        x = self.conv02(x, *scale_attr[0])

        # downstream
        copy0 = x.clone()
        x = self.pool1(x, data)
        x = self.conv11(x, *scale_attr[1])
        x = self.conv12(x, *scale_attr[1])

        copy1 = x.clone()
        x = self.pool2(x, data)
        x = self.conv21(x, *scale_attr[2])
        x = self.conv22(x, *scale_attr[2])

        # upstream
        x = self.unpool2(x, data)
        x = torch.cat((x, copy1), dim=1)  # "copy/cat"
        x = self.conv13(x, *scale_attr[1])
        x = self.conv14(x, *scale_attr[1])

        x = self.unpool1(x, data)
        x = torch.cat((x, copy0), dim=1)  # "copy/cat"
        x = self.conv03(x, *scale_attr[0])
        x = self.conv04(x, *scale_attr[0])

        # decoder
        x = self.conv05(x, *scale_attr[0])

        return x


class FullModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config["encoder"]["n_channels"]
        self.n_classes = config["probe"]["n_classes"]
        self.probe_name = config["probe"]["name"]

        # set encoder
        self.encoder = self.str_to_attr(config["decoder"]["name"])(config)
        if "MAE" not in config["decoder"]["name"]:
            config["dim"] = self.encoder.dim_out

        # set probe
        self.probe = self.str_to_attr(self.probe_name)(config["probe"], config["dim"])
        self.final_activation = self.str_to_attr(config["probe"]["activation"])() \
            if "activation" in config["probe"] else Sigmoid()

    def forward(self, batch):
        x = batch.x
        if "MAE" in self.encoder.__class__.__name__:
            x = rearrange(x, "(b c n) l -> b c n l", b=len(batch), c=self.n_channels)
        else:
            x = rearrange(x, "(b c n) l -> b c n l", b=1, c=self.n_channels)
        x = self.encoder(x)

        x = rearrange(x, "b c n l -> (b c n) l")
        x = self.probe(x[:, :, None], batch)

        if self.n_classes == 1:
            return self.final_activation(x[:, :, 0])
        else:
            x_inner = self.final_activation(x[:, 0, 0])
            x_outer = F.leaky_relu(x[:, 1, 0])
            x_cls = x[:, 2, 0]
            return torch.stack([x_inner, x_outer, x_cls], dim=1)

    @staticmethod
    def str_to_attr(attr_name):
        return getattr(sys.modules[__name__], attr_name)
