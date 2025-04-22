# import necessary libraries
import lightning.pytorch as pl
import sys

from surface_meshing.losses import *
from torch.nn import *
from torch.optim import *
from torch.optim.lr_scheduler import *
from .model import *


class SequentialScheduler(LRScheduler):
    def __init__(self, optimizer, scheduler1, scheduler2, switch_epoch):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.switch_epoch = switch_epoch
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.switch_epoch:
            return self.scheduler1.get_last_lr()
        else:
            return self.scheduler2.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.switch_epoch:
            self.scheduler1.step(epoch)
        else:
            self.scheduler2.step(epoch)
        super().step(epoch)


class LightningBase(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.c_d, self.c_m, self.c_o = config["DATA"], config["MODEL"], config["OPTIMIZATION"]

    def warmup_lambda(self, iter_):
        return iter_ / self.c_o["n_warmup"] if iter_ < self.c_o["n_warmup"] else 1

    @staticmethod
    def str_to_attr(attr_name):
        return getattr(sys.modules[__name__], attr_name)

    def configure_optimizers(self):
        optimizer = self.str_to_attr(self.c_o["name"])(self.model.parameters(), **self.c_o["optimizer"])

        if self.c_o["n_warmup"] == 0:
            scheduler = self.str_to_attr(self.c_o["lr_policy"])(optimizer, **self.c_o["scheduler"])
        else:
            scheduler1 = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
            scheduler2 = self.str_to_attr(self.c_o["lr_policy"])(optimizer, **self.c_o["scheduler"])
            scheduler = SequentialScheduler(optimizer, scheduler1, scheduler2, self.c_o["n_warmup"])
        return [optimizer], [scheduler]


class LightningMAE(LightningBase):
    """
    Lightning-style trainer for masked auto-encoding
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = self.str_to_attr(self.c_m["loss"])()

        self.model = self.str_to_attr(self.c_m["decoder"]["name"])(self.c_m)

    def training_step(self, batch, batch_idx):
        pred_pixel_values, decoder_tokens, masked_patches, masked_indices = self.model(batch)

        loss = self.loss(pred_pixel_values, masked_patches)
        loss_reg = torch.mean(torch.sum((decoder_tokens**2), dim=-1))
        self.log(f"train/{self.c_m['loss']}", loss)
        self.log("train/loss_reg", loss_reg)
        return loss + self.c_o["lambdas"][0] * loss_reg

    def validation_step(self, batch, batch_idx):
        pred_pixel_values, decoder_tokens, masked_patches, masked_indices = self.model(batch)

        loss = self.loss(pred_pixel_values, masked_patches)
        loss_reg = torch.mean(torch.sum((decoder_tokens**2), dim=-1))
        self.log(f"val/{self.c_m['loss']}", loss)
        self.log("val/loss_reg", loss_reg)

        if batch_idx % 50 == 0:
            batch_masked = self.model.reconstruct_image(batch[:, 0], masked_indices)
            batch_pred = self.model.reconstruct_image(batch[:, 0], masked_indices, pred_pixel_values)
            self.log_images(batch.cpu(), batch_masked, batch_pred, batch_idx)
        return loss + self.c_o["lambdas"][0] * loss_reg

    def log_images(self, true, masked, pred, idx):
        norm = (true.min(), true.max())

        for image, identifier in zip((true, masked, pred), ("true", "masked", "pred")):
            image = torch.clamp((image - norm[0]) / (norm[1] - norm[0]), 0, 1)
            image = image.swapaxes(2, 3)
            for i in range(image.shape[0]):
                self.logger.experiment.add_image(f"{identifier}/rand_struct_{idx}", image[i], self.current_epoch)
