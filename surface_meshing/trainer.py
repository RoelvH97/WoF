# import necessary libraries
import os
import torch

from einops import rearrange
from os.path import dirname
from pretraining.trainer import LightningBase
from torch_geometric.data import Batch, Data
from utils import numpy_to_sitk, sitk_to_numpy, stl_to_mask, Icosahedron, MPRTransform
from .losses import NestedBoundaryLoss
from .model import FullModel


class LightningSphere(LightningBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = self.str_to_attr(self.c_m["loss"])()

        self.model = FullModel(self.c_m)
        if self.c_o["encoder"]["freeze"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        config_ico = self.c_d["icosphere"]
        self.icosphere = Icosahedron(config_ico["subdivisions"], config_ico["bins"])

    def training_step(self, batch, batch_idx):
        y_pred = self.model(batch)
        y_true = batch.y.sum(axis=1)
        if len(y_true.shape) == 1:
            y_true = y_true[:, None]

        loss = self.loss(y_pred, y_true)
        if isinstance(loss, tuple):
            for l, name in zip(loss, ["InnerLoss", "OuterLoss", "NestedBoundaryLoss", "ClassLoss"]):
                self.log(f"train/{name}", l)
            loss = (self.c_o["lambdas"][0] * (loss[0] + loss[1]) +
                    self.c_o["lambdas"][1] * loss[2] + self.c_o["lambdas"][2] * loss[3])
        else:
            self.log(f"train/{self.c_m['loss']}", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        y_pred = self.model(batch)
        y_true = batch.y.sum(axis=1)
        if len(y_true.shape) == 1:
            y_true = y_true[:, None]

        loss = self.loss(y_pred, y_true)
        if isinstance(loss, dict):
            for l, name in zip(loss, ["InnerLoss", "OuterLoss", "NestedBoundaryLoss", "ClassLoss"]):
                self.log(f"val/{name}", l)
            loss = (self.c_o["lambdas"][0] * (loss[0] + loss[1]) +
                    self.c_o["lambdas"][1] * loss[2] + self.c_o["lambdas"][2] * loss[3])
        else:
            self.log(f"val/{self.c_m['loss']}", loss)

        if batch_idx % 20 == 0:
            if "NestedBoundaryLoss" in self.loss.__class__.__name__:
                # multiply out
                y_pred_tmp = y_pred.clone()
                y_pred_tmp[:, 1] = torch.clip(y_pred_tmp[:, 1], 0, None)

                # probably good to check optimal threshold via AUC on validation set, didn't do it here myself
                y_pred_tmp[:, 1] = y_pred_tmp[:, 1] * (torch.sigmoid(y_pred_tmp[:, 2]) > 0.5)
            else:
                y_pred_tmp = y_pred.clone()

            self.log_meshes(batch, y_pred_tmp, batch_idx)
            self.log_rays(batch, y_pred_tmp, batch_idx, mode="val")

        return loss

    def infer(self, image, landmark, output_path=None):
        # read image
        image, spacing, offset = sitk_to_numpy(image)
        shape = image.shape

        # get image data
        ray_casts = self.icosphere.ray_cast(image, spacing, offset, landmark, self.c_d["icosphere"]["radius"])
        ray_casts = torch.clip(ray_casts, -360, 840)
        ray_casts = (ray_casts - 240) / 600

        data = Data(
            x=ray_casts,
            connection=self.icosphere.data.connection,
            edge_coords=self.icosphere.data.edge_coords,
            edge_index=self.icosphere.data.edge_index,
            frame=self.icosphere.data.frame,
            normal=self.icosphere.data.normal,
            pos=self.icosphere.data.pos,
            weight=self.icosphere.data.weight,
            node_mask=self.icosphere.data.node_mask,
            edge_mask=self.icosphere.data.edge_mask,
        )

        # turn into batch
        data = Batch.from_data_list([data])

        self.eval()
        with torch.no_grad():
            y_pred = self.model(data)

        is_myocardium = self.c_m["probe"]["n_classes"] == 3
        if is_myocardium:
            y_pred_tmp = y_pred.clone()

            # reshape if needed to match expected format
            if len(y_pred_tmp.shape) > 2:
                y_pred_tmp = y_pred_tmp.reshape(len(self.icosphere.vertices), self.c_m["probe"]["n_classes"])

            # apply post-processing like in the validation step
            y_pred_tmp[:, 1] = torch.clip(y_pred_tmp[:, 1], 0, None)
            y_pred_final = y_pred_tmp[:, :2]
        else:
            y_pred_final = y_pred.clone()

            # ensure the prediction has the right shape
            if len(y_pred_final.shape) == 1:
                y_pred_final = y_pred_final.reshape(-1, 1)

        if output_path:
            os.makedirs(dirname(output_path) if dirname(output_path) else '.', exist_ok=True)
            if is_myocardium:
                # generate mask
                y_pred_final[:, 1] = y_pred_final[:, 0] + y_pred_final[:, 1]
                for i in range(2):
                    self.icosphere.to_stl(y_pred_final.cpu().detach().numpy()[:, i:i + 1],
                                          self.c_d["icosphere"]["radius"],
                                          landmark,
                                          output_path + f"_{i}")
                    stl_to_mask(shape, spacing, offset, output_path + f"_{i}.stl")

                # load the masks
                mask_lv, _, _ = sitk_to_numpy(output_path + "_0.nii.gz")
                mask_full, _, _ = sitk_to_numpy(output_path + "_1.nii.gz")

                # remove
                os.remove(output_path + "_0.nii.gz")
                os.remove(output_path + "_1.nii.gz")
                os.remove(output_path + "_0.stl")
                os.remove(output_path + "_1.stl")

                # subtract the masks to get the myocardium
                mask_myo = mask_full - mask_lv
                numpy_to_sitk(mask_myo, spacing, offset, output_path + ".nii.gz")

                # generate the myocardium mesh
                y_pred_final[:, 1] = y_pred_final[:, 1] - y_pred_final[:, 0]
                y_pred_final[:, 1] = y_pred_final[:, 1] * (torch.sigmoid(y_pred_tmp[:, 2]) > 0.5)
                self.icosphere.to_stl_myo(
                    y_pred_final.cpu().detach().numpy(),
                    self.c_d["icosphere"]["radius"],
                    landmark,
                    output_path
                )
            else:
                self.icosphere.to_stl(
                    y_pred_final.cpu().detach().numpy(),
                    self.c_d["icosphere"]["radius"],
                    landmark,
                    output_path
                )
                stl_to_mask(shape, spacing, offset, output_path + ".stl")


    def log_meshes(self, batch, y_pred, idx):
        os.makedirs(os.path.join(self.logger.log_dir, "images"), exist_ok=True)

        # determine if dealing with myocardium data
        is_myocardium = self.c_m["probe"]["n_classes"] == 3
        to_stl = self.icosphere.to_stl_myo if is_myocardium else self.icosphere.to_stl

        # reshape predictions and ground truth
        y_pred = y_pred.reshape(len(batch), -1, self.c_m["probe"]["n_classes"])
        if is_myocardium:
            y_pred = y_pred[:, :, :2]

        batch.y = batch.y.sum(axis=1)
        batch.y = batch.y[:, None] if len(batch.y.shape) == 1 else batch.y

        for i in range(len(batch)):
            # generate mesh file path
            file_path = f"{self.logger.log_dir}/images/{str(idx).zfill(3)}_{str(i).zfill(3)}.stl"

            # convert to STL mesh
            to_stl(
                y_pred[i].cpu().detach().numpy(),
                batch[i].radius.cpu().detach().item(),
                batch[i].landmark.cpu().detach().numpy(),
                file_path,
                batch[i].y.cpu().detach().numpy(),
            )


    def log_rays(self, batch, y_pred, idx, mode="train"):
        y_pred = y_pred.reshape(len(batch), -1, self.c_m["probe"]["n_classes"])

        # determine if dealing with myocardium data (3 classes)
        is_myocardium = self.c_m["probe"]["n_classes"] == 3
        n_classes = 2 if is_myocardium else 1

        for i in range(len(batch)):
            # get sample-specific data
            x = batch[i].x.cpu().detach()
            if is_myocardium:
                y_true = batch[i].y.cpu().detach()
            else:
                y_true = batch[i].y.sum(axis=1, keepdim=True).cpu().detach()
            y_pred_i = y_pred[i]

            # normalize x coordinates to [0,1]
            x = torch.clamp((x - x.min()) / (x.max() - x.min()), 0, 1)
            x_rgb = x[None].repeat(3, 1, 1)  # Create RGB channels

            # handle myocardium by combining inner and outer walls
            y_pred_tmp = y_pred_i.clone()
            y_true_tmp = y_true.clone()

            if is_myocardium:
                y_pred_tmp[:, 1] = y_pred_tmp[:, 0] + y_pred_tmp[:, 1]
                y_true_tmp[:, 1] = y_true_tmp[:, 0] + y_true_tmp[:, 1]

            # transform predictions and ground truth to image indices
            for j in range(n_classes):
                y_pred_idx = torch.clip((y_pred_tmp[:, j] * x.shape[1]).to(int).cpu().detach(), 0, 255)
                y_true_idx = torch.clip((y_true_tmp[:, j] * x.shape[1]).to(int).cpu().detach(), 0, 255)

                # set colors: predictions in red, ground truth in green
                x_rgb[0, torch.arange(x_rgb.shape[1]), y_pred_idx] = 1
                x_rgb[1, torch.arange(x_rgb.shape[1]), y_true_idx] = 1

            # log the visualization
            self.logger.experiment.add_image(f"{mode}/{str(idx).zfill(3)}_{i}_{j}",
                                             torch.swapaxes(x_rgb, 1, 2),self.current_epoch)


class LightningTube(LightningBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = self.str_to_attr(self.c_m["loss"])()
        self.model = FullModel(self.c_m)

        self.transform = MPRTransform(self.c_d["mpr_transform"])

    def training_step(self, batch, batch_idx):
        y_pred = self.model(batch)
        y_true = batch.y.sum(axis=1, keepdim=True)

        loss = self.loss(y_pred, y_true)
        self.log(f"train/{self.c_m['loss']}", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        y_pred = self.model(batch)
        y_true = batch.y.sum(axis=1, keepdim=True)

        loss = self.loss(y_pred, y_true)
        self.log(f"val/{self.c_m['loss']}", loss)

        if batch_idx % 20 == 0:
            self.log_meshes(batch, y_pred)
            self.log_rays(batch, y_pred, mode="val")

        return loss

    def infer(self, image, ctl_list, output_path=None):
        self.transform = MPRTransform(self.c_d["mpr_transform"], device="cpu")

        # read image
        image, spacing, offset = sitk_to_numpy(image)
        shape = image.shape

        # prepare image
        image = torch.from_numpy(image).float()
        spacing = torch.tensor(spacing).float()

        offset = torch.tensor(offset).float()
        offset[:2] *= -1

        data = []
        for ctl in ctl_list:
            ctl = torch.from_numpy(ctl).float()
            ctl = ctl - offset

            # get image data
            mpr = self.transform((image, spacing), ctl, mode="polar")
            mpr = rearrange(mpr, "r theta z -> (z theta) r")

            mpr = mpr - 1024 if mpr.min() >= -10 else mpr
            mpr = torch.clip(mpr, -760, 1240)
            mpr = (mpr - 240) / 1000

            mpr_data = self.transform.generate_multiscale_graph_data(ctl)
            data.append(Data(
                x=mpr,
                ctl=ctl + offset,
                connection=mpr_data.connection,
                edge_coords=mpr_data.edge_coords,
                edge_index=mpr_data.edge_index,
                frame=mpr_data.frame,
                normal=mpr_data.normal,
                pos=mpr_data.pos,
                weight=mpr_data.weight,
                node_mask=mpr_data.node_mask,
                edge_mask=mpr_data.edge_mask,
            ))

        # turn into batch
        batch = Batch.from_data_list(data)

        self.eval()
        with torch.no_grad():
            y_pred = self.model(batch)

        if output_path:
            spacing, offset = spacing.cpu().detach().numpy(), offset.cpu().detach().numpy()
            os.makedirs(dirname(output_path) if dirname(output_path) else '.', exist_ok=True)

            j = 0
            for i in range(len(batch)):
                y_pred_tmp = y_pred[j:j + batch[i].x.shape[0], 0].cpu().detach() * batch[i].x.shape[-1]
                j += batch[i].x.shape[0]

                # reshape
                y_pred_tmp = rearrange(y_pred_tmp, "(z theta) -> theta z", theta=self.transform.n_theta)

                # convert to STL mesh
                self.transform.to_stl(
                    batch[i].ctl.cpu().detach(),
                    y_pred_tmp.T,
                    output_path + f"_{str(i).zfill(3)}",
                )
                if i == 0:
                    mask = stl_to_mask(shape, spacing, offset, output_path + f"_{str(i).zfill(3)}.stl")
                else:
                    mask += stl_to_mask(shape, spacing, offset, output_path + f"_{str(i).zfill(3)}.stl")
                os.remove(output_path + f"_{str(i).zfill(3)}.nii.gz")

            # full mask
            numpy_to_sitk(mask > 0, spacing, offset, output_path + ".nii.gz")

    def log_meshes(self, batch, y_pred):
        os.makedirs(os.path.join(self.logger.log_dir, "images"), exist_ok=True)

        j = 0
        for i in range(len(batch)):
            # generate mesh file path
            file_path = f"{self.logger.log_dir}/images/{str(i).zfill(3)}_{str(batch[i].sample[0].cpu().detach().numpy()).zfill(3)}.stl"

            y_true_tmp = batch[i].y.sum(dim=1).cpu().detach() * batch[i].x.shape[-1]
            y_pred_tmp = y_pred[j:j + y_true_tmp.shape[0], 0].cpu().detach() * batch[i].x.shape[-1]
            j += y_true_tmp.shape[0]

            # reshape
            y_true_tmp = rearrange(y_true_tmp, "(z theta) -> theta z", theta=self.transform.n_theta)
            y_pred_tmp = rearrange(y_pred_tmp, "(z theta) -> theta z", theta=self.transform.n_theta)

            # convert to STL mesh
            self.transform.to_stl(
                batch[i].ctl.cpu().detach(),
                y_pred_tmp.T,
                file_path,
                y_true_tmp.T,
            )

    def log_rays(self, batch, y_pred, mode="train"):
        j = 0
        for i in range(len(batch)):
            x = batch[i].x.cpu().detach()
            y_true_tmp = batch[i].y.sum(dim=1).cpu().detach() * x.shape[-1]
            y_pred_tmp= y_pred[j:j + y_true_tmp.shape[0], 0].cpu().detach() * x.shape[-1]
            j += y_true_tmp.shape[0]

            x = torch.clamp(x + 0.5, 0, 1)
            x = rearrange(x, "(z theta) r -> r theta z", theta=self.transform.n_theta)
            x_rgb = x[None].repeat(3, 1, 1, 1)

            y_pred_tmp = rearrange(y_pred_tmp, "(z theta) -> theta z", theta=self.transform.n_theta)
            y_true_tmp = rearrange(y_true_tmp, "(z theta) -> theta z", theta=self.transform.n_theta)

            # to image indices
            y_pred_tmp = torch.clip(y_pred_tmp, 0, 31).to(int)
            y_true_tmp = torch.clip(y_true_tmp, 0, 31).to(int)

            for k in range(0, x.shape[1], x.shape[1] // 4):
                x_rgb_tmp = x_rgb[:, :, k].cpu().detach()
                x_rgb_tmp[0, y_pred_tmp[k].to(int), torch.arange(x_rgb_tmp.shape[-1])] = 1
                x_rgb_tmp[1, y_true_tmp[k].to(int), torch.arange(x_rgb_tmp.shape[-1])] = 1
                self.logger.experiment.add_image(f"{mode}/{str(i).zfill(3)}_{str(batch[i].sample[0].cpu().detach().numpy()).zfill(3)}_{str(k).zfill(3)}",
                                                 x_rgb_tmp, self.current_epoch)
