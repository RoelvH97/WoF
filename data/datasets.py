# import necessary libraries
import einops
import h5py
import json
import lightning.pytorch as pl
import torch
import trimesh

from utils.polar_transform import PolarTransform
from .augmentation import *
from gem_cnn.transform import SimpleGeometry
from glob import glob
from itertools import cycle, islice
from os.path import basename, exists, join
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as DataLoaderGeo
from utils import Icosahedron, MPRTransform, PolarTransform, sitk_to_numpy
from utils.preprocessing import make_folds, make_folds_asoca


class IcosphereDataset(Dataset):
    def __init__(self, config, mode="train"):
        super().__init__()
        self.config = config
        self.mode = mode

        # data dicts
        self.id_list, self.images = self.setup()
        self.h5_cache = {}

        # define icosphere
        config_ico = config["icosphere"]
        self.icosphere = Icosahedron(config_ico["subdivisions"], config_ico["bins"])
        self.r = config_ico["radius"]  # in mm

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id_ = self.id_list[index]
        path_image = self.images[id_]
        path_h5 = path_image.replace("images", "h5").replace("_0000.nii.gz", ".h5")

        if self.mode == "train" or self.mode == "train_val":
            return self.load_h5(path_h5, augment=True)
        else:
            return self.load_h5(path_h5)

    def setup(self):
        if not exists(join(self.config["root_dir"], "splits_final.json")):
            make_folds(self.config["root_dir"])

        fold = json.load(open(join(self.config["root_dir"], "splits_final.json")))[self.config["fold"]]

        if self.mode == "train_val":
            id_list = fold["train"] + fold["val"]
        elif self.mode == "test":
            id_list_test = sorted(glob(join(self.config["root_dir"], "imagesTs", "*")))
            id_list = [basename(image).replace("_0000.nii.gz", "") for image in id_list_test]
        else:
            id_list = fold[self.mode]

        # load images
        images = {}
        str_ = "Ts" if self.mode == "test" else "Tr"
        for id_ in id_list:
            images[id_] = join(self.config["root_dir"], f"images{str_}", f"{id_}_0000.nii.gz")

        return id_list, images

    def load_h5(self, h5_path, augment=False):
        h5_file = self._get_h5_file(h5_path)

        # load image and label
        image, label = h5_file["CCTA"]["image"], h5_file["CCTA"]["label"]
        spacing, offset, landmarks = label.attrs["spacing"], label.attrs["offset"], label.attrs["landmarks"]

        # generate ray casts
        landmark = landmarks[np.random.randint(0, len(landmarks))]
        if augment:
            landmark, r = self.augment_landmark(landmark)
        else:
            r = self.r
        ray_casts = self.icosphere.ray_cast(image[:, :, :], spacing, offset, landmark, r)[None]
        ray_casts = self.norm(ray_casts)

        if augment:
            ray_casts = self.augment_intensities(ray_casts)

        if self.config["geometric"]:
            return Data(
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
        else:
            return ray_casts

    def augment_landmark(self, landmark):
        # randomize radius and landmark offset
        if np.random.random() < 0.2:
            r = np.random.uniform(self.r * 0.7, self.r * 1.4)
        else:
            r = self.r

        if hasattr(self, 'label'):
            if "Pericardium" in self.config["root_dir"]:
                landmark += np.random.uniform(-30, 30, size=3)
            else:
                landmark += np.random.uniform(-10, 10, size=3)
        else:
            landmark += np.random.uniform(-40, 40, size=3)
        return landmark, r

    @staticmethod
    def augment_intensities(ray_casts):
        ray_casts = ray_casts.detach().cpu().numpy()[None, None]
        if np.random.random() < 0.1:
            ray_casts = augment_gaussian_noise(ray_casts, (0, 0.1))
        if np.random.random() < 0.1:
            ray_casts = augment_gaussian_blur1d(ray_casts, (0.5, 1.5), axis=2)
        if np.random.random() < 0.15:
            ray_casts = augment_brightness_multiplicative(ray_casts, (0.75, 1.25))
        if np.random.random() < 0.15:
            ray_casts = augment_contrast(ray_casts, (0.75, 1.25))
        if np.random.random() < 0.125:
            ray_casts = augment_linear_downsampling_scipy1d(ray_casts, axis=1)
        if np.random.random() < 0.1:
            ray_casts = augment_gamma(ray_casts, (0.7, 1.5), retain_stats=True, invert_image=True)
        if np.random.random() < 0.3:
            ray_casts = augment_gamma(ray_casts, (0.7, 1.5), retain_stats=True)

        ray_casts = torch.from_numpy(ray_casts[0, 0])
        return ray_casts

    @staticmethod
    def norm(x: np.ndarray):
        x = np.clip(x, -360, 840)
        x = (x - 240) / 600
        return x

    def _get_h5_file(self, h5_path):
        # if the file is not in the cache, open it and cache the handle
        if h5_path not in self.h5_cache:
            self.h5_cache[h5_path] = h5py.File(h5_path, "r")
        return self.h5_cache[h5_path]

    def __del__(self):
        # ensure that all open h5py files are closed when the dataset is destroyed
        if hasattr(self, '_h5_cache'):
            for f in self.h5_cache.values():
                try:
                    f.close()
                except Exception:
                    pass

class SupervisedIcoSphereDataset(IcosphereDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = self.config["label"]

        if self.mode == "train":
            if "n_samples" in self.config:
                if self.config["n_samples"] is not None:
                    self.id_list = self.id_list[:self.config["n_samples"]]
            self.id_list = list(islice(cycle(self.id_list), self.config["size"]))

    def load_h5(self, h5_path, augment=False):
        h5_file = self._get_h5_file(h5_path)

        # load image and label
        image, label = h5_file["CCTA"]["image"], h5_file["CCTA"]["label"]
        spacing, offset, landmarks = label.attrs["spacing"], label.attrs["offset"], label.attrs["landmarks"]

        # generate ray casts
        landmark = landmarks[self.label - 1]
        if augment:
            landmark, r = self.augment_landmark(landmark)
        else:
            r = self.r
        ray_casts = self.icosphere.ray_cast(image[:, :, :], spacing, offset, landmark, r)[None]
        ray_casts = self.norm(ray_casts)

        if self.label == 5:
            # myocardium
            labels = torch.empty(ray_casts.shape[1:] + (2,), dtype=torch.float32)
            labels[:, :, 0] = self.icosphere.ray_cast(label[:, :, :] == 1, spacing, offset, landmark, r)
            labels[:, :, 0] = self.first_nonzero(labels[:, :, 0])
            labels[:, :, 1] = self.icosphere.ray_cast(label[:, :, :] == 5, spacing, offset, landmark, r)
        else:
            # atrioventricular/pericardium
            labels = torch.clone(self.icosphere.ray_cast(label[:, :, :] == self.label, spacing, offset, landmark, r))
            labels = self.first_nonzero(labels)
        labels = labels / self.icosphere.n_bins

        if augment:
            ray_casts = self.augment_intensities(ray_casts)

        if self.config["geometric"]:
            data = Data(
                x=ray_casts[0],
                y=labels,
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
            if augment:
                return data
            else:
                image_dict = {"shape": image.shape, "spacing": spacing, "offset": offset,
                              "landmark": landmark, "radius": r}
                for key, value in image_dict.items():
                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value)
                    data[key] = value
                return data
        else:
            return ray_casts, labels

    @staticmethod
    def first_nonzero(arr):
        zero = (arr <= 0)
        mask = zero.cummax(dim=1)[0]
        return torch.where(mask, torch.zeros_like(arr), arr)


class IcosphereDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["n_workers"]

        # Datasets will be created in the setup stage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if "label" in self.config:
            self.dataset = SupervisedIcoSphereDataset
        else:
            self.dataset = IcosphereDataset

        if self.config["geometric"]:
            self.dataloader = DataLoaderGeo
        else:
            self.dataloader = DataLoader

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.config["mode"] == "train_val":
                self.train_dataset = self.dataset(self.config, mode="train_val")
            else:
                self.train_dataset = self.dataset(self.config, mode="train")
            self.val_dataset = self.dataset(self.config, mode="val")

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset(self.config, mode="test")

    def train_dataloader(self):
        return self.dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,  # shuffle for training
        )

    def val_dataloader(self):
        return self.dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

class TubularDataset(Dataset):
    def __init__(self, config, mode="train"):
        super().__init__()
        self.config = config
        self.mode = mode

        # data dicts
        self.id_list, self.images = self.setup()
        self.h5_cache, self.mpr_cache = {}, {}

        # set transform
        self.l = config["sample_length"]
        self.transform = MPRTransform(config["mpr_transform"], device="cpu")
        self.transform_polar = PolarTransform(config["polar_transform"], device="cpu")

        # for step/epoch consistency
        if self.mode == "train":
            if "n_samples" in self.config:
                if self.config["n_samples"] is not None:
                    self.id_list = self.id_list[:self.config["n_samples"]]
            self.id_list = list(islice(cycle(self.id_list), self.config["size"]))

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id_ = self.id_list[index]
        path_image = self.images[id_]
        path_h5 = path_image.replace("images", "h5").replace("_0000.nrrd", ".h5")

        if self.mode == "train" or self.mode == "train_val":
            if self.config["from_h5"]:
                return self.load_h5(path_h5, augment=True)
            else:
                return self.load_mpr(path_h5, augment=True)
        else:
            return self.load_h5(path_h5)

    def setup(self):
        if not exists(join(self.config["root_dir"], "splits_final.json")):
            make_folds_asoca(self.config["root_dir"])

        fold = json.load(open(join(self.config["root_dir"], "splits_final.json")))[self.config["fold"]]

        if self.mode == "train_val":
            id_list = fold["train"] + fold["val"]
        elif self.mode == "test":
            id_list_test = sorted(glob(join(self.config["root_dir"], "imagesTs", "*")))
            id_list = [basename(image).replace("_0000.nrrd", "") for image in id_list_test]
        else:
            id_list = fold[self.mode]

        # load images
        images = {}
        str_ = "Ts" if self.mode == "test" else "Tr"
        for id_ in id_list:
            images[id_] = join(self.config["root_dir"], f"images{str_}", f"{id_}_0000.nrrd")

        return id_list, images

    def load_h5(self, h5_path, augment=False):
        h5_file = self._get_h5_file(h5_path)

        # load image and label
        image, label, ctls = h5_file["CCTA"]["image"], h5_file["CCTA"]["label"], h5_file["CCTA"]["centerlines"]
        spacing, offset = label.attrs["spacing"], label.attrs["offset"]
        offset[:2] *= -1

        valid_ctl = False
        while not valid_ctl:
            # random centerline
            sample = np.random.randint(0, np.max(ctls[:, 3]) + 1)
            ctl = ctls[ctls[:, 3] == sample, :3]
            ctl = ctl - offset

            if len(ctl) > self.l:
                valid_ctl = True
            else:
                print(f"Invalid centerline: {ctl.shape}")

        # random sub-sample
        if augment:
            sample = np.random.randint(0, ctl.shape[0] - self.l)
            ctl = ctl[sample:sample + self.l]

            # generate slightly morphed centerline
            ctl = self.augment_centerline(ctl)

        # generate MPR
        image, label = torch.from_numpy(image[:, :, :]).float(), torch.from_numpy(label[:, :, :]).float()
        ctl, spacing, offset = torch.from_numpy(ctl).float(), torch.tensor(spacing).float(), torch.tensor(offset).float()

        mpr = self.transform((image, spacing), ctl, mode="polar")
        mpr = self.norm(mpr)

        ref = self.transform((label, spacing), ctl, mode="polar")
        ref = self.first_nonzero(ref > 0.5) * ref
        ref = ref / self.config["mpr_transform"]["ps_polar"][0]

        mpr_data = self.transform.generate_multiscale_graph_data(ctl)
        if self.config["geometric"]:
            mpr = einops.rearrange(mpr, "r theta z -> (z theta) r")
            ref = einops.rearrange(ref, "r theta z -> (z theta) r")
            return Data(
                x=mpr,
                y=ref,
                ctl=ctl + offset,
                sample=sample,
                connection=mpr_data.connection,
                edge_coords=mpr_data.edge_coords,
                edge_index=mpr_data.edge_index,
                frame=mpr_data.frame,
                normal=mpr_data.normal,
                pos=mpr_data.pos,
                weight=mpr_data.weight,
                node_mask=mpr_data.node_mask,
                edge_mask=mpr_data.edge_mask,
            )
        else:
            return mpr[None], ref[None], ctl + offset

    def load_mpr(self, path_h5, augment=False):
        mpr_data = self._get_mpr_data(path_h5)
        mpr_data = random.choice(mpr_data)

        x = torch.from_numpy(self.norm(mpr_data[0])).float()
        y = torch.from_numpy(mpr_data[1])
        vertices, normals = mpr_data[2].vertices.copy(), mpr_data[2].vertex_normals.copy()

        if augment:
            sample = np.random.randint(0, x.shape[2] - self.l)
            x = x[:, :, sample:sample + self.l]
            y = y[:, :, sample:sample + self.l]

            # get ctl sub-sample
            vertices = vertices[sample * self.transform.n_theta:(sample + self.l) * self.transform.n_theta]
            normals = normals[sample * self.transform.n_theta:(sample + self.l) * self.transform.n_theta]

        mpr_data = self.transform_polar(torch.stack((x, y), dim=0))
        mpr_data[:, 1] /= self.config["mpr_transform"]["ps_polar"][0]

        # get ctl data
        edges, edge_attrs = self.transform._generate_multiscale_edges(x.shape[2], [0, 3, 7])
        data = Data(pos=torch.from_numpy(np.array(vertices)).float(), normal=torch.from_numpy(normals).float())
        data = self.transform.multiscale_tube_graph(data, edges, edge_attrs)
        data = SimpleGeometry(gauge_def="random")(data)

        if self.config["geometric"]:
            mpr = einops.rearrange(mpr_data[0, 0], "z r theta -> (z theta) r")
            ref = einops.rearrange(mpr_data[0, 1], "z r theta -> (z theta) r")
            mpr, ref = torch.flip(mpr, [1]), torch.flip(ref, [1])
            return Data(
                x=mpr,
                y=ref,
                connection=data.connection,
                edge_coords=data.edge_coords,
                edge_index=data.edge_index,
                frame=data.frame,
                normal=data.normal,
                pos=data.pos,
                weight=data.weight,
                node_mask=data.node_mask,
                edge_mask=data.edge_mask,
            )
        else:
            return mpr_data[:, 0], mpr_data[:, 1]

    @staticmethod
    def augment_centerline(ctl, noise_std=0.5, smooth_sigma=2.0):
        noise = np.random.normal(0, noise_std, ctl.shape)
        smooth_noise = gaussian_filter1d(noise, sigma=smooth_sigma, axis=0)

        return ctl + smooth_noise

    @staticmethod
    def norm(x: np.ndarray):
        x = x - 1024 if x.min() >= -10 else x

        x = np.clip(x, -760, 1240)
        x = (x - 240) / 1000
        return x

    @staticmethod
    def first_nonzero(arr):
        zero = (arr <= 0)
        mask = zero.cummax(dim=0)[0]
        return torch.where(mask, torch.zeros_like(arr), arr)

    def _get_h5_file(self, h5_path):
        # if the file is not in the cache, open it and cache the handle
        if h5_path not in self.h5_cache:
            self.h5_cache[h5_path] = h5py.File(h5_path, "r")
        return self.h5_cache[h5_path]

    def _get_mpr_data(self, h5_path):
        # if the data is not yet loaded, load it
        if h5_path not in self.mpr_cache:
            id_ = basename(h5_path).replace(".h5", "")
            files_ref = sorted(glob(join(self.config["root_dir"], "mprsTr", id_, f"mpr_label*.nrrd")))
            files_mpr = [f.replace("mpr_label", "mpr") for f in files_ref]
            files_ctl = sorted(glob(join(self.config["root_dir"], "centerlinesTr", id_, "lumen_template*.stl")))

            mpr_data = []
            for mpr, ref, ctl in zip(files_mpr, files_ref, files_ctl):
                mpr, spacing, offset = sitk_to_numpy(mpr)
                ref, _, _ = sitk_to_numpy(ref)
                ctl = trimesh.load(ctl)

                if mpr.shape[2] > self.l and len(ctl.vertices) > self.l * self.transform.n_theta:
                    mpr_data.append((mpr, ref, ctl, spacing, offset))

            self.mpr_cache[h5_path] = mpr_data

        return self.mpr_cache[h5_path]

    def __del__(self):
        # ensure that all open h5py files are closed when the dataset is destroyed
        if hasattr(self, '_h5_cache'):
            for f in self.h5_cache.values():
                try:
                    f.close()
                except Exception:
                    pass


class TubularDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["n_workers"]

        # Datasets will be created in the setup stage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if self.config["geometric"]:
            self.dataloader = DataLoaderGeo
        else:
            self.dataloader = DataLoader

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.config["mode"] == "train_val":
                self.train_dataset = TubularDataset(self.config, mode="train_val")
            else:
                self.train_dataset = TubularDataset(self.config, mode="train")
            self.val_dataset = TubularDataset(self.config, mode="val")

        if stage == "test" or stage is None:
            self.test_dataset = TubularDataset(self.config, mode="test")

    def train_dataloader(self):
        return self.dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,  # shuffle for training
        )

    def val_dataloader(self):
        return self.dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )