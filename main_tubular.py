# import necessary libraries
import argparse
import copy
import datetime
import json
import lightning.pytorch as pl
import os
import pyvista as pv
import torch

from data import TubularDataModule
from glob import glob
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from os.path import basename, dirname, join
from surface_meshing import LightningTube
from utils.preprocessing import resample_line


def train(config):
    now = datetime.datetime.now()
    now = f"{str(now.day).zfill(2)}-{str(now.month).zfill(2)}-{now.year}_"\
          f"{str(now.hour).zfill(2)}-{str(now.minute).zfill(2)}-{str(now.second).zfill(2)}"
    config_d, config_g, config_m, config_o = config["DATA"], config["GENERAL"], config["MODEL"], config["OPTIMIZATION"]

    # init
    pl.seed_everything(config["GENERAL"]["seed"])

    # save config
    config_file = join(os.getcwd(), f"lightning_logs/{config_g['name']}/{now}", "config.json")
    os.makedirs(dirname(config_file), exist_ok=True)
    with open(config_file, "w") as outfile:
        json.dump(config, outfile)

    data_module = TubularDataModule(config_d)
    model = LightningTube(config)

    # outline trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"lightning_logs/{config_g['name']}/{now}/model",
        save_top_k=5,
        save_last=True,
        monitor=f"val/{config_m['loss']}",
        mode="min",
    )
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=config_g["name"],
        version=now,
        default_hp_metric=False,
    )
    trainer = pl.Trainer(
        accumulate_grad_batches=1,
        callbacks=[checkpoint_callback,
                   LearningRateMonitor(logging_interval='epoch'),
                   RichProgressBar()],
        check_val_every_n_epoch=config_o["eval_every"],
        deterministic=False,
        devices=[config_o["gpu"]],
        gradient_clip_algorithm="norm",
        gradient_clip_val=config_o["clip_grad"],
        logger=logger,
        max_epochs=config_o["n_warmup"] + sum(config_o["scheduler"]["T_0"] * config_o["scheduler"]["T_mult"] ** i
                                              for i in range(config_o["n_cycles"])),
        num_sanity_val_steps=2,
    )

    ckpt = join(os.getcwd(), "lightning_logs", config_o["resume"], "model", "last.ckpt") if config_o["resume"] else None
    trainer.fit(model, data_module, ckpt_path=ckpt)


def infer(config):
    model = LightningTube(config)

    # load model parameters
    ckpt = os.path.join(config["MODEL"]["ckpt"], "model", "last.ckpt")
    state_dict = torch.load(ckpt, map_location=f"cuda:{config['OPTIMIZATION']['gpu']}")
    model.load_state_dict(state_dict["state_dict"])

    # iterate over dataset
    data_module = TubularDataModule(config["DATA"])
    data_module.setup()

    for image in data_module.val_dataset.id_list:
        name = join(config["MODEL"]["ckpt"], "outputs", image)
        print(f"Processing {name}")

        # extract centerlines from the .vtp file
        ctl_file = data_module.val_dataset.images[image].replace("images", "centerlines").replace("_0000.nrrd", ".vtp")
        mesh = pv.read(ctl_file)
        points, lines = mesh.points, mesh.lines

        ctls, i = [], 0
        while i < len(lines):
            n_pts = int(lines[i])

            # extract the indices for this centerline
            indices = lines[i + 1: i + 1 + n_pts].astype(int)
            ctl_coords = points[indices]

            # resample the centerline to a spacing of 0.5.
            ctl = resample_line(ctl_coords, spacing=config["DATA"]["mpr_transform"]["spacing"][-1])
            ctls.append(ctl)

            # move to next centerline in the flat lines array.
            i += n_pts + 1

        image = data_module.val_dataset.images[image]
        model.infer(image, ctls, name)


if __name__ == "__main__":
    # parse config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The configuration file')

    args = parser.parse_args()
    config = json.load(open(args.config))
    print(json.dumps(config, sort_keys=True, indent=4))

    if config['GENERAL']['mode'] == "train":
        name = copy.deepcopy(config['GENERAL']["name"])
        for i in range(5):
            config['GENERAL']["name"] = f"{name}_{i}"
            config["DATA"]["fold"] = i
            train(config)
    elif config['GENERAL']['mode'] == "infer":
        infer(config)
