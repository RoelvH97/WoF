# import necessary libraries
import h5py
import json
import numpy as np
import os
import pyvista as pv
import vtk

from glob import glob
from os.path import basename, exists, join
from scipy import interpolate, ndimage
from shutil import copy2
from sklearn.model_selection import KFold
from .read_write import sitk_to_numpy


def make_folds(dset_path):
    kfold = KFold(n_splits=5, shuffle=True)

    images = sorted(glob(join(dset_path, "imagesTr", "*.nii.gz")))
    images = sorted(list(set([basename(image).replace("_0000.nii.gz", "") for image in images])))
    images = np.array(images)

    folds = []
    for i, (train_ix, val_ix) in enumerate(kfold.split(images)):
        train = list(images[train_ix])
        val = list(images[val_ix])
        folds.append({"train": train, "val": val})

    folds_file = join(dset_path, "splits_final.json")
    with open(folds_file, "w") as out_file:
        json.dump(folds, out_file)


"""Spherical things"""

def generate_landmarks(dset_path, mode="Tr"):
    """
    This function expects a path pointing to an nnUNetv2-like raw directory (e.g. Dataset001_Pericardium), with
    a set of training images (imagesTr) and corresponding reference segmentations (labelsTr). It will create a new
    subdirectory named landmarksTr, containing the landmarks of the reference segmentations in terms of the CoM.
    """
    os.makedirs(join(dset_path, f"landmarks{mode}"), exist_ok=True)

    labels = sorted(glob(join(dset_path, f"labels{mode}", "*.nii.gz")))
    for label in labels:
        name = basename(label).replace(".nii.gz", ".json")
        print(name)

        # get unique values
        label, spacing, offset = sitk_to_numpy(label)
        unique = np.unique(label)[1:]

        landmark_dict = {}
        for u in unique:
            # get CoM
            coords_local = ndimage.center_of_mass(label == u)
            coords_world = tuple(a * b + c + b / 2 for a, b, c in zip(coords_local, spacing, offset))
            landmark_dict[str(int(u))] = [coords_local, coords_world]

        if len(unique) > 1:
            # get CoM of the full reference segmentation (e.g. MMWHS)
            coords_local = ndimage.center_of_mass(label > 0)
            coords_world = tuple(a * b + c + b / 2 for a, b, c in zip(coords_local, spacing, offset))
            landmark_dict[str(int(u + 1))] = [coords_local, coords_world]

        landmark_file = join(dset_path, f"landmarks{mode}", name)
        with open(landmark_file, "w") as out_file:
            json.dump(landmark_dict, out_file)


def nnunet_to_hdf5(dset_path, mode="Tr", format=".nii.gz"):
    """
    This function expects a path pointing to an nnUNetv2-like raw directory (e.g. Dataset001_Pericardium), with
    a set of training images (imagesTr) and corresponding reference segmentations (labelsTr). It will create a new
    subdirectory named hdf5, containing the images and labels in a single HDF5 file. If landmarks are present, they
    will be added to the HDF5 file as well.
    """
    os.makedirs(join(dset_path, f"h5{mode}"), exist_ok=True)

    images = sorted(glob(join(dset_path, f"images{mode}", f"*{format}")))
    for image in images:
        name = basename(image).replace(f"_0000{format}", ".h5")
        print(name)

        label = image.replace(f"images{mode}", f"labels{mode}").replace(f"_0000{format}", format)
        landmarks = label.replace(f"labels{mode}", f"landmarks{mode}").replace(format, ".json")
        centerlines = label.replace(f"labels{mode}", f"centerlines{mode}").replace(format, ".vtp")

        # get data
        image, spacing, offset = sitk_to_numpy(image)
        label, _, _ = sitk_to_numpy(label)

        # save to HDF5
        h5_file = join(dset_path, f"h5{mode}", name)
        with h5py.File(h5_file, "w") as f:
            group = f.create_group("CCTA")

            data = group.create_dataset("image", data=image, compression="lzf")
            data.attrs["spacing"] = spacing
            data.attrs["offset"] = offset

            data = group.create_dataset("label", data=label, compression="lzf")
            data.attrs["spacing"] = spacing
            data.attrs["offset"] = offset

            if exists(landmarks):
                with open(landmarks, "r") as in_file:
                    landmarks = json.load(in_file)
                    landmarks = np.stack([np.array(mark[1]) for mark in landmarks.values()], axis=0)
                    data.attrs["landmarks"] = landmarks

            if exists(centerlines):
                mesh = pv.read(centerlines)
                points, lines = mesh.points, mesh.lines

                combined_list = []

                i, cline_id = 0, 0
                while i < len(lines):
                    n_pts = int(lines[i])

                    # extract the indices for this centerline
                    indices = lines[i + 1: i + 1 + n_pts].astype(int)
                    cl_coords = points[indices]

                    # resample the centerline to a spacing of 0.5.
                    resampled = resample_line(cl_coords, spacing=0.5)

                    # create an array with the centerline id in the 4th column.
                    num_resampled = resampled.shape[0]
                    cl_with_id = np.hstack((resampled, np.full((num_resampled, 1), cline_id, dtype=np.float32)))
                    combined_list.append(cl_with_id)

                    # move to next centerline in the flat lines array.
                    i += n_pts + 1
                    cline_id += 1

                ctls = np.vstack(combined_list)
                group.create_dataset("centerlines", data=ctls, compression="lzf")


"""Tubular things"""

def make_folds_asoca(dset_path, shuffle=False):
    kfold = KFold(n_splits=5, shuffle=shuffle)

    images_train = sorted(glob(join(dset_path, "imagesTr", "*.nrrd")))
    images_train = sorted(list(set([basename(image).replace("_0000.nrrd", "")
                                    for image in images_train if "DISEASED" in image])))
    images_train = np.array(images_train)

    folds = []
    for i, (train_ix, val_ix) in enumerate(kfold.split(images_train)):
        # make sure that ED and ES are always in the same set
        train = list(images_train[train_ix])
        train_new = [i.replace("DISEASED", "NORMAL") for i in train]
        train = sorted(train + train_new)

        val = list(images_train[val_ix])
        val_new = [i.replace("DISEASED", "NORMAL") for i in val]
        val = sorted(val + val_new)
        folds.append({"train": train, "val": val})

    folds_file = join(dset_path, "splits_final.json")
    with open(folds_file, "w") as out_file:
        json.dump(folds, out_file)


def asoca_to_nnunet(raw_path, out_path):
    """
    This function expects a path pointing to a directory containing the ASOCA data (e.g. ASOCADataAccess).
    It will create a new directory at out_path, containing the images and labels in the nnUNetv2-like format.
    """
    for subdir in ["Normal", "Diseased"]:
        for subsubdir, subdir_out in zip(["CTCA", f"Testset_{subdir}"], ["Tr", "Ts"]):
            path_write = join(out_path, f"images{subdir_out}")
            os.makedirs(path_write, exist_ok=True)

            # process images
            images = sorted(glob(join(raw_path, subdir, subsubdir, f"*.nrrd")))
            for image in images:
                if subdir_out == "Tr":
                    id_ = basename(image).split(".")[0].split("_")[1]
                else:
                    id_ = basename(image).split(".")[0]

                file_write = join(path_write, f"{subdir.upper()}_{id_.zfill(3)}_0000.nrrd")
                copy2(image, file_write)
                print(f"Copied image: {basename(image)} -> {basename(file_write)}")

            # process available reference segmentations
            if subsubdir == "CTCA":
                path_write = join(out_path, f"labels{subdir_out}")
                os.makedirs(path_write, exist_ok=True)

                labels = sorted(glob(join(raw_path, subdir, "Annotations", f"{subdir}_*.nrrd")))
                for label in labels:
                    id_ = basename(label).split(".")[0].split("_")[1]

                    file_write = join(path_write, f"{subdir.upper()}_{id_.zfill(3)}.nrrd")
                    copy2(label, file_write)
                    print(f"Copied label: {basename(label)} -> {basename(file_write)}")

                path_write = join(out_path, f"centerlines{subdir_out}")
                os.makedirs(path_write, exist_ok=True)

                centerlines = sorted(glob(join(raw_path, subdir, "Centerlines", f"{subdir}_*.vtp")))
                for centerlines_pat in centerlines:
                    id_ = basename(centerlines_pat).split(".")[0].split("_")[1]

                    file_write = join(path_write, f"{subdir.upper()}_{id_.zfill(3)}.vtp")
                    copy2(centerlines_pat, file_write)
                    print(f"Copied centerlines: {basename(centerlines_pat)} -> {basename(file_write)}")


def resample_line(coords, spacing=0.5):
    """
    Resample a 3D curve (centerline) so that the spacing between consecutive points is ~spacing.
    """
    # compute cumulative arc-length along the curve
    seg_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cum_dist = np.insert(np.cumsum(seg_lengths), 0, 0)

    # create new equally spaced arc-length positions from 0 to total length.
    total_length = cum_dist[-1]

    # ensure at least two points.
    num_samples = max(int(np.ceil(total_length / spacing)) + 1, 2)
    new_dist = np.linspace(0, total_length, num_samples)

    # interpolate each coordinate axis independently.
    interp_func = interpolate.interp1d(cum_dist, coords, axis=0)
    resampled_coords = interp_func(new_dist)
    return resampled_coords
