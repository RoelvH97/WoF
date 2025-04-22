# import necessary libraries
import numpy as np
import vtk
import SimpleITK as sitk

from vtk.util.numpy_support import vtk_to_numpy


def sitk_to_numpy(mhd_file, swap=True):
    image = sitk.ReadImage(mhd_file)
    spacing = image.GetSpacing()
    offset = image.GetOrigin()

    image = sitk.GetArrayFromImage(image)
    if swap:
        image = np.swapaxes(image, 0, 2)
    return image, spacing, offset


def numpy_to_sitk(image, spacing, offset, filename, swap=True):
    if swap:
        image = np.swapaxes(image, 0, 2)
    image = sitk.GetImageFromArray(image.astype(np.int16))
    image.SetSpacing(spacing.astype(float))
    image.SetOrigin(offset.astype(float))

    writer = sitk.ImageFileWriter()
    writer.SetUseCompression(True)
    writer.SetFileName(filename)
    writer.Execute(image)


def stl_to_mask(shape, spacing, offset, filename, extension=".nii.gz", save=True):
    # load the STL file
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    # apply the offset to the mesh
    transform = vtk.vtkTransform()
    transform.Translate([-o for o in offset])  # back to voxel space, so invert the offset

    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetInputConnection(reader.GetOutputPort())
    transform_filter.SetTransform(transform)

    # convert the mesh into a stencil
    data_to_stencil = vtk.vtkPolyDataToImageStencil()
    data_to_stencil.SetInputConnection(transform_filter.GetOutputPort())
    data_to_stencil.SetOutputSpacing(*spacing)
    data_to_stencil.SetOutputOrigin(*[0 for _ in spacing])  # voxel corner
    data_to_stencil.SetOutputWholeExtent(0, shape[0] - 1, 0, shape[1] - 1, 0, shape[2] - 1)

    # convert the stencil to an image
    stencil_to_image = vtk.vtkImageStencilToImage()
    stencil_to_image.SetInputConnection(data_to_stencil.GetOutputPort())
    stencil_to_image.SetOutsideValue(0)  # background value
    stencil_to_image.SetInsideValue(1)  # foreground value
    stencil_to_image.Update()

    # convert VTK image data to numpy array
    vtk_image = stencil_to_image.GetOutput()
    shape = vtk_image.GetDimensions()
    binary_mask = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
    binary_mask = binary_mask.reshape(shape[2], shape[1], shape[0])
    binary_mask = np.transpose(binary_mask, (2, 1, 0))

    if save:
        numpy_to_sitk(binary_mask, spacing, offset, f"{filename[:-4]}{extension}")
    return binary_mask
