import einops
import numpy as np
import pyvista as pv
import trimesh
import torch
import torch.nn.functional as F

from gem_cnn.transform import SimpleGeometry
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import knn
from torch_sparse import coalesce
from typing import Any, List, Tuple, Dict, Optional, Union, Literal


class MPRTransform(nn.Module):
    """
    Multi-Planar Reformation (MPR) Transform for coronary arteries in CCTA images.

    This class transforms a 3D CCTA volume along a vessel centerline to create straightened views of coronary arteries,
    either in Cartesian (du, dv, z) or polar (r, angle, z) coordinates.
    """
    def __init__(self, config, device="cuda", max_points=600):
        """
        Args:
            config: Configuration dictionary with keys:
                - ps: Patch size for Cartesian coordinates [height, width]
                - spacing: Voxel spacing [dx, dy, dz]
                - ps_polar: Patch size for polar coordinates [radial_steps, angular_steps]
                - spacing_polar: Spacing for polar coordinates
                - resample: Boolean indicating whether to resample the centerline
            device: Device to run computations on ("cuda" or "cpu")
            max_points: Maximum number of centerline points to support
        """
        super(MPRTransform, self).__init__()
        self.config = config
        self.device = torch.device(device)

        # extract configuration parameters
        self.patch_size = config["ps"]
        self.spacing = config["spacing"]
        self.patch_size_polar = config["ps_polar"]
        self.spacing_polar = config["spacing_polar"]

        self.n_theta = self.patch_size_polar[1]
        self.max_points = max_points

        # calculate margins (half the patch size)
        self.margin = (torch.tensor(self.patch_size, dtype=torch.float32, device=self.device) - 1) / 2
        self.margin_polar = (torch.tensor(self.patch_size_polar, dtype=torch.float32, device=self.device) - 1) / 2

        # initialize sampling grids
        self.cartesian_grid = self._init_cartesian_grid()
        self.polar_grid = self._init_polar_grid()

    def _create_grid(self, height: int, width: int, depth: int) -> torch.Tensor:
        """
        Create a generic 2D grid repeated along a third dimension.
        """
        # create steps in both directions
        u_steps = torch.linspace(0, height, height)
        v_steps = torch.linspace(0, width, width)

        # create the 2D grid
        u_grid, v_grid = torch.meshgrid(u_steps, v_steps, indexing="ij")
        grid = torch.stack([u_grid, v_grid], dim=0)

        # repeat for each depth point
        grid = einops.repeat(grid, "uv X Y -> uv X Y N", N=depth).clone()
        return grid.to(self.device)

    def _init_cartesian_grid(self) -> torch.Tensor:
        return self._create_grid(
            self.patch_size[0],
            self.patch_size[1],
            self.max_points
        )

    def _init_polar_grid(self) -> torch.Tensor:
        return self._create_grid(
            self.patch_size_polar[0],
            self.patch_size_polar[1],
            self.max_points
        )

    def _generate_angles(self, n_samples: int, repeat: int = None) -> torch.Tensor:
        """
        Generate evenly spaced angles around a circle.
        """
        angles = torch.linspace(0, 2 * torch.pi, n_samples, device=self.device)

        if repeat is not None:
            angles = angles.unsqueeze(0).repeat(repeat, 1)

        return angles

    def _rotate_vectors_around_axis(self,
                                    axis: torch.Tensor,
                                    reference: torch.Tensor,
                                    angles: torch.Tensor) -> torch.Tensor:
        """
        Rotate vectors around an axis using Rodrigues' rotation formula.

        Args:
            axis: Unit vectors representing rotation axes, shape [..., 3]
            reference: Reference vectors to rotate, shape [..., 3]
            angles: Angles to rotate by, shape depends on application
        """
        # shape the inputs for broadcasting
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        if angles.dim() == 1:
            # for vessel mesh case: expand cos/sin for broadcasting
            cos_expanded = cos_angles.unsqueeze(0).unsqueeze(2)
            sin_expanded = sin_angles.unsqueeze(0).unsqueeze(2)

            # calculate cross product for second reference vector
            second_ref = torch.cross(axis, reference, dim=1)

            # apply rotation
            result = (reference.unsqueeze(1) * cos_expanded +
                      second_ref.unsqueeze(1) * sin_expanded)

        else:
            # for MPR case: already properly shaped for broadcasting
            cos_theta = cos_angles.unsqueeze(1)
            sin_theta = sin_angles.unsqueeze(1)

            # calculate dot product for projection along axis
            dot_product = torch.sum(reference.unsqueeze(2) * axis.unsqueeze(2), dim=1, keepdim=True)

            # apply Rodrigues' rotation formula
            result = (reference.unsqueeze(2) * cos_theta +
                      torch.cross(axis, reference, dim=1).unsqueeze(2) * sin_theta +
                      axis.unsqueeze(2) * (1 - cos_theta) * dot_product)

        return result

    def resample_centerline(self,
                            centerline: torch.Tensor,
                            resolution: Optional[float] = None,
                            size: Optional[int] = None) -> torch.Tensor:
        """
        Resample the centerline to either a specific resolution or size.

        Args:
            centerline: Tensor of shape [N, 3] representing 3D points along the centerline
            resolution: desired spacing between points in mm (optional)
            size: Desired number of points (optional)
        """
        # validate parameters
        if size is not None and resolution is not None:
            raise ValueError("Only one of resolution or size should be specified")
        if size is None and resolution is None:
            raise ValueError("Either resolution or size must be specified")

        # determine sampling points
        if size is not None:
            l_steps = torch.linspace(-1, 1, size)
        else:
            # calculate total length of centerline
            diff = centerline[1:] - centerline[:-1]
            length = torch.norm(diff, dim=1).sum()  # total length in mm
            num_points = int(torch.round(length / resolution))
            l_steps = torch.linspace(-1, 1, num_points)

        # create sampling grid
        xyz_steps = torch.linspace(-1, 1, 3)
        l_grid, xyz_grid = torch.meshgrid(l_steps, xyz_steps, indexing="ij")
        grid = torch.stack([xyz_grid, l_grid], dim=-1).to(self.device)

        # apply grid sampling
        return F.grid_sample(centerline[None, None], grid[None], align_corners=True)[0, 0]

    def compute_tangents(self, centerline: torch.Tensor) -> torch.Tensor:
        """
        Compute tangent vectors along the centerline.
        """
        # for interior points, use central difference
        interior_tangents = centerline[2:] - centerline[:-2]
        interior_tangents = interior_tangents / torch.norm(interior_tangents, dim=1, keepdim=True)

        # for endpoints, use forward/backward difference
        first_tangent = centerline[1] - centerline[0]
        first_tangent = first_tangent / torch.norm(first_tangent)

        last_tangent = centerline[-1] - centerline[-2]
        last_tangent = last_tangent / torch.norm(last_tangent)

        # combine
        return torch.cat([first_tangent[None], interior_tangents, last_tangent[None]], dim=0).to(self.device)

    def compute_single_frame(self, tangent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute orthogonal frame vectors (u,v) from a single tangent vector.
        """
        # choose an arbitrary reference vector not parallel to the tangent
        reference = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        if torch.dot(tangent, reference) > 0.9:
            reference = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)

        # compute u and v using cross products to form orthogonal frame
        u = torch.cross(tangent, reference)
        u = u / torch.norm(u)
        v = torch.cross(tangent, u)

        return u, v

    def compute_continuous_frame(self, tangents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a continuous, non-flipping frame along the centerline.
        """
        num_points = len(tangents)
        u_vectors = torch.empty((num_points, 3), dtype=torch.float32, device=self.device)
        v_vectors = torch.empty((num_points, 3), dtype=torch.float32, device=self.device)

        # initialize the first frame
        u_vectors[0], v_vectors[0] = self.compute_single_frame(tangents[0])

        # propagate the frame along the centerline
        for i in range(1, num_points):
            # compute new u vector perpendicular to both current tangent and previous v
            u = torch.cross(tangents[i], v_vectors[i - 1])
            u = u / torch.norm(u)
            v = torch.cross(tangents[i], u)

            # ensure consistent orientation (no sudden flips)
            if torch.dot(u, u_vectors[i - 1]) < 0:
                u = -u
                v = -v

            u_vectors[i] = u
            v_vectors[i] = v

        return u_vectors, v_vectors

    def get_coordinate_frame(self,
                             centerline: torch.Tensor,
                             return_tangents: bool = False) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Get the local coordinate frame (u,v vectors) for the centerline.
        """
        # compute frame
        tangent_vectors = self.compute_tangents(centerline)
        u_vectors, v_vectors = self.compute_continuous_frame(tangent_vectors)

        if return_tangents:
            return tangent_vectors, u_vectors, v_vectors
        else:
            return u_vectors, v_vectors

    def forward(self,
                input_data: Tuple,
                centerline: torch.Tensor,
                mode: Literal["default", "polar"] = "default") -> torch.Tensor:
        """
        Perform the MPR transformation.

        Args:
            input_data: Tuple containing (volume, spacing)
            centerline_data: List containing [centerline_points, optional_frame_data]
            mode: Transformation mode - "default" for Cartesian or "polar" for polar coordinates

        Returns:
            Transformed MPR volume
        """
        # unpack input data
        volume, spacing = input_data
        volume = volume.to(self.device)
        centerline = centerline.to(self.device)

        while len(volume.shape) < 4:
            volume = volume.unsqueeze(0)

        # resample centerline if needed
        if self.config["resample"]:
            centerline = self.resample_centerline(centerline, resolution=self.spacing[2])
        else:
            # calculate actual spacing along centerline
            segment_lengths = torch.norm(centerline[1:] - centerline[:-1], dim=1)
            actual_resolution = segment_lengths.mean()
            self.spacing[2] = actual_resolution.cpu().detach().item()

        if mode == "default":
            # Cartesian grid
            grid = self.cartesian_grid[..., :centerline.shape[0]]
            u_vectors, v_vectors = self.get_coordinate_frame(centerline)

            # transform the grid based on the centerline and coordinate frame
            xs = (
                    centerline[:, 0] +
                    (grid[0] - self.margin[0]) * u_vectors[:, 0] * self.spacing[0] +
                    (grid[1] - self.margin[1]) * v_vectors[:, 0] * self.spacing[1]
            )
            ys = (
                    centerline[:, 1] +
                    (grid[0] - self.margin[0]) * u_vectors[:, 1] * self.spacing[0] +
                    (grid[1] - self.margin[1]) * v_vectors[:, 1] * self.spacing[1]
            )
            zs = (
                    centerline[:, 2] +
                    (grid[0] - self.margin[0]) * u_vectors[:, 2] * self.spacing[0] +
                    (grid[1] - self.margin[1]) * v_vectors[:, 2] * self.spacing[1]
            )
        else:
            # polar grid
            grid = self.polar_grid[..., :centerline.shape[0]]
            tangent_vectors, u_vectors, v_vectors = self.get_coordinate_frame(centerline, return_tangents=True)

            # apply Rodrigues' rotation formula to get rotated vectors
            angles = self._generate_angles(self.n_theta, repeat=tangent_vectors.shape[0])
            rotated_vectors = self._rotate_vectors_around_axis(tangent_vectors, u_vectors, angles)

            # transform the grid based on the centerline and rotated vectors
            xs = centerline[:, 0] + grid[0] * rotated_vectors[:, 0].T * self.spacing_polar[0]
            ys = centerline[:, 1] + grid[0] * rotated_vectors[:, 1].T * self.spacing_polar[0]
            zs = centerline[:, 2] + grid[0] * rotated_vectors[:, 2].T * self.spacing_polar[0]

        # normalize coordinates to [-1, 1] range for grid_sample
        xs = (xs / spacing[0]) / volume.shape[1] * 2 - 1
        ys = (ys / spacing[1]) / volume.shape[2] * 2 - 1
        zs = (zs / spacing[2]) / volume.shape[3] * 2 - 1

        # apply grid_sample to transform the volume
        sampling_grid = torch.stack([zs, ys, xs], dim=-1)[None]
        return F.grid_sample(volume[None], sampling_grid, align_corners=True)[0, 0]

    def generate_vessel_graph(self,
                              centerline: torch.Tensor,
                              radius: float = 0.5,
                              scale_levels: List[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generate a multiscale graph representation of a vessel from its centerline.

        Args:
            centerline: Tensor of shape [N, 3] with centerline points
            radius: Radius of the tube in mm
            scale_levels: List of scale levels for multiscale edges (default: [0, 3, 7])

        Returns:
            Dictionary containing:
                - vertices: Tensor of shape [N*n_theta, 3] with vertex coordinates
                - normals: Tensor of shape [N*n_theta, 3] with vertex normals
                - edges: Tensor of shape [2, num_edges] with edge indices
                - edge_attrs: Tensor of shape [num_edges] with edge attributes (scale level)
        """
        if scale_levels is None:
            scale_levels = [0, 3, 7]

        # ensure centerline is a PyTorch tensor on the correct device
        if not isinstance(centerline, torch.Tensor):
            centerline = torch.tensor(centerline, dtype=torch.float32, device=self.device)
        centerline = centerline.to(self.device)

        # get tangents and UV vectors
        tangents = self.compute_tangents(centerline)
        u_vectors, v_vectors = self.compute_continuous_frame(tangents)

        # compute circular cross-sections by rotating u_vectors around tangents
        angles = self._generate_angles(self.n_theta)
        rotated_vectors = self._rotate_vectors_around_axis(tangents, u_vectors, angles)

        # number of axial and angular points
        n_z = centerline.shape[0]

        # generate vertices and normals
        vertices, normals = self._generate_vertices_and_normals(centerline, rotated_vectors, radius)

        # generate edges for the multiscale graph
        edges, edge_attrs = self._generate_multiscale_edges(n_z, scale_levels)

        return {
            "vertices": vertices.reshape(-1, 3),
            "normals": normals.reshape(-1, 3),
            "edges": edges,
            "edge_attrs": edge_attrs
        }

    def _generate_vertices_and_normals(self,
                                       centerline: torch.Tensor,
                                       rotated_vectors: torch.Tensor,
                                       radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # calculate vertices
        centerline_expanded = centerline.unsqueeze(1).expand(-1, self.n_theta, -1)  # [n_z, n_theta, 3]
        vertices = centerline_expanded + radius * rotated_vectors

        # normals are just the normalized rotated vectors
        normals = rotated_vectors / torch.norm(rotated_vectors, dim=2, keepdim=True)

        return vertices, normals

    def _generate_multiscale_edges(self,
                                   n_z: int,
                                   scale_levels: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiscale edge connectivity for the vessel graph using vectorized operations.

        Args:
            n_z: Number of points along centerline
            scale_levels: List of scale levels for multiscale edges

        Returns:
            edges: Tensor of shape [2, num_edges] with edge indices
            edge_attrs: Tensor of shape [num_edges] with edge attributes (scale level)
        """
        # --- Generate circular connections (vectorized) ---
        # create grid of all z-values and theta-indices
        z_indices = torch.arange(n_z, device=self.device)
        theta_indices = torch.arange(self.n_theta, device=self.device)

        z_grid, theta_grid = torch.meshgrid(z_indices, theta_indices, indexing="ij")

        # calculate source and target indices for circular connections
        source_indices = (z_grid * self.n_theta + theta_grid).flatten()
        target_indices = (z_grid * self.n_theta + (theta_grid + 1) % self.n_theta).flatten()

        # create forward and backward edges
        forward_edges = torch.stack([source_indices, target_indices], dim=1)
        backward_edges = torch.stack([target_indices, source_indices], dim=1)

        # combine into a single tensor
        circular_edges = torch.cat([forward_edges, backward_edges], dim=0)
        circular_attrs = torch.ones(circular_edges.shape[0], device=self.device)

        # --- Generate multiscale connections ---
        multiscale_edges_list = []
        multiscale_attrs_list = []

        for s in scale_levels:
            step = s + 1

            # generate valid z and theta coordinates at this scale\
            valid_z = torch.arange(step, n_z, step, device=self.device)
            valid_theta = torch.arange(0, self.n_theta, step, device=self.device)

            if len(valid_z) == 0 or len(valid_theta) == 0:
                continue

            # create a grid of all valid (z, theta) combinations
            z_ms_grid, theta_ms_grid = torch.meshgrid(valid_z, valid_theta, indexing="ij")

            # calculate the four quad corner indices for each valid pair
            # (current z, current theta)
            A = (z_ms_grid * self.n_theta + theta_ms_grid).flatten()
            # (previous z, current theta)
            B = ((z_ms_grid - step) * self.n_theta + theta_ms_grid).flatten()
            # (current z, next theta)
            C = (z_ms_grid * self.n_theta + ((theta_ms_grid + step) % self.n_theta)).flatten()
            # (previous z, next theta)
            D = ((z_ms_grid - step) * self.n_theta + ((theta_ms_grid + step) % self.n_theta)).flatten()

            # create all connections for this scale level in a single operation
            # forward connections (A to others)
            AB_edges = torch.stack([A, B], dim=1)
            AC_edges = torch.stack([A, C], dim=1)
            AD_edges = torch.stack([A, D], dim=1)

            # backward connections (others to A)
            BA_edges = torch.stack([B, A], dim=1)
            CA_edges = torch.stack([C, A], dim=1)
            DA_edges = torch.stack([D, A], dim=1)

            # combine all edges for this scale
            scale_edges = torch.cat([AB_edges, AC_edges, AD_edges,
                                     BA_edges, CA_edges, DA_edges], dim=0)

            # create edge attributes for this scale
            scale_attrs = torch.full((scale_edges.shape[0],), step, device=self.device)

            multiscale_edges_list.append(scale_edges)
            multiscale_attrs_list.append(scale_attrs)

        # combine circular and multiscale edges
        if multiscale_edges_list:
            all_edges = torch.cat([circular_edges] + multiscale_edges_list, dim=0)
            all_attrs = torch.cat([circular_attrs] + multiscale_attrs_list, dim=0)
        else:
            all_edges = circular_edges
            all_attrs = circular_attrs

        # transpose to get the desired output shape [2, num_edges]
        edges = all_edges.t()

        return edges, all_attrs

    def generate_multiscale_graph_data(self,
                                       centerline: torch.Tensor,
                                       radius: float = 0.5,
                                       scale_levels: List[int] = None) -> Any:
        """
        Generate a complete PyTorch Geometric Data object representing a multiscale
        vessel graph directly from a centerline.

        Args:
            centerline: Tensor of shape [N, 3] with centerline points
            radius: Radius of the tube in mm
            scale_levels: List of scale levels for multiscale edges (default: [0, 3, 7])

        Returns:
            PyTorch Geometric Data object with:
                - pos: Vertex positions
                - normal: Vertex normals
                - edge_index: Multiscale edge connections
                - edge_mask: Edge type/level masks
                - node_mask: Node hierarchy level masks
        """
        # generate the basic vessel graph
        vessel_graph = self.generate_vessel_graph(centerline, radius, scale_levels)

        # process into multiscale structure
        data = Data(pos=vessel_graph["vertices"], normal=vessel_graph["normals"])
        data = self.multiscale_tube_graph(data, vessel_graph["edges"], vessel_graph["edge_attrs"])
        return SimpleGeometry(gauge_def="random")(data)

    def multiscale_tube_graph(self, data, edges, edge_attrs):
        """
        Transform a vessel graph into a multiscale graph structure suitable for GNNs.

        Args:
            data: PyTorch Geometric Data object with vessel surface points
            edges: Edge indices tensor of shape [2, num_edges]
            edge_attrs: Edge attributes tensor of shape [num_edges] with scale information

        Returns:
            Modified Data object with multiscale graph structure
        """
        scales = torch.unique(edge_attrs)

        data.edge_coords = None
        batch = data.batch if hasattr(data, "batch") else None
        pos = data.pos

        # create empty lists to store edge indices and masks
        edge_index = []
        edge_mask = []
        node_mask = torch.zeros(data.num_nodes, device=self.device)

        # sample points on the surface based on the subdivision level
        original_idx = torch.arange(data.num_nodes, device=self.device)
        batch = batch if batch is not None else torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)

        for i, scale in enumerate(scales):
            edge_idx = edges[:, edge_attrs == scale]
            pool_idx = torch.unique(edge_idx)

            if scale == 8:
                # special handling for scale 8 (also applies at scale 4, but gives same pool_idx)
                a = pool_idx // (self.n_theta * scales[i - 1])
                b = (pool_idx % self.n_theta) // scales[i - 1]
                pool_idx = a * self.n_theta // scales[i - 1] + b
                pool_idx = pool_idx.int()

            if scale != 1:
                # add edges for pooling
                pool_neigh = knn(x=pos[pool_idx], y=pos, k=1,
                                 batch_x=batch[pool_idx], batch_y=batch)[1]
                edge_index.append(torch.stack((original_idx[pool_idx][pool_neigh],
                                               original_idx), dim=0))
                edge_mask.append(torch.ones(original_idx.shape[0],
                                            dtype=torch.long,
                                            device=self.device) * (0b1 << (i * 2)))

            # sample nodes
            original_idx = original_idx[pool_idx]
            pos = pos[pool_idx]
            batch = batch[pool_idx]
            node_mask[original_idx] = i

            # add edges for convolution at this scale
            radius_edges = torch.cat((edge_idx,
                                      torch.stack((torch.unique(edge_idx),
                                                   torch.unique(edge_idx)))), 1)

            edge_index.append(radius_edges)
            edge_mask.append(torch.ones(radius_edges.size(1),
                                        dtype=torch.long,
                                        device=self.device) * (0b1 << (i * 2 + 1)))

        # sort edges and combine duplicates with an add operation (bitwise OR)
        edge_index = torch.cat(edge_index, dim=1)
        edge_mask = torch.cat(edge_mask)
        edge_index, edge_mask = coalesce(
            edge_index, edge_mask, data.num_nodes, data.num_nodes, "add"
        )

        # store in data object
        data.edge_index = edge_index
        data.node_mask = node_mask
        data.edge_mask = edge_mask
        return data

    def to_stl(self, centerline, radii, filename, ref=None):
        """
        Args:
            centerline: Tensor of shape [N, 3] with centerline points
            radii: A tensor of shape [N, n_theta] for radius varying by centerline point and angle
        """
        # get tangents and UV vectors
        tangents = self.compute_tangents(centerline)
        u_vectors, v_vectors = self.compute_continuous_frame(tangents)
        angles = self._generate_angles(self.n_theta)

        # compute circular cross-sections by rotating u_vectors around tangents
        rotated_vectors = self._rotate_vectors_around_axis(tangents, u_vectors, angles)

        # reshape for broadcasting
        centerline_expanded = centerline.unsqueeze(1).expand(-1, self.n_theta, -1)  # [z, n_theta, 3]
        radius_expanded = radii.unsqueeze(2)

        # warp vertices
        vertices = centerline_expanded + radius_expanded * rotated_vectors.cpu().detach() * self.spacing_polar[0]

        # build tube faces
        z = centerline.shape[0]
        z_indices = torch.arange(z - 1, device=self.device)
        theta_indices = torch.arange(self.n_theta, device=self.device)
        z_grid, theta_grid = torch.meshgrid(z_indices, theta_indices, indexing="ij")

        current = (z_grid * self.n_theta + theta_grid).flatten()
        next_theta = (z_grid * self.n_theta + (theta_grid + 1) % self.n_theta).flatten()
        next_z = ((z_grid + 1) * self.n_theta + theta_grid).flatten()
        next_both = ((z_grid + 1) * self.n_theta + (theta_grid + 1) % self.n_theta).flatten()

        triangles1 = torch.stack([current, next_z, next_theta], dim=1)
        triangles2 = torch.stack([next_theta, next_z, next_both], dim=1)
        faces = torch.cat([triangles1, triangles2], dim=0)

        # reshape vertices to [z*n_theta, 3]
        vertices = vertices.reshape(-1, 3)

        # save mesh to STL file
        mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
        mesh.export(f"{filename}.stl")

        if ref is not None:
            diff = (radii - ref) * self.spacing_polar[0]
            diff = diff.reshape(-1)

            faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces.cpu().numpy()])

            mesh_pv = pv.PolyData(vertices.cpu().numpy(), faces_pv)
            mesh_pv.point_data["error"] = diff
            mesh_pv.save(f"{filename}_error.vtk")
