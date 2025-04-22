# import necessary libraries
import numpy as np
import pyvista as pv
import torch
import trimesh

from gem_cnn.transform import SimpleGeometry
from stl import mesh
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import knn
from torch_sparse import coalesce


class Icosahedron:
    def __init__(self, subdivisions=0, n_bins=10):
        # icosahedron properties
        self.subdivisions = subdivisions
        self.n_bins = n_bins

        # initial icosahedron vertices, edges and faces
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        self.vertices = [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]]
        self.vertices = [v / np.linalg.norm(v) for v in self.vertices]

        self.edges = [
            [0, 1], [0, 5], [0, 7], [0, 10], [0, 11],
            [1, 5], [1, 7], [1, 8], [1, 9],
            [2, 3], [2, 4], [2, 6], [2, 10], [2, 11],
            [3, 4], [3, 6], [3, 8], [3, 9],
            [4, 5], [4, 9], [4, 11],
            [5, 9], [5, 11],
            [6, 7], [6, 8], [6, 10],
            [7, 8], [7, 10],
            [8, 9],
            [10, 11]]

        self.faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]

        # subdivide if necessary
        self.edges_full = self.edges
        self.faces_full = self.faces
        self.edge_attrs = [0] * len(self.edges)
        for _ in range(subdivisions):
            self.subdivide()

        # arrays to store the ray coordinates for each vertex direction on the unit sphere
        self.ray_coordinates = np.zeros((len(self.vertices), n_bins, 3), dtype=np.float32)
        self.ray_samples = np.zeros((len(self.vertices), n_bins), dtype=np.float32)
        self.displaced_and_scaled_coordinates = None  # for image-specific coordinates

        # precompute the ray coordinates on the unit sphere
        self.precompute_ray_coordinates(n_bins)

        # to arrays
        self.vertices = np.array(self.vertices)
        self.edges = np.array(self.edges)
        self.faces = np.array(self.faces)
        self.normals = self.compute_normals()

        self.edges_full = np.array(self.edges_full)
        self.faces_full = np.array(self.faces_full)
        self.edge_attrs = np.array(self.edge_attrs)

        # make undirected graph
        self.edges_full = np.concatenate((self.edges_full, self.edges_full[:, ::-1]), axis=0)
        self.edge_attrs = np.concatenate((self.edge_attrs, self.edge_attrs), axis=0)
        self.data = self.generate_gnn_data()

    def subdivide(self):
        """
        Subdivide the current geometry by creating new vertices at the midpoint
        of every edge and subdividing each triangle face into four new triangles.
        """
        midpoint_cache = {}

        def get_midpoint(point1, point2):
            # ensure unique representation of the midpoint between two points
            smaller_index = min(point1, point2)
            greater_index = max(point1, point2)
            key = f"{smaller_index}-{greater_index}"

            # if the midpoint is cached, return it
            if key in midpoint_cache:
                return midpoint_cache[key]

            # calculate, normalize and store the new midpoint
            midpoint = (self.vertices[point1] + self.vertices[point2]) / 2
            midpoint = midpoint / np.linalg.norm(midpoint)
            self.vertices.append(midpoint)
            midpoint_cache[key] = len(self.vertices) - 1
            return len(self.vertices) - 1

        new_edges = []
        new_faces = []
        for face in self.faces:
            # calculate midpoints for the current triangle
            midpoints = [get_midpoint(face[i], face[(i + 1) % 3]) for i in range(3)]

            # generate new edges from the original triangle and its midpoints
            new_edges.append([face[0], midpoints[0]])
            new_edges.append([face[1], midpoints[1]])
            new_edges.append([face[2], midpoints[2]])
            new_edges.append([midpoints[0], midpoints[1]])
            new_edges.append([midpoints[1], midpoints[2]])
            new_edges.append([midpoints[2], midpoints[0]])

            # generate four new triangles from the original triangle and its midpoints
            new_faces.append([face[0], midpoints[0], midpoints[2]])
            new_faces.append([face[1], midpoints[1], midpoints[0]])
            new_faces.append([face[2], midpoints[2], midpoints[1]])
            new_faces.append(midpoints)

        # update
        self.edges_full.extend(new_edges)
        self.faces_full.extend(new_faces)
        self.edge_attrs.extend([max(self.edge_attrs) + 1] * len(new_edges))

        self.edges = new_edges
        self.faces = new_faces

    def precompute_ray_coordinates(self, steps):
        """Precompute the ray coordinates on the unit sphere."""
        for idx, direction in enumerate(self.vertices):
            sample_coordinates = []
            current_point = np.zeros(3)  # starting at the origin
            direction = np.array(direction)

            # this can be more efficient, but we only do it once so whatever
            for _ in range(steps):
                current_point += direction / steps
                sample_coordinates.append(list(current_point))
            self.ray_coordinates[idx] = np.array(sample_coordinates)

    def compute_normals(self):
        # nothing special here
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        return mesh.vertex_normals

    def generate_gnn_data(self):
        edge_index = self.edges_full[self.edge_attrs == 4]
        edge_index = edge_index.T
        edge_index = torch.from_numpy(np.concatenate((edge_index, edge_index[::-1, :]), axis=1))

        data = Data(edge_index=edge_index,
                    pos=torch.from_numpy(self.vertices.astype(np.float32)),
                    normal=torch.from_numpy(self.normals.astype(np.float32)),)
        data = self.generate_multiscale_graph(data)
        data = SimpleGeometry()(data)
        return data

    def generate_multiscale_graph(self, data):
        data.edge_coords = None
        batch = data.batch if "batch" in data else None
        pos = data.pos

        # create empty tensors to store edge indices and masks
        edge_index = []
        edge_mask = []
        node_mask = torch.zeros(data.num_nodes)

        # we sample points on the surface based on the subdivision level of the icosahedron
        original_idx = torch.arange(data.num_nodes)
        batch = batch if batch is not None else torch.zeros(data.num_nodes, dtype=torch.long)
        for i, subdiv in enumerate(range(self.subdivisions, -1, -2)):
            edge_idx = self.edges_full[self.edge_attrs == subdiv]
            pool_idx = torch.from_numpy(np.unique(edge_idx))

            if subdiv != self.subdivisions:
                # add edges for pooling
                pool_neigh = knn(x=pos[pool_idx], y=pos, k=1, batch_x=batch[pool_idx], batch_y=batch)[1]
                edge_index.append(torch.stack((original_idx[pool_idx][pool_neigh], original_idx), dim=0))
                edge_mask.append(torch.ones(original_idx.shape[0], dtype=torch.long) * (0b1 << (i * 2)))

            # sample nodes
            original_idx = original_idx[pool_idx]
            pos = pos[pool_idx]
            batch = batch[pool_idx]
            node_mask[original_idx] = i

            # CONVOLUTION EDGES
            # add edges for convolution at subdiv resolution
            radius_edges = torch.from_numpy(edge_idx).T
            edge_index.append(radius_edges)
            edge_mask.append(torch.ones(radius_edges.size(1), dtype=torch.long) * (0b1 << (i * 2 + 1)))

        # add self loops
        for i in range(len(edge_index)):
            uniques = torch.unique(edge_index[i])
            edge_index[i] = torch.cat((edge_index[i], torch.stack((uniques, uniques))), 1)
            edge_mask[i] = torch.cat((edge_mask[i], torch.ones(uniques.size(0), dtype=torch.long) * edge_mask[i][0]), 0)

        # sort edges and combine duplicates with an add (=bitwise OR, as bitwise & gives 0) operation
        edge_index = torch.cat(edge_index, dim=1)
        edge_mask = torch.cat(edge_mask)
        edge_index, edge_mask = coalesce(edge_index, edge_mask, data.num_nodes, data.num_nodes, "add")

        # store in data object
        data.edge_index = edge_index
        data.node_mask = node_mask
        data.edge_mask = edge_mask
        return data

    def ray_cast(self, image, spacing, offset, center, radius):
        # grid sample pads with zero, so we need to shift the image intensity range
        min_ = image.min()
        if min_ < 0:
            image = image - min_

        # center to voxel space
        center = (center - np.array(offset)) / np.array(spacing)
        self.displaced_and_scaled_coordinates = self.ray_coordinates * radius / np.array(spacing) + np.array(center)

        coords_grid = (self.displaced_and_scaled_coordinates / np.array(image.shape)) * 2 - 1
        coords_grid = np.stack((coords_grid[:, :, 2], coords_grid[:, :, 1], coords_grid[:, :, 0]), axis=-1)
        ray_casts = F.grid_sample(torch.from_numpy(image[None, None].astype(np.float32)),
                                  torch.from_numpy(coords_grid[None, None].astype(np.float32)),
                                  align_corners=True)[0, 0, 0]

        if min_ < 0:
            ray_casts = ray_casts + min_
        return ray_casts

    def to_stl(self, radii, scale, center, filename, ref=None):
        vertices = self.vertices * radii * scale + np.array(center)

        mesh_stl = mesh.Mesh(np.zeros(len(self.faces), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(self.faces):
            for j in range(3):
                mesh_stl.vectors[i][j] = vertices[f[j]]
        mesh_stl.save(f"{filename}.stl")

        if ref is not None:
            diff = (radii - ref) * scale
            faces_pv = np.hstack([np.full((self.faces.shape[0], 1), 3), self.faces])

            mesh_pv = pv.PolyData(vertices, faces_pv)
            mesh_pv.point_data["error"] = diff
            mesh_pv.save(f"{filename}_error.vtk")

    def to_stl_myo(self, radii, scale, center, filename, ref=None, threshold=0.5):
        # get differences
        if ref is not None:
            diff_inner = (radii[:, 0] - ref[:, 0]) * scale
            diff_outer = ((radii[:, 0] + radii[:, 1]) - (ref[:, 0] + ref[:, 1])) * scale
        else:
            diff_inner, diff_outer = None, None

        outer_thickness = radii[:, 1] * scale
        outer_zero = np.where(outer_thickness < threshold)[0]  # smaller than threshold in mm

        try:
            outer_zero_dil = self.dilate(outer_zero, self.faces, 1)
        except ValueError:
            outer_zero_dil = outer_zero  # a bit ugly but this is Python so I do what I want
        outer_remove = outer_zero

        # inner_mesh
        radii_inner = self.vertices * radii[:, :1] * scale + np.array(center)
        vertices_inner, faces_inner, radii_inner, diff_inner = self.remove(outer_remove, radii_inner, diff_inner)

        # invert face normals
        faces_inner = faces_inner[:, ::-1]

        # outer mesh
        radii[:, 1][outer_zero_dil] = 0
        radii_outer = self.vertices * (radii[:, :1] + radii[:, 1:]) * scale + np.array(center)
        vertices_outer, faces_outer, radii_outer, diff_outer = self.remove(outer_remove, radii_outer, diff_outer)

        # stack vertices and faces
        vertices, faces, diff = self.merge((radii_inner, faces_inner),
                                           (radii_outer, faces_outer),
                                           (diff_inner, diff_outer))

        mesh_stl = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                mesh_stl.vectors[i][j] = vertices[f[j]]
        mesh_stl.save(f"{filename}.stl")

        if ref is not None:
            faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])

            mesh_pv = pv.PolyData(vertices, faces_pv)
            mesh_pv.point_data["error"] = diff
            mesh_pv.save(f"{filename}_error.vtk")

    @staticmethod
    def dilate(indices, faces, n=1):
        for _ in range(n):
            indices = np.unique(np.concatenate([np.where(faces == vertex)[0] for vertex in indices]))
            indices = np.unique(faces[indices])
        return indices

    @staticmethod
    def merge(inner, outer, diff=(None, None)):
        overlap = (inner[0] == outer[0]).sum(axis=1) == 3
        n_verts = len(inner[0])

        vertex_map = np.arange(n_verts)
        vertex_map[overlap] = np.where(overlap)[0]
        new_indices = np.arange(n_verts, n_verts + (~overlap).sum())
        vertex_map[~overlap] = new_indices

        vertices_merged = np.vstack([
            inner[0],                   # original inner vertices
            outer[0][~overlap]          # only non-overlapping outer vertices
        ])

        faces_outer_remapped = vertex_map[outer[1]]
        faces_outer_valid = np.all(faces_outer_remapped < len(vertices_merged), axis=1)
        faces_outer_valid = faces_outer_remapped[faces_outer_valid]
        faces_merged = np.vstack([inner[1], faces_outer_valid])

        diff_merged = None
        if diff[0] is not None and diff[1] is not None:
            diff_merged = np.concatenate([
                diff[0],                # original inner differences
                diff[1][~overlap]])     # only non-overlapping outer differences

        return vertices_merged, faces_merged, diff_merged

    def remove(self, to_remove, radii, diff=None):
        keep_mask = np.ones(len(self.vertices), dtype=bool)
        keep_mask[to_remove] = False

        # create index mapping for remapping edges and faces
        new_indices = np.cumsum(keep_mask) - 1
        new_vertices = self.vertices[keep_mask]

        face_mask = np.all(~np.isin(self.faces, to_remove), axis=1)
        new_faces = self.faces[face_mask]
        new_faces = new_indices[new_faces]

        # handle difference values if provided
        new_diff = None
        if diff is not None:
            new_diff = diff[keep_mask]

        return new_vertices, new_faces, radii[keep_mask], new_diff
