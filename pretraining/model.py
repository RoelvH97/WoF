"""
adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py
"""
import e3nn
import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange
from utils import Icosahedron
from vit_pytorch.vit import Transformer


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # for spherical harmonics
        icosphere = Icosahedron(subdivisions=config["icosphere_subdivisions"], n_bins=1)
        self.pos_embedding = torch.from_numpy(icosphere.vertices).float()

        # image properties
        image_x, image_y = config["size_full"]
        patch_x, patch_y = config["size_patch"]
        assert image_x % patch_x == 0 and image_y % patch_y == 0,\
            'image dimensions must be divisible by the patch size.'

        # model properties
        dim, dim_head, n_channels, n_heads, n_layers = (config["dim"], config["dim_head"], config["n_channels"],
                                                        config["n_heads"], config["n_layers"])
        dropout, dropout_emb = (config["dropout"], config["dropout_emb"])
        self.dim = dim

        # embedding and un-embedding
        patch_dim = n_channels * patch_x * patch_y
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_x, p2=patch_y),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_image = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                  h=image_x // patch_x, w=image_y // patch_y,
                                  p1=patch_x, p2=patch_y)

        # actual transformer
        self.dropout = nn.Dropout(dropout_emb)
        self.transformer = Transformer(dim, n_layers, n_heads, dim_head, dim * 4, dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x += self.pos_embedding.to(x.device, dtype=x.dtype)

        x = self.dropout(x)
        x = self.transformer(x)

        return self.mlp_head(x.mean(dim=1))

class MAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = ViT(config["encoder"])
        config = config["decoder"]

        # model properties
        dim, dim_head, masking_ratio, n_heads, n_layers = (config["dim"], config["dim_head"], config["masking_ratio"],
                                                           config["n_heads"], config["n_layers"])

        assert 0 < masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        n_patches, dim_encoder = self.encoder.pos_embedding.shape[-2], self.encoder.dim
        ray_values_per_patch = self.encoder.to_patch_embedding[2].weight.shape[-1]

        self.to_image = self.encoder.to_image
        self.to_patch = self.encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*self.encoder.to_patch_embedding[1:])

        # set decoder
        self.decoder_dim = dim
        self.enc_to_dec = nn.Linear(dim_encoder, dim) if dim_encoder != dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(dim))
        self.decoder = Transformer(dim, n_layers, n_heads, dim_head, dim * 4)
        self.to_rays = nn.Linear(dim, ray_values_per_patch)

    def forward(self, x):
        mask_tokens, raw_decoder_tokens, masked_patches, _, masked_indices, _, _ = self.to_prefinal(x)
        pred_ray_values = self.to_rays(mask_tokens)

        return pred_ray_values, raw_decoder_tokens, masked_patches, masked_indices

    def to_pre_decode(self, x):
        device = x.device

        # get patches
        patches = self.to_patch(x)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)

        # random rotate the [-1, 1] spherical position embedding (rotation augmentation)
        rotmat = e3nn.o3.rand_matrix(batch)
        pos_embedding = torch.einsum('ij, bjk -> bik', self.encoder.pos_embedding, rotmat)

        # derive spherical harmonics
        pos_embedding = e3nn.o3.spherical_harmonics(list(range(8)), pos_embedding,
                                                    normalize=False, normalization="norm")

        # tile to match token size
        pos_embedding_inp = torch.tile(pos_embedding, (1, 1, tokens.shape[-1] // pos_embedding.shape[-1]))
        tokens += pos_embedding_inp.to(device, dtype=tokens.dtype)

        # calculate patch percentage that needs to be masked, and randomly divide it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal
        # - the paper says you can get away with a smaller dimension for decoder
        raw_decoder_tokens = self.enc_to_dec(encoded_tokens)

        # batch indices for masked and unmasked tokens
        pos_embedding = pos_embedding.to(device, dtype=tokens.dtype)
        batch_index_masked = torch.arange(batch).unsqueeze(1).expand(batch, num_masked)
        batch_index_unmasked = torch.arange(batch).unsqueeze(1).expand(batch, num_patches - num_masked)

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = raw_decoder_tokens + pos_embedding[batch_index_unmasked, unmasked_indices]

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + pos_embedding[batch_index_masked, masked_indices]

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        return batch_range, raw_decoder_tokens, decoder_tokens, encoded_tokens, masked_indices, masked_patches, unmasked_indices

    def to_prefinal(self, x):
        batch_range, raw_decoder_tokens, decoder_tokens, encoded_tokens, masked_indices, masked_patches, unmasked_indices = self.to_pre_decode(x)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to ray values
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        return mask_tokens, raw_decoder_tokens, masked_patches, batch_range, masked_indices, encoded_tokens, unmasked_indices

    def reconstruct_image(self, patches, masked_indices=None, pred_ray_values=None):
        """
        Reconstructs the image given patches. Can also reconstruct the masked image as well as the predicted image.
        To reconstruct the raw image from the patches, set masked_indices=None and pred_ray_values=None. To reconstruct
        the masked image, set masked_indices= the masked_indices tensor created in the `forward` call. To reconstruct the
        predicted image, set masked_indices and pred_ray_values = to their respective tensors created in the `forward` call.
        """
        patches = patches.cpu()

        masked_indices_in = masked_indices is not None
        predicted_rays_in = pred_ray_values is not None

        if masked_indices_in:
            masked_indices = masked_indices.cpu()
        if predicted_rays_in:
            pred_ray_values = pred_ray_values.cpu()

        reconstructed_image = patches.clone()
        if masked_indices_in or predicted_rays_in:
            for i in range(reconstructed_image.shape[0]):
                if masked_indices_in and predicted_rays_in:
                    reconstructed_image[i, masked_indices[i].cpu()] = pred_ray_values[i, :].cpu().float()
                elif masked_indices_in:
                    reconstructed_image[i, masked_indices[i].cpu()] = 0

        reconstructed_image = self.to_image(reconstructed_image)
        return reconstructed_image


class MAEncoder(MAE):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        x = self.to_patch(x)
        b, n, *_ = x.shape

        if self.training:
            # random rotate the [-1, 1] spherical position embedding (rotation augmentation)
            rotmat = e3nn.o3.rand_matrix(b)
            pos_embedding = torch.einsum('ij, bjk -> bik', self.encoder.pos_embedding, rotmat)
            pos_embedding = e3nn.o3.spherical_harmonics(list(range(8)), pos_embedding,
                                                        normalize=False, normalization="norm")
        else:
            pos_embedding = e3nn.o3.spherical_harmonics(list(range(8)), self.encoder.pos_embedding,
                                                        normalize=False, normalization="norm")

        # patch to encoder tokens and add positional embedding
        x = self.patch_to_emb(x)

        # tile to match token size
        pos_embedding_inp = torch.tile(pos_embedding, (1, 1, x.shape[-1] // pos_embedding.shape[-1]))
        x += pos_embedding_inp.to(x.device, dtype=x.dtype)

        # attend with vision transformer
        x = self.encoder.transformer(x)

        # project encoder to decoder dimensions
        return self.enc_to_dec(x)[:, None]
