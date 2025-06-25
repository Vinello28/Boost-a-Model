"""
ViT Feature Extractor Module
Gestisce l'estrazione di feature usando Vision Transformer (DINOv2)
"""

import torch
import torch.nn as nn
import torch.nn.modules.utils as nn_utils
import torch.nn.functional as F
from torchvision import transforms
import types
import math
import gc
from typing import Union, Tuple, List, Optional
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes cosine similarity between all possible pairs in two sets of vectors."""
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def _to_cartesian(coords, shape):
    """Takes raveled coordinates and returns them in a cartesian coordinate frame"""
    if torch.is_tensor(coords):
        coords = coords.long()

    # Calculate rows and columns for all indices
    width = shape[1]
    rows = coords // width
    cols = coords % width

    # Stack coordinates
    result = torch.stack([rows, cols], dim=-1)
    return result


def find_correspondences_batch(descriptors1, descriptors2, num_pairs=18, distance_threshold=1):
    """Find correspondences between two images using their descriptors."""
    B, _, t_m_1, d_h = descriptors1.size()
    num_patches = (int(np.sqrt(t_m_1)), int(np.sqrt(t_m_1)))

    # Calculate similarities
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    sim_1, nn_1 = torch.max(similarities, dim=-1)
    sim_2, nn_2 = torch.max(similarities, dim=-2)

    # Check if we're dealing with the same image
    is_same_image = sim_1.mean().item() > 0.99

    if is_same_image:
        # For same image, take random points
        num_points = min(num_pairs, t_m_1)

        # Generate random indices
        perm = torch.randperm(t_m_1, device=descriptors1.device)
        indices = perm[:num_points]

        # Convert to coordinates
        points1 = _to_cartesian(indices, num_patches)
        points2 = points1.clone()  # Same points for same image

        # Get similarity scores (should all be 1.0)
        sim_scores = torch.ones(num_points, device=descriptors1.device)

        return points1, points2, sim_scores

    else:
        # Original logic for different images
        nn_1, nn_2 = nn_1[:, 0, :], nn_2[:, 0, :]
        cyclical_idxs = torch.gather(nn_2, dim=-1, index=nn_1)

        # Create image indices
        image_idxs = torch.arange(t_m_1, device=descriptors1.device)[None, :].repeat(B, 1)

        # Convert to cartesian coordinates
        cyclical_idxs_ij = _to_cartesian(cyclical_idxs, shape=num_patches)
        image_idxs_ij = _to_cartesian(image_idxs, shape=num_patches)

        # Calculate distances
        b, hw, ij_dim = cyclical_idxs_ij.size()
        cyclical_dists = -torch.nn.PairwiseDistance(p=2)(
            cyclical_idxs_ij.view(-1, ij_dim),
            image_idxs_ij.view(-1, ij_dim)
        ).view(b, hw)

        # Normalize distances
        cyclical_dists_norm = cyclical_dists - cyclical_dists.min(1, keepdim=True)[0]
        cyclical_dists_norm /= (cyclical_dists_norm.max(1, keepdim=True)[0] + 1e-8)  # Add small epsilon

        # Sort values and get selected points
        sorted_vals, selected_points_image_1 = cyclical_dists_norm.sort(dim=-1, descending=True)

        # Filter points based on distance
        mask = sorted_vals >= distance_threshold
        filtered_points = selected_points_image_1[mask]

        # Select points
        num_available = filtered_points.numel()
        num_to_select = min(num_pairs, num_available)

        if num_to_select > 0:
            perm = torch.randperm(num_available, device=descriptors1.device)
            selected_indices = perm[:num_to_select]
            selected_points_image_1 = filtered_points[selected_indices].unsqueeze(0)

            # Get corresponding points in image 2
            selected_points_image_2 = torch.gather(nn_1, dim=-1, index=selected_points_image_1)

            # Get similarity scores
            sim_selected_12 = torch.gather(sim_1[:, 0, :], dim=-1, index=selected_points_image_1)

            # Convert to coordinates
            points1 = _to_cartesian(selected_points_image_1[0], num_patches)
            points2 = _to_cartesian(selected_points_image_2[0], num_patches)

            return points1, points2, sim_selected_12
        else:
            return None, None, None


def scale_points_from_patch(points, vit_image_size=518, num_patches=37):
    """Scale points from patch coordinates to pixel coordinates"""
    points = (points + 0.5) / num_patches * vit_image_size
    return points


def visualize_correspondences(image1, image2, points1, points2, save_path=None):
    """Visualize correspondences between two images."""
    if isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, Image.Image):
        image2 = np.array(image2)

    if torch.is_tensor(points1):
        points1 = points1.cpu().detach().numpy()
    if torch.is_tensor(points2):
        points2 = points2.cpu().detach().numpy()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(image1)
    ax2.imshow(image2)

    ax1.axis('off')
    ax2.axis('off')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(points1)))

    for i, ((y1, x1), (y2, x2), color) in enumerate(zip(points1, points2, colors)):
        ax1.plot(x1, y1, 'o', color=color, markersize=8)
        ax1.text(x1 + 5, y1 + 5, str(i), color=color, fontsize=8)

        ax2.plot(x2, y2, 'o', color=color, markersize=8)
        ax2.text(x2 + 5, y2 + 5, str(i), color=color, fontsize=8)

        con = ConnectionPatch(
            xyA=(x1, y1), xyB=(x2, y2),
            coordsA="data", coordsB="data",
            axesA=ax1, axesB=ax2, color=color, alpha=0.5
        )
        fig.add_artist(con)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


class ViTExtractor:
    """ This class facilitates extraction of features, descriptors, and saliency maps from a ViT.
    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(self, model_type: str = 'dinov2_vits14', stride: int = None, model: nn.Module = None, device: str = 'cuda'):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | dinov2_vits14 | dinov2_vitb14 |
                          vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
                       If None, automatically calculated to be compatible with patch_size.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        
        # Enhanced device detection and reporting
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
            print(f"🚀 ViTExtractor using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            print(f"⚠️  CUDA richiesta ma non disponibile - usando CPU")
            print(f"   Installa PyTorch con supporto CUDA per prestazioni migliori")
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
            print(f"💻 ViTExtractor using device: {self.device}")
        
        print(f"🔧 Model type: {model_type}")
        print(f"🎯 Target device: {self.device}")
        
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type)

        # Auto-calculate compatible stride if not provided
        if stride is None:
            stride = self._get_compatible_stride(self.model)
            print(f"Auto-calculated stride: {stride} for model {model_type}")

        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        self.p = self.model.patch_embed.patch_size
        if type(self.p) == tuple:
            self.p = self.p[0]
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | dinov2_vits14 | dinov2_vitb14 | vit_small_patch8_224 | vit_small_patch16_224 | 
                           vit_base_patch8_224 | vit_base_patch16_224]
        :return: the model
        """
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        
        if 'dinov2' in model_type:
            model = torch.hub.load('facebookresearch/dinov2', model_type)
        elif 'dino' in model_type:
            model = torch.hub.load('facebookresearch/dino:main', model_type)
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            import timm
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                'vit_small_patch16_224': 'dino_vits16',
                'vit_small_patch8_224': 'dino_vits8',
                'vit_base_patch16_224': 'dino_vitb16',
                'vit_base_patch8_224': 'dino_vitb8'
            }
            model = torch.hub.load('facebookresearch/dino:main', model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict['head.weight']
            del temp_state_dict['head.bias']
            model.load_state_dict(temp_state_dict)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if type(patch_size) == tuple:
            patch_size = patch_size[0]
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def preprocess(self, image_path: Union[str, Path],
                   load_size: Union[int, Tuple[int, int]] = None, patch_size: int = 14) -> Tuple[
        torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """

        def divisible_by_num(num, dim):
            return num * (dim // num)

        pil_image = Image.open(image_path).convert('RGB')
        if load_size is not None:
            pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)

            width, height = pil_image.size
            new_width = divisible_by_num(patch_size, width)
            new_height = divisible_by_num(patch_size, height)
            pil_image = pil_image.resize((new_width, new_height), resample=Image.LANCZOS)

        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image

    def preprocess_pil(self, pil_image):
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)

            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 1) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(B, bin_x.shape[1], self.num_patches[0], self.num_patches[1])
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3 ** k
            avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros((B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3 ** k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                            if i == y and j == x and k != 0:
                                continue
                            if 0 <= i < self.num_patches[0] and 0 <= j < self.num_patches[1]:
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                         :, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                         :, :, temp_i,
                                                                                                         temp_j]
                            part_idx += 1
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        # Bx1x(t-1)x(dxh)
        return bin_x

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, facet: str = 'key',
                            bin: bool = False, include_cls: bool = False) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        if facet == 'token':
            x.unsqueeze_(dim=1)  # Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert not bin, "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert self.model_type == "dino_vits8", f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], 'attn')
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0]  # Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1)  # Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
        return cls_attn_maps

    def detect_vit_features(self, goal_image, current_image, num_pairs=10, dino_input_size=518):
        """Detect features using DINOv2 - Ottimizzato e pulito"""
        print(f"🔍 Testing ViT Visual Servoing...")
        print(f"   Goal: {goal_image}")  
        print(f"   Current: {current_image}")
        print(f"   Metodo: Vision Transformer (DINOv2)")
        print(f"   🎯 Computing device: {self.device.upper()}")
        
        # Handle both file paths and PIL Image objects
        if isinstance(goal_image, Image.Image):
            goal_img = goal_image
            goal_name = "goal_image"
        else:
            goal_img = Image.open(goal_image).convert('RGB')
            goal_name = Path(goal_image).name
            
        if isinstance(current_image, Image.Image):
            current_img = current_image  
            current_name = "current_image"
        else:
            current_img = Image.open(current_image).convert('RGB')
            current_name = Path(current_image).name
        
        goal_image_resized = goal_img.resize((dino_input_size, dino_input_size))
        current_image_resized = current_img.resize((dino_input_size, dino_input_size))
        
        print(f"Processando: {goal_name} -> {current_name}")
        print(f"Metodo: ViT (Vision Transformer) su {self.device.upper()}")

        with torch.no_grad():
            # Process images using preprocess_pil
            goal_tensor = self.preprocess_pil(goal_image_resized)
            current_tensor = self.preprocess_pil(current_image_resized)

            # Extract descriptors using 'token' facet for better performance
            desc1 = self.extract_descriptors(
                goal_tensor.to(self.device),
                layer=11,
                facet='token',
                bin=False  # No binning for simplicity
            )
            desc2 = self.extract_descriptors(
                current_tensor.to(self.device),
                layer=11,
                facet='token', 
                bin=False
            )
            
            print(f"DEBUG: goal_descriptors shape: {desc1.shape}")
            print(f"DEBUG: current_descriptors shape: {desc2.shape}")

            # Find correspondences using the optimized batch function
            points1, points2, sim_selected_12 = find_correspondences_batch(
                desc1, desc2,
                num_pairs=num_pairs,
                distance_threshold=0.5
            )

            if points1 is None or points2 is None:
                print("❌ Errore: Nessuna corrispondenza trovata")
                return None, None

            print(f"✅ Trovate {len(points1)} corrispondenze")

            # Scale points from patch coordinates to pixel coordinates  
            scale = dino_input_size / int(np.sqrt(desc1.size(-2)))
            points1_scaled = points1 * scale + scale / 2
            points2_scaled = points2 * scale + scale / 2
            
            # Convert to numpy for compatibility
            points1_np = points1_scaled.cpu().numpy() if torch.is_tensor(points1_scaled) else points1_scaled
            points2_np = points2_scaled.cpu().numpy() if torch.is_tensor(points2_scaled) else points2_scaled

            # Scale to original image dimensions
            goal_scale_x = goal_img.width / dino_input_size
            goal_scale_y = goal_img.height / dino_input_size
            current_scale_x = current_img.width / dino_input_size  
            current_scale_y = current_img.height / dino_input_size

            # Apply scaling - coordinates are in [y, x] format, convert to [x, y]
            goal_points_final = np.column_stack([
                points1_np[:, 1] * goal_scale_x,  # x coordinates
                points1_np[:, 0] * goal_scale_y   # y coordinates  
            ])

            current_points_final = np.column_stack([
                points2_np[:, 1] * current_scale_x,  # x coordinates
                points2_np[:, 0] * current_scale_y   # y coordinates
            ])

            # Debug output for first few points
            print(f"🔍 DEBUG: Original image sizes: goal={goal_img.size}, current={current_img.size}")
            print(f"🔍 DEBUG: DINO input size: {dino_input_size}")
            print(f"🔍 DEBUG: Scale factors: goal=({goal_scale_x:.3f}, {goal_scale_y:.3f}), current=({current_scale_x:.3f}, {current_scale_y:.3f})")

            for i in range(min(3, len(goal_points_final))):
                print(f"🔍 Point {i+1}: goal=({goal_points_final[i][0]:.1f},{goal_points_final[i][1]:.1f}), current=({current_points_final[i][0]:.1f},{current_points_final[i][1]:.1f})")

            avg_similarity = sim_selected_12.mean().item() if torch.is_tensor(sim_selected_12) else np.mean(sim_selected_12)
            print(f"✅ ViT feature matching completato: {len(goal_points_final)} corrispondenze")
            print(f"📊 Similarità media: {avg_similarity:.4f}")

            return goal_points_final, current_points_final

    def _get_compatible_stride(self, model) -> int:
        """
        Calculate a compatible stride for the given model based on its patch_size.
        
        :param model: The ViT model
        :return: Compatible stride value
        """
        patch_size = model.patch_embed.patch_size
        if type(patch_size) == tuple:
            patch_size = patch_size[0]
        
        # Find the largest divisor of patch_size that gives good resolution
        # Prioritize smaller strides for higher resolution
        possible_strides = [i for i in range(1, patch_size + 1) if patch_size % i == 0]
        
        # For patch_size 14: divisors are [1, 2, 7, 14]
        # For patch_size 16: divisors are [1, 2, 4, 8, 16]
        # For patch_size 8: divisors are [1, 2, 4, 8]
        
        # Choose the best stride based on patch_size
        if patch_size == 14:
            # For DINOv2 models with patch_size 14, prefer stride 2 or 7
            if 2 in possible_strides:
                return 2  # High resolution
            elif 7 in possible_strides:
                return 7  # Medium resolution
            else:
                return possible_strides[0]  # Fallback to smallest
        elif patch_size == 16:
            # For standard ViT models with patch_size 16, prefer stride 4
            if 4 in possible_strides:
                return 4
            elif 2 in possible_strides:
                return 2
            else:
                return possible_strides[0]
        elif patch_size == 8:
            # For patch_size 8, prefer stride 2 or 4
            if 2 in possible_strides:
                return 2
            elif 4 in possible_strides:
                return 4
            else:
                return possible_strides[0]
        else:
            # Default: choose the smallest divisor greater than 1, or 1 if none exists
            return possible_strides[1] if len(possible_strides) > 1 else 1
