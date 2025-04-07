import copy
import math
import warnings
from typing import Optional, Sequence, Union, Tuple
import mmengine
import numpy as np
import torch
from einops import einops
from mmengine import to_2tuple, MessageHub, dist
from mmengine.dist import get_dist_info
from mmengine.model import ModuleList, BaseModule
from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from transformers.models.mamba.modeling_mamba import MambaMixer
from mmdet.models import nlc_to_nchw, nchw_to_nlc
from mmpretrain.evaluation import Accuracy
from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.registry import MODELS
from mmpretrain.models import ImageClassifier, build_norm_layer, resize_pos_embed, ClsHead
from mmdet.structures.bbox import bbox2roi
from mmcv.cnn.bricks.transformer import PatchEmbed
import torch.nn.functional as F
from mmpretrain.structures import DataSample
from mmdet.models import FPN as MMDET_FPN
from mmseg.models.backbones.unet import BasicConvBlock
from mmseg.models.utils import UpConvBlock
from opencd.registry import MODELS as CDMODELS
from mmseg.registry import MODELS as MMSEG_MODELS

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
    warnings.warn('horovod is not installed')

try:
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer
except ImportError:
    Mamba2Mixer = None
    warnings.warn('mamba2 is not installed')


@MODELS.register_module()
class DynamicVisPretrainClassifier(ImageClassifier):
    def __init__(
            self,
            pre_neck=None,
            *args,
            **kwargs):
        super(DynamicVisPretrainClassifier, self).__init__(*args, **kwargs)
        if pre_neck is not None:
            self.pre_neck = MODELS.build(pre_neck)

    def extract_feat(self, inputs, data_samples):
        bboxes = [data_sample.gt_instances.bboxes for data_sample in data_samples]
        rois = bbox2roi(bboxes)
        x = self.backbone(inputs)
        if hasattr(self, 'pre_neck'):
            x = self.pre_neck(x)
        x = self.neck(x[:self.neck.num_inputs], rois)
        return x

    def loss(self, inputs, data_samples):
        feats = self.extract_feat(inputs, data_samples)
        return self.head.loss(feats, data_samples)

    def predict(self, inputs, data_samples, **kwargs):
        feats = self.extract_feat(inputs, data_samples)
        return self.head.predict(feats, data_samples, **kwargs)


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


@MODELS.register_module()
class MambaBlock(BaseModule):
    def __init__(
            self,
            path_type='forward_reverse_mean', # forward, shuffle, forward_reverse_mean, forward_reverse_gate, forward_reverse_shuffle_gate, forward_reverse_shuffle_mean
            embed_dims=768,
            layer_norm_epsilon=1e-5,
            layer_cfgs=dict(),
            layer_idx=0,
            mamba2=False,
            init_cfg=None
    ):
        super(MambaBlock, self).__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.pre_norm = MambaRMSNorm(self.embed_dims, eps=layer_norm_epsilon)

        if mamba2:
            print('Using Mamba2Mixer')
            head_dim = 24
            if layer_cfgs.get('head_dim', None):
                head_dim = layer_cfgs['head_dim']
            _layer_cfg = dict(
                hidden_size=self.embed_dims,
                expand=2,
                head_dim=head_dim,
                state_size=32,
                num_heads=self.embed_dims * 2 // head_dim,
                conv_kernel=4,
                time_step_rank=math.ceil(self.embed_dims / 16),
                use_conv_bias=True,
                hidden_act="silu",
                layer_norm_epsilon=1e-5,
                rms_norm=True,
                n_groups=2,
                chunk_size=256,
                time_step_limit=(0.0, float("inf")),
                time_step_min=0.001,
                time_step_max=0.1,
                time_step_floor=1e-4,
                use_bias=False,
            )
            _layer_cfg.update(layer_cfgs)
            _layer_cfg = mmengine.Config(_layer_cfg)
            self.mamba_layer = Mamba2Mixer(_layer_cfg, layer_idx)
        else:
            _layer_cfg = dict(
                hidden_size=self.embed_dims,
                state_size=16,
                intermediate_size=self.embed_dims * 2,
                conv_kernel=4,
                time_step_rank=math.ceil(self.embed_dims / 16),
                use_conv_bias=True,
                hidden_act="silu",
                use_bias=False,
                use_mambapy=True,
            )
            '''use_mambapy=True, Determines the fallback strategy during training if the CUDA-based official implementation of Mamba is not avaiable. 
            If `True`, the mamba.py implementation is used. If `False`, the naive and slower implementation is used. 
            Consider switching to the naive version if memory is limited.'''
            _layer_cfg.update(layer_cfgs)
            _layer_cfg = mmengine.Config(_layer_cfg)
            self.mamba_layer = MambaMixer(_layer_cfg, layer_idx)

        self.path_type = path_type
        if 'gate' in self.path_type:
            gate_out_dim = 2
            if 'shuffle' in self.path_type:
                gate_out_dim = 3
            self.gate_layer = nn.Sequential(
                nn.Linear(gate_out_dim*self.embed_dims, gate_out_dim, bias=False),
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        B, N, C = x.shape
        residual = x
        # x = self.pre_norm(x.to(dtype=self.pre_norm.weight.dtype))
        if 'forward' == self.path_type:
            x = self.pre_norm(x.to(dtype=self.pre_norm.weight.dtype))
            x = self.mamba_layer(x)

        if 'forward_reverse_mean' == self.path_type:
            x_inputs = [x, torch.flip(x, [1])]
            x_inputs = torch.cat(x_inputs, dim=0)
            x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
            x_inputs = self.mamba_layer(x_inputs)
            forward_x, reverse_x = torch.split(x_inputs, B, dim=0)
            x = (forward_x + torch.flip(reverse_x, [1])) / 2

        if 'forward_reverse_gate' == self.path_type:
            x_inputs = [x, torch.flip(x, [1])]
            x_inputs = torch.cat(x_inputs, dim=0)
            x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
            x_inputs = self.mamba_layer(x_inputs)
            forward_x, reverse_x = torch.split(x_inputs, B, dim=0)
            reverse_x = torch.flip(reverse_x, [1])
            mean_forward_x = torch.mean(forward_x, dim=1, keepdim=True)
            mean_reverse_x = torch.mean(reverse_x, dim=1, keepdim=True)
            gate = torch.cat([mean_forward_x, mean_reverse_x], dim=-1)
            gate = self.gate_layer(gate)
            x = gate[:, :, 0:1] * forward_x + gate[:, :, 1:2] * reverse_x

        if 'forward_reverse_shuffle_gate' == self.path_type:
            x_inputs = [x, torch.flip(x, [1])]
            rand_index = torch.randperm(x.size(1))
            x_inputs.append(x[:, rand_index])
            x_inputs = torch.cat(x_inputs, dim=0)
            x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
            x_inputs = self.mamba_layer(x_inputs)
            forward_x, reverse_x, shuffle_x = torch.split(x_inputs, B, dim=0)
            reverse_x = torch.flip(reverse_x, [1])
            # reverse the random index
            rand_index = torch.argsort(rand_index)
            shuffle_x = shuffle_x[:, rand_index]
            mean_forward_x = torch.mean(forward_x, dim=1, keepdim=True)
            mean_reverse_x = torch.mean(reverse_x, dim=1, keepdim=True)
            mean_shuffle_x = torch.mean(shuffle_x, dim=1, keepdim=True)
            gate = torch.cat([mean_forward_x, mean_reverse_x, mean_shuffle_x], dim=-1)
            gate = self.gate_layer(gate)
            x = gate[:, :, 0:1] * forward_x + gate[:, :, 1:2] * reverse_x + gate[:, :, 2:3] * shuffle_x

        if 'forward_reverse_shuffle_mean' == self.path_type:
            x_inputs = [x, torch.flip(x, [1])]
            rand_index = torch.randperm(x.size(1))
            x_inputs.append(x[:, rand_index])
            x_inputs = torch.cat(x_inputs, dim=0)
            x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
            x_inputs = self.mamba_layer(x_inputs)
            forward_x, reverse_x, shuffle_x = torch.split(x_inputs, B, dim=0)
            reverse_x = torch.flip(reverse_x, [1])
            # reverse the random index
            rand_index = torch.argsort(rand_index)
            shuffle_x = shuffle_x[:, rand_index]
            x = (forward_x + reverse_x + shuffle_x) / 3

        if 'shuffle' == self.path_type:
            rand_index = torch.randperm(x.size(1))
            x_inputs = x[:, rand_index]
            x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
            x_inputs = self.mamba_layer(x_inputs)
            # reverse the random index
            rand_index = torch.argsort(rand_index)
            x = x_inputs[:, rand_index]

        x = residual + x
        return x


def insert_global_token(x, global_token_pos, global_token=None):
    if global_token_pos == 'none' or global_token is None:
        return x

    B, N, C = x.shape
    if global_token_pos == 'headtail':
        x = torch.cat([global_token, x, global_token], dim=1)
    elif global_token_pos == 'mid':
        x = torch.cat([x[:, :N//2, ...], global_token, x[:, N//2:, ...]], dim=1)
    elif global_token_pos == 'head':
        x = torch.cat([global_token, x], dim=1)
    else:
        raise ValueError(f'global_token_pos={global_token_pos} is not supported')
    return x


def split_global_token(x, global_token_pos, global_token=None):
    if global_token_pos == 'none':
        return x

    B, N, C = x.shape
    _, N_global, _ = global_token.shape
    if global_token_pos == 'headtail':
        x = x[:, N_global:-N_global, ...]
    elif global_token_pos == 'mid':
        x = torch.cat([x[:, :N//2, ...], x[:, N//2+N_global:, ...]], dim=1)
    elif global_token_pos == 'head':
        x = x[:, N_global:, ...]
    else:
        raise ValueError(f'global_token_pos={global_token_pos} is not supported')
    return x


@MODELS.register_module()
class SpatialSparseMixer(BaseModule):
    def __init__(
            self,
            embed_dims,
            reduction_ratio=1,
            norm_topk_prob=False,
            path_type='forward_reverse_mean',
            layer_norm_epsilon=1e-5,
            layer_cfgs=dict(),
            layer_idx=0,
            sampling_scale=dict(type='fixed', val=0.2),  # fixed, decay
            global_token_cfg=dict(pos='none', num=1),  # pos: none, head, tail, mid; num: -1, 1, 2, 4
            is_softmax_on_x=False,
            loading_loss=False,
            mamba2=False,
            init_cfg=None
    ):
        super(SpatialSparseMixer, self).__init__(init_cfg)
        self.norm_topk_prob = norm_topk_prob
        self.reduction_ratio = reduction_ratio
        self.sampling_scale = sampling_scale
        self.loading_loss = loading_loss

        if self.reduction_ratio > 0:
            self.spatial_gate = nn.Linear(embed_dims, 1, bias=False)

        _layer_cfg = dict(
            path_type=path_type,
            embed_dims=embed_dims,
            layer_norm_epsilon=layer_norm_epsilon,
            layer_cfgs=layer_cfgs,
            layer_idx=layer_idx,
            mamba2=mamba2
        )
        self.spatial_mamba_mixer = MambaBlock(**_layer_cfg)

        self.global_token_cfg = global_token_cfg
        self.is_softmax_on_x = is_softmax_on_x

    def forward(self, inputs):
        if self.global_token_cfg['pos'] != 'none':
            global_token = F.adaptive_avg_pool1d(inputs.permute(0, 2, 1).contiguous(), self.global_token_cfg['num']).permute(0, 2, 1)
        else:
            global_token = None

        if self.reduction_ratio > 0:
            B, N, C = inputs.shape
            router_logits = self.spatial_gate(inputs).squeeze(-1)  # (B, N)
            original_routing_weights = F.softmax(router_logits, dim=1)

            if self.sampling_scale is not None and self.sampling_scale['val'] > 0 and self.training:
                if self.sampling_scale['type'] == 'fixed':
                    sampling_scale = self.sampling_scale['val']
                else:
                    # get the current epoch and max epochs
                    message_hub = MessageHub.get_current_instance()
                    current_epoch = message_hub.get_info('epoch')
                    max_epochs = message_hub.get_info('max_epochs')
                    # we decrease the sampling scale linearly between 0.00001 to self.sampling_scale
                    sampling_scale = (self.sampling_scale['val'] - 0.0001) * (1 - current_epoch / max_epochs) + 0.0001

                gumbel = torch.distributions.gumbel.Gumbel(0, sampling_scale).rsample
                tmp_router_logits = router_logits.detach() + gumbel(router_logits.shape).to(router_logits.device)
                tmp_routing_weights = F.softmax(tmp_router_logits, dim=1)
            else:
                tmp_routing_weights = original_routing_weights.detach()

            top_k = N // self.reduction_ratio
            _, selected_token_id = torch.topk(tmp_routing_weights, top_k, dim=-1)
            # original_routing_weights: (B, N)
            # selected_token: (B, top_k)
            # routing_weights: (B, top_k)
            routing_weights = original_routing_weights[torch.arange(B)[:, None], selected_token_id]

            if self.loading_loss:
                top_k_mask = F.one_hot(selected_token_id, num_classes=N)
            else:
                top_k_mask = None

            if self.norm_topk_prob:
                routing_weights = routing_weights / torch.sum(routing_weights, dim=-1, keepdim=True)

            current_state = inputs[torch.arange(B)[:, None], selected_token_id]

            current_state = insert_global_token(current_state, self.global_token_cfg['pos'], global_token)
            current_state = self.spatial_mamba_mixer(current_state)  # (B, top_k, C)
            current_state = split_global_token(current_state, self.global_token_cfg['pos'], global_token)
            current_state = current_state * routing_weights[:, :, None]  # (B, top_k, C)

            if self.is_softmax_on_x:
                residual_x = inputs * original_routing_weights[:, :, None]
                residual_x[torch.arange(B)[:, None], selected_token_id] = current_state
                new_inputs = inputs + residual_x
            else:
                new_inputs = inputs.clone()
                new_inputs[torch.arange(B)[:, None], selected_token_id] = inputs[torch.arange(B)[:, None], selected_token_id] + current_state
        else:
            # already has the residual connection
            inputs = insert_global_token(inputs, self.global_token_cfg['pos'], global_token)
            inputs = self.spatial_mamba_mixer(inputs)
            new_inputs = split_global_token(inputs, self.global_token_cfg['pos'], global_token)
            router_logits = None
            top_k_mask = None
        # get the loss to make selected more balanced in the training and add noise
        return new_inputs, (router_logits, top_k_mask)



@MODELS.register_module()
class DynamicVisBackbone(BaseBackbone):
    arch_zoo = {
        **dict.fromkeys(
            ['b', 'base'], {
                'c_embed_dims': [96, 192, 384, 768],
                'num_layers': [2, 4, 16, 4],
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'c_embed_dims': [128, 256, 512, 1024],
                'num_layers': [2, 4, 32, 4],
            }),
    }

    OUT_TYPES = {'featmap', 'avg_featmap'}
    def __init__(self,
                 mamba2=False,
                 arch='b',
                 frozen_stages=-1,
                 branch_format='spatial',  ## spatial, channel, spatial-channel-serial, spatial-channel-parallel
                 path_type='forward_reverse_mean', ## forward, shuffle, forward_reverse_mean, forward_reverse_gate, forward_reverse_shuffle_gate, forward_reverse_shuffle_mean
                 sampling_scale=dict(type='fixed', val=0.1), ## type: fixed, decay; val: 0.1, 0.2, 0.3; None
                 global_token_cfg=dict(pos='head', num=-1),  ## pos: none, head, tail, mid; num: -1, 1, 2, 4, 8, 16
                 img_size=512,
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 spatial_token_keep_ratios=[8, 4, 2, 1],
                 channel_token_keep_ratios=[1, 2, 2, 4],
                 in_channels=3,
                 out_indices=(0, 1, 2, 3),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 layer_cfgs=dict(),
                 norm_topk_prob=False,
                 is_softmax_on_x=True,
                 loading_loss=False,
                 with_pe=True,
                 out_type='avg_featmap', # featmap, avg_featmap
                 interpolate_mode='bicubic',
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d', mode='fan_in', nonlinearity='linear')
                 ]
                 ):
        super(DynamicVisBackbone, self).__init__(init_cfg)
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please choose from {self.OUT_TYPES}')
        self.out_type = out_type
        self.interpolate_mode = interpolate_mode
        self.frozen_stages = frozen_stages

        self.arch_settings = self.arch_zoo[arch]
        self.c_embed_dims = self.arch_settings['c_embed_dims']
        self.num_layers = self.arch_settings['num_layers']

        self.img_size = to_2tuple(img_size)
        self.path_type = path_type
        self.out_indices = out_indices
        self.global_token_cfg = global_token_cfg
        self.with_pe = with_pe

        if self.out_type == 'avg_featmap':
            self.post_meanpool_norm = build_norm_layer(norm_cfg, self.c_embed_dims[-1])

        cur = 0
        self.layers = ModuleList()

        input_size = img_size
        for i, num_layer in enumerate(self.num_layers):
            in_dims = in_channels if i == 0 else self.c_embed_dims[i - 1]
            out_dims = self.c_embed_dims[i]

            patch_embed = PatchEmbed(
                in_channels=in_dims,
                embed_dims=out_dims,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg,
                input_size=input_size,
            )
            input_size = patch_embed.init_out_size
            num_patches = patch_embed.init_out_size[0] * patch_embed.init_out_size[1]

            if i == 0:
                if with_pe:
                    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, out_dims))
                    self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

            global_token_cfg_ = copy.deepcopy(global_token_cfg)
            if global_token_cfg.get('num') == -1:
                global_token_cfg_['num'] = patch_embed.init_out_size[0]
            layer = ModuleList([
                SpatialSparseMixer(
                    mamba2=mamba2,
                    embed_dims=self.c_embed_dims[i],
                    reduction_ratio=spatial_token_keep_ratios[i],
                    norm_topk_prob=norm_topk_prob,
                    path_type=path_type,
                    layer_norm_epsilon=1e-5,
                    layer_cfgs=layer_cfgs,
                    layer_idx=cur + idx,
                    sampling_scale=sampling_scale,
                    global_token_cfg=global_token_cfg_,
                    is_softmax_on_x=is_softmax_on_x,
                    loading_loss=loading_loss,
                ) for idx in range(num_layer)
                # SCSparseMixer(
                #     branch_format=branch_format,
                #     spatial_embed_dims=out_dims,
                #     channel_embed_dims=(self.s_embed_dims[i], self.s_embed_dims[i]),
                #     spatial_reduction_ratio=spatial_sr_ratios[i],
                #     channel_reduction_ratio=channel_sr_ratios[i],
                #     norm_topk_prob=norm_topk_prob,
                #     path_type=path_type,
                #     norm_cfg=norm_cfg,
                #     layer_cfgs=layer_cfgs,
                #     layer_idx=cur + idx,
                #     sampling_scale=sampling_scale,
                #     global_token_pos=global_token_pos,
                #     is_softmax_on_x=is_softmax_on_x,
                #     loading_loss=False,
                # ) for idx in range(num_layer)
            ])
            norm = build_norm_layer(norm_cfg, out_dims)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer
        if self.frozen_stages > 0:
            self._freeze_stages()
        if mmengine.dist.is_main_process():
            self.print_trainable_parameters()

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = True
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers):
            if self.out_type == 'avg_featmap':
                self.post_meanpool_norm.eval()
                for param in self.post_meanpool_norm.parameters():
                    param.requires_grad = False

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            print(f'Cannot find {name} in state_dict')
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1])))
            pos_embed_shape = to_2tuple(
                int(np.sqrt(self.pos_embed.shape[1])))

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                num_extra_tokens=0)

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        not_trainable_param_names = []
        for name, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
            else:
                not_trainable_param_names.append(name)
        print(f"not trainable parameters: {set(not_trainable_param_names)}")
        print(f"not trainable parameters: {set(['.'.join(n.split('.')[0:2]) for n in not_trainable_param_names])}")
        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def forward(self, x):
        outs = []
        router_infos = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            if i == 0 and self.with_pe:
                x = x + self.pos_embed.to(device=x.device)
            for block in layer[1]:
                x, router_info = block(x)
                router_infos.append(router_info)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(self._format_output(x, hw_shape))
        return tuple(outs)

    def _format_output(self, x, hw_shape):
        if self.out_type == 'featmap':
            return x
        if self.out_type == 'avg_featmap':
            return self.post_meanpool_norm(torch.mean(x, dim=[2, 3]))
        return x


@MODELS.register_module()
class DynamicVisClsHead(ClsHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 load_balancing_loss_cfg: dict = None,  # dict(type='fixed', val=0.1)
                 init_cfg: Optional[dict] = dict(type='Normal', layer='Linear', std=0.01),
                 **kwargs
                 ):
        super(DynamicVisClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.load_balancing_loss_cfg = load_balancing_loss_cfg

    def loss(self, feats, data_samples, **kwargs) -> dict:
        # feats = featsdict['feats']
        # router_infos = featsdict['router_infos']
        cls_score = self(feats)

        if 'gt_score' in data_samples[0]:
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['cls_loss'] = loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                                     'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})
        # calculate load balancing loss
        if self.load_balancing_loss_cfg is not None and self.load_balancing_loss_cfg['val'] > 0:
            load_balancing_loss = load_balancing_loss_func(router_infos)

            if self.load_balancing_loss_cfg['type'] == 'fixed':
                factor = self.load_balancing_loss_cfg['val']
            else:
                message_hub = MessageHub.get_current_instance()
                current_epoch = message_hub.get_info('epoch')
                max_epochs = message_hub.get_info('max_epochs')
                factor = (self.load_balancing_loss_cfg['val'] - 0.00001) * (1 - current_epoch / max_epochs) + 0.00001
            losses['load_balancing_loss'] = factor * load_balancing_loss

        return losses

    def predict(self, feats, data_samples, **kwargs) -> dict:
        # feats = featsdict['feats']
        cls_score = self(feats)
        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def forward(self, feats):
        pre_logits = feats[-1]
        cls_score = self.fc(pre_logits)
        return cls_score

def load_balancing_loss_func(gate_logits):
    router_logits_with_length = {}
    topk_token_mask_with_length = {}
    for router_info in gate_logits:
        s_router_logits, s_top_k_mask, c_router_logits, c_top_k_mask = router_info
        if s_router_logits is not None:
            length = s_router_logits.size(-1)
            if length not in router_logits_with_length:
                router_logits_with_length[length] = []
                topk_token_mask_with_length[length] = []
            router_logits_with_length[length].append(s_router_logits)
            topk_token_mask_with_length[length].append(s_top_k_mask)
        if c_router_logits is not None:
            length = c_router_logits.size(-1)
            if length not in router_logits_with_length:
                router_logits_with_length[length] = []
                topk_token_mask_with_length[length] = []
            router_logits_with_length[length].append(c_router_logits)
            topk_token_mask_with_length[length].append(c_top_k_mask)

    overall_loss = 0
    for length, router_logits in router_logits_with_length.items():
        concatenated_gate_logits = torch.cat(router_logits, dim=0)
        routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
        top_k_mask = torch.cat(topk_token_mask_with_length[length], dim=0)

        # Compute the percentage of tokens routed to each position
        tokens_per_position = torch.mean(top_k_mask.float(), dim=0)
        # Compute the average probability of routing to these position
        router_prob_per_position = torch.mean(routing_weights, dim=0)
        loss = torch.sum(tokens_per_position * router_prob_per_position.unsqueeze(0))
        overall_loss += loss
    return overall_loss


@MODELS.register_module()
class MILCrossEntropy(nn.Module):
    def __init__(self):
        super(MILCrossEntropy, self).__init__()

    def forward(self, pred_logits, target, dim=-1, weighted_unk=False, weights=None, avg_positives=False):
        # if weighted_unk:
        #     pred_logits[target == 2] /= weighted_unk
        target[target == 2] = 0
        probs = F.softmax(pred_logits, dim=-1)
        # only consider the valid targets
        valid_mask = torch.any(target > 0, dim=-1)
        probs = probs[valid_mask]
        target = target[valid_mask]
        if len(target) == 0:
            return torch.tensor(0., device=pred_logits.device)
        if avg_positives:  # average the logits over positive targets
            loss = -torch.log(torch.sum(target * probs, dim=dim) / (torch.sum(target, dim=dim) + 1e-6))
        else:  # sum the logits over positive targets
            loss = -torch.log(torch.sum(target * probs, dim=dim))
        if weights is not None:
            return (loss * weights).mean()
        return loss.mean()


@MODELS.register_module()
class DynamicVisPretrainClsHead(ClsHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 with_mil=False,
                 mil_weight=0.5,
                 use_horovod=False,
                 init_cfg: Optional[dict] = dict(type='Normal', layer='Linear', std=0.01),
                 **kwargs
                 ):
        super(DynamicVisPretrainClsHead, self).__init__(init_cfg=init_cfg, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.with_mil = with_mil
        self.use_horovod = use_horovod
        if with_mil:
            self.category_embedding = nn.Embedding(num_classes, in_channels)
            self.mil_loss = MILCrossEntropy()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.mil_weight = mil_weight

    def gather_features(
            self,
            features,
            rank,
            world_size,
            local_loss=False,
            gather_with_grad=True,
            use_horovod=False,

    ):
        if use_horovod:
            assert hvd is not None, 'Please install horovod'
            if gather_with_grad:
                all_features = hvd.allgather(features)
            else:
                with torch.no_grad():
                    all_features = hvd.allgather(features)
                if not local_loss:
                    # ensure grads for local rank when all_* features don't have a gradient
                    gathered_features = list(all_features.chunk(world_size, dim=0))
                    gathered_features[rank] = features
                    all_features = torch.cat(gathered_features, dim=0)
        else:
            # We gather tensors from all gpus
            if gather_with_grad:
                all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
            else:
                data_size = torch.tensor(features.size(0), device=features.device)
                sizes_list = dist.all_gather(data_size)
                max_length = max(sizes_list)
                size_diff = max_length.item() - data_size.item()
                if size_diff:
                    padding = torch.zeros(
                        size_diff, *features.size()[1:], device=features.device, dtype=features.dtype)
                    features = torch.cat((features, padding))

                gather_list = dist.all_gather(features)

                all_data = []
                for tensor, size in zip(gather_list, sizes_list):
                    all_data.append(tensor[:size])

                if not local_loss:
                    # ensure grads for local rank when all_* features don't have a gradient
                    all_data[rank] = features[:data_size]
                all_features = torch.cat(all_data, dim=0)

        return all_features

    def loss(self, feats, data_samples, **kwargs) -> dict:
        cls_score = self(feats)
        target = torch.cat([data_sample.gt_instances.labels for data_sample in data_samples])

        losses = dict()
        loss = self.loss_module(cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['cls_loss'] = loss
        if self.with_mil:
            rank, world_size = get_dist_info()
            if isinstance(feats, tuple):
                object_feats = feats[-1]
            else:
                object_feats = feats
            object_feats = torch.mean(object_feats, dim=[2, 3])
            object_feats = F.normalize(object_feats, p=2, dim=-1)
            object_class_embeddings = self.category_embedding.weight
            object_class_embeddings = F.normalize(object_class_embeddings, p=2, dim=-1)
            if world_size > 1:
                all_feats = self.gather_features(object_feats, rank, world_size,
                                                 use_horovod=self.use_horovod,
                                                 local_loss=False,
                                                 gather_with_grad=False)
                all_target_labels = self.gather_features(target, rank, world_size,
                                                         use_horovod=self.use_horovod,
                                                         local_loss=False,
                                                         gather_with_grad=False)
            else:
                all_feats = object_feats
                all_target_labels = target
            all_labels = torch.zeros(all_feats.size(0), self.num_classes, device=all_feats.device)
            all_labels[torch.arange(all_feats.size(0)), all_target_labels] = 1

            logit_scale = self.logit_scale.exp()
            logits_per_feats = logit_scale * torch.matmul(all_feats, object_class_embeddings.t())
            logits_per_embedding = logits_per_feats.t()
            mil_loss = self.mil_loss(logits_per_feats, all_labels, weights=None, avg_positives=False)
            mil_loss += self.mil_loss(logits_per_embedding, all_labels.t(), weights=None, avg_positives=False)
            losses['mil_loss'] = self.mil_weight * mil_loss / 2.0

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                                     'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses

    def predict(self, feats, data_samples, **kwargs) -> dict:
        cls_score = self(feats)
        predictions = self._get_predictions(cls_score, data_samples=data_samples)
        return predictions

    def forward(self, feats):
        if isinstance(feats, tuple):
            pre_logits = feats[-1]
        else:
            pre_logits = feats
        pre_logits = torch.mean(pre_logits, dim=[2, 3])
        cls_score = self.fc(pre_logits)
        return cls_score

    def _get_predictions(self, cls_score, data_samples):
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        meta_keys = ('img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction')
        idx = 0
        for data_sample in data_samples:
            num_boxes = len(data_sample.gt_instances)
            for box_i in range(num_boxes):
                data_sample_cls = DataSample()
                data_sample_cls.set_gt_label(data_sample.gt_instances.labels[box_i])
                for key in meta_keys:
                    if key in data_sample:
                        data_sample_cls.set_field(data_sample.get(key), key, field_type='metainfo')
                data_sample_cls.set_pred_score(pred_scores[idx])
                data_sample_cls.set_pred_label(pred_labels[idx])
                out_data_samples.append(data_sample_cls)
                idx += 1
        return out_data_samples






# 未整理，Not Reorganized
@MODELS.register_module()
class ChannelSparseMixer(BaseModule):
    def __init__(
            self,
            embed_dims,
            reduction_ratio=1,
            norm_topk_prob=False,
            path_type='forward_reverse_mean',
            norm_cfg=dict(type='LN', eps=1e-6),
            layer_cfgs=dict(),
            layer_idx=0,
            sampling_scale=dict(type='fixed', val=0.2),
            global_token_pos='none',
            is_softmax_on_x=False,
            loading_loss=False,
            conv_embedding_dim=0,
    ):
        super(ChannelSparseMixer, self).__init__()
        assert isinstance(embed_dims, Union[Sequence]), 'embed_dims should be a list of int'
        self.norm_topk_prob = norm_topk_prob
        self.reduction_ratio = reduction_ratio
        self.sampling_scale = sampling_scale
        self.embed_dims = embed_dims
        self.loading_loss = loading_loss

        self.conv_embedding_dim = conv_embedding_dim

        if self.reduction_ratio > 1:
            self.channel_gate = nn.Linear(embed_dims[0]*embed_dims[1], 1, bias=False)
        _layer_cfg = dict(
            path_type=path_type,
            embed_dims=embed_dims[0]*embed_dims[1],
            norm_cfg=norm_cfg,
            layer_cfgs=layer_cfgs,
            layer_idx=layer_idx,
        )
        self.channel_mamba_mixer = MambaBlock(**_layer_cfg)
        self.global_token_pos = global_token_pos
        self.is_softmax_on_x = is_softmax_on_x

        if self.conv_embedding_dim > 0:
            self.conv_embedding_pre = nn.Sequential(
                nn.Conv2d(conv_embedding_dim, conv_embedding_dim, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
            )
            self.conv_embedding_post = nn.Sequential(
                nn.Conv2d(conv_embedding_dim, conv_embedding_dim, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
            )

    def forward(self, inputs, only_downsample=False):
        # downsample the spatial dimension
        # reshape the input tensor (B, N, C) to (B, C, H, W)
        original_shape = (int(math.sqrt(inputs.shape[1])), int(math.sqrt(inputs.shape[1])))

        downsample_x = nlc_to_nchw(inputs, hw_shape=original_shape)
        downsample_x = F.interpolate(downsample_x, size=self.embed_dims, mode='bilinear', align_corners=False)
        if self.conv_embedding_dim > 0:
            downsample_x = self.conv_embedding_pre(downsample_x)
        downsample_x = nchw_to_nlc(downsample_x)
        downsample_x = einops.rearrange(downsample_x, 'b n c -> b c n')
        B, C, N = downsample_x.shape

        if self.global_token_pos != 'none':
            global_token = torch.mean(downsample_x, dim=1, keepdim=True)
        else:
            global_token = None

        if self.reduction_ratio > 1:
            router_logits = self.channel_gate(downsample_x).squeeze(-1)  # (B, C)
            original_routing_weights = F.softmax(router_logits, dim=1)
            if self.sampling_scale is not None and self.sampling_scale['val'] > 0 and self.training:
                if self.sampling_scale['type'] == 'fixed':
                    sampling_scale = self.sampling_scale['val']
                else:
                    # get the current epoch and max epochs
                    message_hub = MessageHub.get_current_instance()
                    current_epoch = message_hub.get_info('epoch')
                    max_epochs = float(message_hub.get_info('max_epochs'))
                    # we decrease the sampling scale linearly between 0.00001 to self.sampling_scale
                    sampling_scale = (self.sampling_scale['val'] - 0.001) * (1 - current_epoch / max_epochs) + 0.001
                gumbel = torch.distributions.gumbel.Gumbel(0, sampling_scale).rsample
                tmp_router_logits = router_logits.detach() + gumbel(router_logits.shape).to(router_logits.device)
                tmp_routing_weights = F.softmax(tmp_router_logits, dim=1)
            else:
                tmp_routing_weights = original_routing_weights.detach()
            # sr_ratios=[8, 4, 2, 1],
            top_k = C // self.reduction_ratio

            _, selected_token = torch.topk(tmp_routing_weights, top_k, dim=-1)
            routing_weights = original_routing_weights[torch.arange(B)[:, None], selected_token]

            if self.loading_loss:
                top_k_mask = F.one_hot(selected_token, num_classes=C)

            if self.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            current_state = downsample_x[torch.arange(B)[:, None], selected_token]

            current_state = insert_global_token(current_state, self.global_token_pos, global_token)
            current_state = self.channel_mamba_mixer(current_state)  # (B, top_k, n)
            current_state = split_global_token(current_state, self.global_token_pos)
            current_state = current_state * routing_weights[:, :, None]  # (B, top_k, n)

            if self.is_softmax_on_x:
                residual_x = downsample_x * original_routing_weights[:, :, None]
                residual_x[torch.arange(B)[:, None], selected_token] = current_state
                new_downsample_x = downsample_x + residual_x
            else:
                new_downsample_x = downsample_x.clone()
                new_downsample_x[torch.arange(B)[:, None], selected_token] = downsample_x[torch.arange(B)[:, None], selected_token] + current_state

        else:
            downsample_x = insert_global_token(downsample_x, self.global_token_pos, global_token)
            # already has the residual connection
            downsample_x = self.channel_mamba_mixer(downsample_x)
            new_downsample_x = split_global_token(downsample_x, self.global_token_pos)

        router_logits = None
        top_k_mask = None

        new_downsample_x = einops.rearrange(new_downsample_x, 'b c n -> b n c')
        # upsample the spatial dimension
        new_downsample_x = nlc_to_nchw(new_downsample_x, hw_shape=self.embed_dims)
        new_downsample_x = F.interpolate(new_downsample_x, size=original_shape, mode='bilinear', align_corners=False)
        if self.conv_embedding_dim > 0:
            new_downsample_x = self.conv_embedding_post(new_downsample_x)
        new_downsample_x = nchw_to_nlc(new_downsample_x)
        if only_downsample:
            inputs = new_downsample_x
        else:
            inputs = (new_downsample_x + inputs) / 2
        return inputs, (router_logits, top_k_mask)


# 未整理，Not Reorganized
@MODELS.register_module()
class SCSparseMixer(BaseModule):
    def __init__(
            self,
            branch_format,  # spatial, channel, spatial-channel-serial, spatial-channel-parallel
            spatial_embed_dims,
            channel_embed_dims,
            spatial_reduction_ratio,
            channel_reduction_ratio,
            norm_topk_prob,
            path_type,
            norm_cfg,
            layer_cfgs,
            layer_idx,
            sampling_scale=dict(),
            global_token_pos='none',
            is_softmax_on_x=False,
            loading_loss=False,
    ):
        super(SCSparseMixer, self).__init__()
        self.branch_format = branch_format
        assert self.branch_format in ['spatial', 'channel', 'spatial-channel-serial', 'spatial-channel-parallel']
        if 'spatial' in branch_format:
            self.spatial_mixer = SpatialSparseMixer(
                embed_dims=spatial_embed_dims,
                reduction_ratio=spatial_reduction_ratio,
                norm_topk_prob=norm_topk_prob,
                path_type=path_type,
                norm_cfg=norm_cfg,
                layer_cfgs=layer_cfgs,
                layer_idx=layer_idx,
                sampling_scale=sampling_scale,
                global_token_pos=global_token_pos,
                is_softmax_on_x=is_softmax_on_x,
                loading_loss=loading_loss
            )

        if 'channel' in branch_format:
            self.channel_mixer = ChannelSparseMixer(
                embed_dims=channel_embed_dims,
                reduction_ratio=channel_reduction_ratio,
                norm_topk_prob=norm_topk_prob,
                path_type=path_type,
                norm_cfg=norm_cfg,
                layer_cfgs=layer_cfgs,
                layer_idx=layer_idx,
                sampling_scale=sampling_scale,
                global_token_pos=global_token_pos,
                is_softmax_on_x=is_softmax_on_x,
                loading_loss=loading_loss,
                conv_embedding_dim=spatial_embed_dims
            )

    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        # import ipdb; ipdb.set_trace()
        if 'spatial' == self.branch_format:
            x, (spatial_router_logits, s_top_k_mask) = self.spatial_mixer(x)
            channel_router_logits = None
            c_top_k_mask = None
        elif 'channel' == self.branch_format:
            x, (channel_router_logits, c_top_k_mask) = self.channel_mixer(x)
            spatial_router_logits = None
            s_top_k_mask = None
        elif 'spatial-channel-serial' == self.branch_format:
            # torch.autograd.set_detect_anomaly(True)
            x, (spatial_router_logits, s_top_k_mask) = self.spatial_mixer(x)
            x, (channel_router_logits, c_top_k_mask) = self.channel_mixer(x)
        elif 'spatial-channel-parallel' == self.branch_format:
            x_s, (spatial_router_logits, s_top_k_mask) = self.spatial_mixer(x)
            x_c, (channel_router_logits, c_top_k_mask) = self.channel_mixer(x)
            x = (x_s + x_c) / 2
        return x,  (spatial_router_logits, s_top_k_mask, channel_router_logits, c_top_k_mask)



@CDMODELS.register_module()
class DualInputsFPN(MMDET_FPN):
    def forward(self, x0: Tuple[Tensor], x1: Tuple[Tensor]) -> (Tuple[Tensor], Tuple[Tensor]):
        bs = x0[0].shape[0]
        x = [torch.cat([x0[i], x1[i]], dim=0) for i in range(len(x0))]
        x = super().forward(x)
        x0 = [i[:bs] for i in x]
        x1 = [i[bs:] for i in x]
        return x0, x1

@CDMODELS.register_module()
class DualInputsSimpleFusionNeck(BaseModule):
    def __init__(self, in_channels, return_tuple=False, init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.return_tuple = return_tuple
        for i, c in enumerate(in_channels):
            layer = nn.Sequential(
                nn.Conv2d(c * 2, c, 1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            )
            self.add_module(f'layer{i}', layer)

    def forward(self, x1, x2):
        x = [torch.cat([x1[i], x2[i]], dim=1) for i in range(len(x1))]
        x = [getattr(self, f'layer{i}')(item) for i, item in enumerate(x)]
        if self.return_tuple:
            return (x, )
        return x


@MMSEG_MODELS.register_module()
class UNetDecodeNeck(BaseModule):
    def __init__(self,
                in_channels=[256, 256, 256, 256, 256],
                dec_num_convs=(2, 2, 2, 2),
                dec_dilations=(1, 1, 1, 1),
                with_cp=False,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                upsample_cfg=dict(type='InterpConv'),
                norm_eval=False,
                init_cfg=None):
        super().__init__(init_cfg)
        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(
                    type='Constant',
                    val=1,
                    layer=['_BatchNorm', 'GroupNorm'])
            ]

        self.norm_eval = norm_eval

        self.in_channels = in_channels
        self.decoder = nn.ModuleList()

        for i in range(len(self.in_channels) - 1):
            self.decoder.append(
                UpConvBlock(
                    conv_block=BasicConvBlock,
                    in_channels=self.in_channels[i + 1],
                    skip_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    num_convs=dec_num_convs[i],
                    stride=1,
                    dilation=dec_dilations[i],
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    upsample_cfg=upsample_cfg,
                    dcn=None,
                    plugins=None))

    def forward(self, enc_outs):
        #  torch.Size([32, 256, 128, 128]), torch.Size([32, 256, 64, 64]), torch.Size([32, 256, 32, 32]), torch.Size([32, 256, 16, 16]), torch.Size([32, 256, 8, 8])
        dec_outs = [enc_outs[-1]]
        x = enc_outs[-1]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)
        # torch.Size([32, 256, 8, 8]), torch.Size([32, 256, 16, 16]), torch.Size([32, 256, 32, 32]), torch.Size([32, 256, 64, 64]), torch.Size([32, 256, 128, 128])
        return dec_outs

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()