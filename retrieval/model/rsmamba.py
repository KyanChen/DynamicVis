import torch
import torchvision
from torch import nn
from mmpretrain.utils import register_all_modules
register_all_modules()
import dynamicvis
from mmpretrain.registry import MODELS
from .model_registry import register_model


@register_model
class DynamicVisBackbone(nn.Module):
    def __init__(self, arch='b', img_size=512, checkpoint=None, *args, **kwargs):
        super().__init__()

        backbone = dict(
            type='mmpretrain.DynamicVisBackbone',
            arch=arch,
            # frozen_stages=1,
            path_type='forward_reverse_mean',
            sampling_scale=dict(type='decay', val=0.1),
            global_token_cfg=dict(pos='head', num=-1),
            is_softmax_on_x=True,
            img_size=img_size,
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            spatial_token_keep_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3,),
            out_type='featmap',
            init_cfg=dict(
                type='Pretrained',
                checkpoint=checkpoint,
                prefix='backbone.'),
        ),

        model = MODELS.build(backbone)
        model.init_weights()
        self.model = model

    def forward(self, input):
        # import ipdb; ipdb.set_trace()
        # squeeze time dimension
        input = input.squeeze(2)

        x = self.model(input)

        x = torch.mean(x[-1], dim=[2, 3])
        return x
