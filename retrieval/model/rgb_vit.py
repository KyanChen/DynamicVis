
import torch
import torchvision
from torch import nn
from .model_registry import register_model


@register_model
class RGB_ViT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = torchvision.models.vision_transformer.vit_b_16(
            weights=torchvision.models.vision_transformer.ViT_B_16_Weights)

    def forward(self, input):
        # squeeze time dimension
        input = input.squeeze(2)

        # Run forward pass
        x = self.model._process_input(input)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x
