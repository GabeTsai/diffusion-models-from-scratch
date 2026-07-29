"""Network architectures shared by the diffusion notebooks."""

import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet


class ConditionedUnet(Unet):
    """
    Base Unet with learned class embedding added to timestep embedding for classifier-free guidance.
    """
    def __init__(self, num_classes, dim, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(num_classes + 1, 4 * dim)

    def forward(self, x, time, y):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time) + self.class_emb(y)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
