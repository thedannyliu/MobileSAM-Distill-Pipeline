import torch
import torch.nn as nn
import torch.nn.functional as F

from mobile_sam import sam_model_registry
from mobile_sam.modeling import MaskDecoder, PromptEncoder, Sam, TinyViT, TwoWayTransformer

try:
    from timm.layers import SqueezeExcite
except ImportError:  # pragma: no cover - older timm compatibility
    from timm.models.layers import SqueezeExcite


MOBILE_SAM_IMAGE_SIZE = 1024
MOBILE_SAM_EMBED_SIZE = 64
MOBILE_SAM_EMBED_DIM = 256


def torch_load(path, map_location="cpu", weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dBN(nn.Sequential):
    def __init__(self, in_chans, out_chans, kernel_size=1, stride=1, padding=0, groups=1, bn_weight_init=1):
        super().__init__(
            nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_chans),
        )
        nn.init.constant_(self[1].weight, bn_weight_init)
        nn.init.constant_(self[1].bias, 0)


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class RepVGGDW(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv2dBN(channels, channels, 3, 1, 1, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv(x) + self.conv1(x) + x)


class RepViTBlock(nn.Module):
    def __init__(self, in_chans, hidden_dim, out_chans, kernel_size, stride, use_se, use_hs):
        super().__init__()
        if hidden_dim != 2 * in_chans:
            raise ValueError("RepViT m0.9 expects hidden_dim == 2 * in_chans")
        activation = nn.GELU()
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2dBN(in_chans, in_chans, kernel_size, stride, (kernel_size - 1) // 2, groups=in_chans),
                SqueezeExcite(in_chans, 0.25) if use_se else nn.Identity(),
                Conv2dBN(in_chans, out_chans),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2dBN(out_chans, 2 * out_chans),
                    activation,
                    Conv2dBN(2 * out_chans, out_chans, bn_weight_init=0),
                )
            )
        elif stride == 1 and in_chans == out_chans:
            self.token_mixer = nn.Sequential(
                RepVGGDW(in_chans),
                SqueezeExcite(in_chans, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2dBN(in_chans, hidden_dim),
                    activation,
                    Conv2dBN(hidden_dim, out_chans, bn_weight_init=0),
                )
            )
        else:
            raise ValueError("RepViT stride must be 1 with same channels or 2")

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViTBackbone(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        input_channel = cfgs[0][2]
        layers = [
            nn.Sequential(
                Conv2dBN(3, input_channel // 2, 3, 2, 1),
                nn.GELU(),
                Conv2dBN(input_channel // 2, input_channel, 3, 2, 1),
            )
        ]
        for kernel_size, expansion, channels, use_se, use_hs, stride in cfgs:
            output_channel = _make_divisible(channels, 8)
            hidden_dim = _make_divisible(input_channel * expansion, 8)
            layers.append(RepViTBlock(input_channel, hidden_dim, output_channel, kernel_size, stride, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        self.out_channels = input_channel

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x


def repvit_m0_9_backbone():
    cfgs = [
        [3, 2, 48, 1, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 96, 0, 0, 2],
        [3, 2, 96, 1, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 192, 0, 1, 2],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 384, 0, 1, 2],
        [3, 2, 384, 1, 1, 1],
        [3, 2, 384, 0, 1, 1],
    ]
    return RepViTBackbone(cfgs)


class RepViTMobileImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_size = MOBILE_SAM_IMAGE_SIZE
        self.backbone = repvit_m0_9_backbone()
        self.head = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels, MOBILE_SAM_EMBED_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(MOBILE_SAM_EMBED_DIM),
            nn.GELU(),
            nn.Conv2d(MOBILE_SAM_EMBED_DIM, MOBILE_SAM_EMBED_DIM, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.head(self.backbone(x))
        if x.shape[-2:] != (MOBILE_SAM_EMBED_SIZE, MOBILE_SAM_EMBED_SIZE):
            x = F.interpolate(x, size=(MOBILE_SAM_EMBED_SIZE, MOBILE_SAM_EMBED_SIZE), mode="bilinear", align_corners=False)
        return x


def build_tinyvit_image_encoder():
    return TinyViT(
        img_size=MOBILE_SAM_IMAGE_SIZE,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )


def build_student_image_encoder(student_arch):
    if student_arch == "tinyvit":
        return build_tinyvit_image_encoder()
    if student_arch == "repvit_m0_9":
        return RepViTMobileImageEncoder()
    raise ValueError(f"Unsupported student_arch: {student_arch}")


def build_mobilesam_shell(student_arch, mobile_sam_ckpt=None):
    if student_arch == "tinyvit":
        return sam_model_registry["vit_t"](checkpoint=mobile_sam_ckpt)

    prompt_embed_dim = MOBILE_SAM_EMBED_DIM
    image_embedding_size = MOBILE_SAM_EMBED_SIZE
    model = Sam(
        image_encoder=build_student_image_encoder(student_arch),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(MOBILE_SAM_IMAGE_SIZE, MOBILE_SAM_IMAGE_SIZE),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if mobile_sam_ckpt is not None:
        state_dict = torch_load(mobile_sam_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def unwrap_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "image_encoder"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_image_encoder_checkpoint(model, checkpoint_path, strict=True):
    checkpoint = torch_load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = unwrap_checkpoint_state(checkpoint)
    return model.image_encoder.load_state_dict(state_dict, strict=strict)
