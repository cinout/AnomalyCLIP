from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


# For VV self-attention!!!!!
class Attention(nn.Module):
    def __init__(
        self,
        out_dim,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        settings="",
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        # replace k & q by v
        k = v
        q = k

        # self-attention, higher temperate for resnets performs better
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        design_details=None,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, whole=False, ffn=False):
        # dual paths for blocks deeper than "d"

        if isinstance(self.attn, Attention):
            if isinstance(x, list):
                if not ffn:
                    x, x_ori = x
                    x_res = self.attention(self.ln_1(x_ori))
                    x_res, x_ori_res = x_res
                    x_ori += x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res  # skip ffn for the new path
                    return [x, x_ori]
                else:
                    x, x_ori_1 = x
                    x_res = self.attention(self.ln_1(x_ori_1))
                    x_res, x_ori_res = x_res
                    x_ori = x_ori_1 + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res  # skip ffn for the new path
                    x = x_res + x_ori_1
                    x = x + self.mlp(self.ln_2(x))
                    return [x, x_ori]
            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]

        # singl path before "d"
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_learnable_token(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        design_details=None,
        text_layer=False,
        i=0,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.i = i
        self.compound_prompt_nctx = design_details[
            "learnabel_text_embedding_length"
        ]  # 4
        self.text_layer = text_layer
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):

        # dual paths for blocks deeper than "d"
        # NOT ARRIVED
        if isinstance(self.attn, Attention):
            x = inputs[0]
            if isinstance(x, list):
                x, x_ori = x
                x_res = self.attention(self.ln_1(x_ori))
                x_res, x_ori_res = x_res
                x_ori += x_ori_res
                x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                x += x_res  # skip ffn for the new path
                return [x, x_ori]

            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]

        # singl path before "d"
        else:
            # ARRIVED HERE
            # this is where text transformer works
            x = inputs[0]  # [77, 2, 768]

            compound_prompts_deeper = inputs[1]
            counter = inputs[2]  # 0,0,1,2...,8,8,8

            if not self.first_layer:
                # First check if the ith layer needs compound prompts or not
                if not (counter > len(compound_prompts_deeper) - 1):  # if counter <= 7
                    # Appending the learnable tokens in different way

                    # First remove the learnable tokens (middle part) from previous layer
                    prefix = x[:1, :, :]
                    suffix = x[1 + self.compound_prompt_nctx :, :, :]

                    # then create the new learnable tokens
                    textual_context = compound_prompts_deeper[counter]
                    textual_context = (
                        textual_context.expand(x.shape[1], -1, -1)
                        .permute(1, 0, 2)
                        .half()
                    )

                    # Add the learnable tokens of this layer to the input
                    x = torch.cat([prefix, textual_context, suffix], dim=0)

                    # Once done, update the counter, so that the next time, it does not use same learnable tokens
                    counter += 1
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return [x, compound_prompts_deeper, counter]


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,  # 24
        heads: int,
        attn_mask: torch.Tensor = None,
        need_weights: bool = False,
        design_details=None,
        text_layer=False,  # False if from VisionTransformer, True if just text Transformer
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        self.design_deatails = design_details

        if self.text_layer and (design_details is not None):
            self.resblocks = nn.ModuleList(
                [
                    ResidualAttentionBlock_learnable_token(
                        width, heads, attn_mask, design_details, text_layer, i=i
                    )
                    for i in range(layers)
                ]
            )
        else:
            # the resblocks for VisionTransformer, whose attn are replaced with VV later
            self.resblocks = nn.ModuleList(
                [
                    ResidualAttentionBlock(
                        width,
                        heads,
                        attn_mask,
                    )
                    for i in range(layers)
                ]
            )

    def ori_CLIP_with_patch_forward(self, x, out_layers):
        idx = 0
        out_tokens = []
        for r in self.resblocks:
            idx += 1
            x = r(x)
            if idx in out_layers:
                if isinstance(x, list):
                    out_tokens.append(x[1])
                else:
                    out_tokens.append(x)

        return [x, x], out_tokens

    def AnomalyCLIP_forward(self, x, out_layers, ffn):
        idx = 0
        out_tokens = []
        for r in self.resblocks:
            idx += 1
            x = r(x, ffn=ffn)
            if idx in out_layers:
                if isinstance(x, list):
                    out_tokens.append(x[0])  # difference
                else:
                    out_tokens.append(x)
        return x, out_tokens  # out_tokens: len==4, each size [1370, 8, 1024]

    def forward(
        self, x: torch.Tensor, out_layers=[6, 12, 18, 24], DPAM_layer=None, ffn=False
    ):
        # visual encoder forward
        if not self.text_layer:
            out_tokens = []

            if DPAM_layer is None:
                [x, x], out_tokens = self.ori_CLIP_with_patch_forward(x, out_layers)
                return [x, x], out_tokens
            else:
                x, out_tokens = self.AnomalyCLIP_forward(x, out_layers, ffn)
                return x, out_tokens

        # text encoder forward
        # ori text embedding
        elif self.design_deatails is None:
            for idx, r in enumerate(self.resblocks):
                x = r(x)
            return x

        # insert learnable text embedding
        elif self.design_deatails is not None:
            for idx, r in enumerate(self.resblocks):
                x = r(x)
            return x[0]

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,  # 336
        patch_size: int,  # 14
        width: int,  # 1024
        layers: int,  # 24
        heads: int,  # 16
        output_dim: int,  # 768
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # [1024]
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )  # [577, 1024]
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, need_weights=True)
        self.attn = None
        self.embed_dim = width
        self.num_heads = heads

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))  # [1024, 768]

    @torch.no_grad()
    def DAPM_replace(self, DPAM_layer):
        if DPAM_layer is not None:
            for i in range(
                1, DPAM_layer
            ):  # i [1, 19], inclusive <=> layer [6, 24] inclusive
                # create v-v attention, which returns [v-v, original]
                self.attn = Attention(
                    self.embed_dim, self.embed_dim, self.num_heads, True
                )
                self.attn.qkv.weight.data = self.transformer.resblocks[
                    -i
                ].attn.in_proj_weight.clone()
                self.attn.qkv.bias.data = self.transformer.resblocks[
                    -i
                ].attn.in_proj_bias.clone()
                self.attn.proj.weight.data = self.transformer.resblocks[
                    -i
                ].attn.out_proj.weight.clone()
                self.attn.proj.bias.data = self.transformer.resblocks[
                    -i
                ].attn.out_proj.bias.clone()
                self.transformer.resblocks[-i].attn = self.attn

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,  # [8, 3, 518, 518]
        features_list,
        ori_patch=False,
        proj_use=True,
        DPAM_layer=None,
        ffn=False,
    ):

        x = self.conv1(x)  # shape = [*, width, grid, grid], [8, 1024, 37, 37]
        x = x.reshape(
            x.shape[0], x.shape[1], -1
        )  # shape = [*, width, grid ** 2], [8, 1024, 1369]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width], [8, 1369, 1024]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width], [8, 1370, 1024]
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)  # 24
        new_side = int((x.shape[1] - 1) ** 0.5)  # 37

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = (
                self.positional_embedding[1:, :]
                .reshape(-1, side, side, x.shape[-1])
                .permute(0, 3, 1, 2)
            )
            new_pos = torch.nn.functional.interpolate(
                new_pos, (new_side, new_side), mode="bilinear"
            )
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(
                1, 2
            )
            self.positional_embedding.data = torch.cat(
                [self.positional_embedding[:1, :], new_pos[0]], 0
            )

        pos = self.positional_embedding.to(x.dtype)
        x = x + pos  # [8, 1370, 1024]
        x = self.ln_pre(x)  # [8, 1370, 1024]

        x = x.permute(1, 0, 2)  # NLD -> LND, [1370, 8, 1024]

        [x, x_ori], patch_tokens = self.transformer(
            x, features_list, DPAM_layer=DPAM_layer, ffn=ffn
        )  # x.shape: [1370, 8, 1024]; x_ori.shape: [1370, 8, 1024]
        # len(patch_tokens): 4; patch_tokens[0].shape: torch.Size([1370, 8, 1024])

        patch_token_list = []
        for patch_token in patch_tokens:
            patch_token = (
                self.ln_post(patch_token.permute(1, 0, 2)) @ self.proj
            )  # LND -> NLD
            patch_token_list.append(patch_token)
        patch_tokens = patch_token_list
        # len(patch_tokens: 4, patch_tokens[0].shape: torch.Size([8, 1370, 768])

        # x_ori[0] must be the GLOBAL embedding
        # return shape: [8, 768], list=4*[8, 1370, 768]
        return x_ori[0, :, :] @ self.proj, patch_tokens


class AnomalyCLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,  # 768
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,  # 1024
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,  # 768
        transformer_heads: int,
        transformer_layers: int,
        design_details=None,
    ):
        super().__init__()

        self.context_length = context_length  # 77

        if isinstance(vision_layers, (tuple, list)):
            # NO!!!
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            # arrive here
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )

        # what is this transformer for? (for processing text)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            text_layer=True,
            design_details=design_details,
        )

        self.vocab_size = vocab_size  # 49408
        self.token_embedding = nn.Embedding(
            vocab_size, transformer_width
        )  # a lookup table?

        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )  # [77, 768]

        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim)
        )  # [768, 768]

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # NOT USED

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(
        self,
        image,
        feature_list=[],
        ori_patch=False,
        proj_use=True,
        DPAM_layer=None,
        ffn=False,
    ):
        # return shape: global:[8, 768], local=4*[8, 1370, 768]
        return self.visual(
            image.type(self.dtype),
            feature_list,
            ori_patch=ori_patch,  # False
            proj_use=proj_use,  # True
            DPAM_layer=DPAM_layer,  # 20
            ffn=ffn,  # False
        )

    # NOT USED in the code
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text_learn(
        self,
        prompts,  # [2, 77, 768] embeddings with 12 Xs replaced already
        tokenized_prompts,  # [2, 77] prompts are tokenized, but not clip_model.token_embedding, with 12 Xs included
        deep_compound_prompts_text=None,  # [4, 768] * 8
        normalize: bool = False,
    ):
        cast_dtype = self.transformer.get_cast_dtype()

        x = prompts + self.positional_embedding.to(cast_dtype)  # [2, 77, 768]
        x = x.permute(1, 0, 2)  # NLD -> LND  [77, 2, 768]

        if deep_compound_prompts_text is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, deep_compound_prompts_text, 0])  # [77, 2, 768]

        x = x.permute(1, 0, 2)  # LND -> NLD, [2, 77, 768]
        x = self.ln_final(x).type(
            self.dtype
        )  # [batch_size, n_ctx, transformer.width], [2, 77, 768]

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # the argmax operation takes the representation at the EOT (end of text) position. There's nothing inherently wrong with taking the average along the sequence dimension, but taking representation at the position of a special token (e.g. the CLS token in ViT and BERT) is empirically known to work better. Other representations are still used since in each attention layer, the [EOT] token is attended to every other location. Argmax is used to locate the index (i_eot) of [EOT] at tokenized prompts. Once we locate it , we use the features of [EOT] by x[batchsize, i_eot] to represent the features of prompts
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )  # [2. 768]

        return x

    # NOT USED in the code
    def forward(self, image, text):
        print("ever jhere???")
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
