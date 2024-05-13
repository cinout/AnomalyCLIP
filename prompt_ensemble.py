import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict

# from open_clip import tokenizer
# simple_tokenizer = tokenizer.SimpleTokenizer()
from copy import deepcopy
import torch.nn as nn

_tokenizer = _Tokenizer()


def tokenize(
    texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False
) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result


# never used
def encode_text_with_prompt_ensemble(model, texts, device):
    prompt_normal = [
        "{}",
        "flawless {}",
        "perfect {}",
        "unblemished {}",
        "{} without flaw",
        "{} without defect",
        "{} without damage",
    ]
    prompt_abnormal = [
        "damaged {}",
        "broken {}",
        "{} with flaw",
        "{} with defect",
        "{} with damage",
    ]
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = [
        "a bad photo of a {}.",
        "a low resolution photo of the {}.",
        "a bad photo of the {}.",
        "a cropped photo of the {}.",
        "a bright photo of a {}.",
        "a dark photo of the {}.",
        "a photo of my {}.",
        "a photo of the cool {}.",
        "a close-up photo of a {}.",
        "a black and white photo of the {}.",
        "a bright photo of the {}.",
        "a cropped photo of a {}.",
        "a jpeg corrupted photo of a {}.",
        "a blurry photo of the {}.",
        "a photo of the {}.",
        "a good photo of the {}.",
        "a photo of one {}.",
        "a close-up photo of the {}.",
        "a photo of a {}.",
        "a low resolution photo of a {}.",
        "a photo of a large {}.",
        "a blurry photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a good photo of a {}.",
        "a photo of the small {}.",
        "a photo of the large {}.",
        "a black and white photo of a {}.",
        "a dark photo of a {}.",
        "a photo of a cool {}.",
        "a photo of a small {}.",
        "there is a {} in the scene.",
        "there is the {} in the scene.",
        "this is a {} in the scene.",
        "this is the {} in the scene.",
        "this is one {} in the scene.",
    ]

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(texts[0]) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence)
        class_embeddings = model.encode_text(prompted_sentence.to(device))
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device).t()

    return text_features


def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])


class VisualAE(nn.Module):
    def __init__(self, in_dim=768):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

        self.enc_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=2, stride=2
        )
        self.enc_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim * 2, kernel_size=2, stride=1
        )
        self.dec_conv1 = nn.ConvTranspose2d(
            in_channels=in_dim * 2, out_channels=in_dim, kernel_size=2, stride=1
        )
        self.dec_conv2 = nn.ConvTranspose2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=2, stride=2
        )

    def forward(self, x):
        # encode
        x = self.enc_conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.enc_conv2(x)
        x = self.relu(x)  # [8, 1536, 17, 17]

        # decode
        x = self.dec_conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dec_conv2(x)  # [8, 768, 36, 36]

        return x


class AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details, args):
        super().__init__()

        """
        configs
        """
        self.meta_net = args.meta_net
        self.maple = args.maple
        self.metanet_patch_and_global = args.metanet_patch_and_global
        self.metanet_patch_only = args.metanet_patch_only
        self.debug_mode = args.debug_mode
        self.visual_ae = args.visual_ae
        self.features_list = args.features_list
        vis_dim = clip_model.visual.output_dim  # 768

        """
        visual AE
        """
        if self.visual_ae:
            autoencoder = VisualAE(in_dim=vis_dim)
            self.aes = _get_clones(autoencoder, len(self.features_list))  # len == 4

        """
        Unused
        """
        ctx_init_pos = ""
        ctx_init_neg = ""

        """
        Initialize
        """
        classnames = ["object"]
        self.n_cls = len(classnames)  # 1

        self.n_ctx = design_details["Prompt_length"]  # 12
        n_ctx_pos = self.n_ctx  # 12
        n_ctx_neg = self.n_ctx  # 12

        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]  # 4

        dtype = clip_model.transformer.get_cast_dtype()
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 768
        self.ctx_dim = ctx_dim

        """
        meta_net
        """

        if self.meta_net:

            self.meta_net = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                        ("relu", nn.ReLU(inplace=True)),
                        (
                            "linear2",
                            nn.Linear(
                                vis_dim // 16,
                                ctx_dim,
                            ),
                        ),
                    ]
                )
            )

        """
        normal/abnormal templates
        """
        self.state_normal_list = [
            "{}",
        ]
        self.state_anomaly_list = [
            "damaged {}",
        ]
        normal_num = len(self.state_normal_list)  # 1
        anormaly_num = len(self.state_anomaly_list)  # 1
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num

        if ctx_init_pos and ctx_init_neg:
            """
            NOT VISTED
            """

            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            # 初始化text成bpd编码
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                # 生成相应的text embedding
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            # 这些是去除出来EOS 和 # CLS, EOS， 获得可学习的textual prompt
            ctx_vectors_pos = embedding_pos[0, 1 : 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1 : 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if True:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)
        else:
            # Arrive here
            """
            create learnable word embeddings
            """
            if True:
                print("Initializing class-specific contexts")
                # 这里是cls是类的个数，n_ctx_pos代表learnable token的长度，ctx_dim表示prompt的dimension
                ctx_vectors_pos = torch.empty(
                    self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype
                )  # (1, 1, 12, 768)
                ctx_vectors_neg = torch.empty(
                    self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype
                )  # (1, 1, 12, 768)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)  # X X X X X X X X X X X X
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)

        """
        initialize learnable token embeddings
        """
        self.compound_prompts_depth = design_details[
            "learnabel_text_embedding_depth"
        ]  # 9
        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )  # learnable token embeddings, [4, 768] * 8

        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        """
        Initialize coupling function for MAPLE
        """
        if self.maple:
            single_layer = nn.Linear(ctx_dim, 1024)
            self.compound_prompt_projections = _get_clones(
                single_layer, self.compound_prompts_depth - 1
            )  # len ==8

        """
        initialize learnable word embeddings
        """
        self.ctx_pos = nn.Parameter(
            ctx_vectors_pos
        )  # to be optimized # (1, 1, 12, 768)
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        prompts_pos = [
            prompt_prefix_pos + " " + template.format(name) + "."
            for template in self.state_normal_list
            for name in classnames
        ]  # ['X X X X X X X X X X X X object.']
        prompts_neg = [
            prompt_prefix_neg + " " + template.format(name) + "."
            for template in self.state_anomaly_list
            for name in classnames
        ]  # ["X X X X X X X X X X X X damaged object."]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []

        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))

        tokenized_prompts_pos = torch.cat(
            tokenized_prompts_pos
        )  # [1, 77]; the [0] is 49406; then [1:13] is 324 (X); then [13:16] is 14115 (object), 269 (.), 49407 (EOT); the rest are 0
        tokenized_prompts_neg = torch.cat(
            tokenized_prompts_neg
        )  # [1, 77]; the [0] is 49406; then [1:13] is 324 (X); then [13:17] is 13568 (damaged), 14115 (object), 269 (.), 49407 (EOT); the rest are 0

        # 生成相应的text embedding
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(
                dtype
            )
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(
                dtype
            )
            n, l, d = embedding_pos.shape  # [1, 77, 768]
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(
                1, 0, 2, 3
            )  # [1, 1, 77, 768]
            embedding_neg = embedding_neg.reshape(
                anormaly_num, self.n_cls, l, d
            ).permute(
                1, 0, 2, 3
            )  # [1, 1, 77, 768]

        """
        take the prefix and suffix from the pos/neg word embeddings
        """
        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer(
            "token_suffix_pos", embedding_pos[:, :, 1 + n_ctx_pos :, :]
        )
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer(
            "token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg :, :]
        )

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(
            normal_num, self.n_cls, d
        ).permute(
            1, 0, 2
        )  # [1, 1, 77]

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(
            anormaly_num, self.n_cls, d
        ).permute(
            1, 0, 2
        )  # [1, 1, 77]

        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)

        # self.proj = nn.Linear(ctx_dim, 768)
        # self.proj.half()
        self.visual_deep_prompts = None
        if self.maple:
            visual_deep_prompts = []
            for index, layer in enumerate(self.compound_prompt_projections):
                visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
            self.visual_deep_prompts = visual_deep_prompts  # [4, 1024] * 8

    def process_patch_features(self, patch_feature, idx):
        # patch_feature: [bs, 1370, 768]
        bs, n, c = patch_feature.shape

        # get global token
        global_token = patch_feature[:, 0, :]

        # reshape feature map
        patch_feature = patch_feature[:, 1:, :]
        side = int((n - 1) ** 0.5)
        patch_feature = patch_feature.reshape(bs, side, side, -1).permute(
            0, 3, 1, 2
        )  # [bs, c, side, side]

        # process by AE
        ae = self.aes[idx]
        patch_feature = ae(patch_feature)

        # reshape back
        patch_feature = patch_feature.reshape(bs, c, -1).permute(
            0, 2, 1
        )  # [bs, n-1, c]
        patch_feature = torch.cat([global_token.unsqueeze(1), patch_feature], dim=1)

        return patch_feature

    def forward(self, image_features=None, patch_features=None, cls_id=None):
        # image_features.shape: [bs, 768]
        # patch_features: 4*[bs, 1370, 768]

        ctx_pos = self.ctx_pos  # (1, 1, 12, 768)
        ctx_neg = self.ctx_neg

        prefix_pos = self.token_prefix_pos  # [1, 1, 1, 768]
        prefix_neg = self.token_prefix_neg  # [1, 1, 1, 768]
        suffix_pos = self.token_suffix_pos  # [1, 1, 64, 768]
        suffix_neg = self.token_suffix_neg  # [1, 1, 64, 768]

        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, d)  # [1, 77]
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, d)  # [1, 77]

        if (
            self.meta_net
            and not self.debug_mode
            and image_features is None
            and patch_features is None
        ):
            raise Exception("Something is not right!")

        if self.meta_net and image_features is not None and patch_features is not None:
            if self.metanet_patch_only:
                patch_features = [
                    torch.mean(feature[:, 1:, :], dim=1) for feature in patch_features
                ]  # 4*[bs, 768]
                patch_features = torch.stack(patch_features, dim=1)  # [bs, 4, 768]
                patch_features = torch.mean(patch_features, dim=1)  # [bs, 768]
                bias = self.meta_net(patch_features)
            elif self.metanet_patch_and_global:
                patch_features = [
                    torch.mean(feature[:, 1:, :], dim=1) for feature in patch_features
                ]  # 4*[bs, 768]
                patch_features = torch.stack(patch_features, dim=1)  # [bs, 4, 768]
                patch_features = torch.mean(patch_features, dim=1)  # [bs, 768]
                bias = self.meta_net(image_features + patch_features)
            else:
                bias = self.meta_net(
                    image_features
                )  # [bs, ctx_dim or ctx_dim*2], ctx_dim=768

            bs, _ = bias.shape

            ctx_pos = ctx_pos.unsqueeze(0)  # (1, 1, 1, 12, 768)
            ctx_neg = ctx_neg.unsqueeze(0)

            bias = bias.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, 1, 768)

            ctx_pos = ctx_pos + bias  # (bs, 1, 1, 12, 768)
            ctx_neg = ctx_neg + bias  # (bs, 1, 1, 12, 768)

            prefix_shape = prefix_pos.shape
            suffix_shape = suffix_pos.shape

            prompts_pos = torch.cat(
                [
                    prefix_pos.unsqueeze(0).expand((bs, *prefix_shape)),
                    ctx_pos,
                    suffix_pos.unsqueeze(0).expand((bs, *suffix_shape)),
                ],
                dim=3,
            )  # [bs, 1, 1, 77, 768]

            prompts_neg = torch.cat(
                [
                    prefix_neg.unsqueeze(0).expand((bs, *prefix_shape)),
                    ctx_neg,
                    suffix_neg.unsqueeze(0).expand((bs, *suffix_shape)),
                ],
                dim=3,
            )  # [bs, 1, 1, 77, 768]

            _, _, _, l, d = prompts_pos.shape
            prompts_pos = prompts_pos.reshape(-1, l, d)  # [bs, 77, 768]
            _, _, _, l, d = prompts_neg.shape
            prompts_neg = prompts_neg.reshape(-1, l, d)

            tokenized_prompts = torch.cat(
                (
                    tokenized_prompts_pos.expand((bs, -1)),
                    tokenized_prompts_neg.expand((bs, -1)),
                ),
                dim=0,
            )  # [2 or 2*bs, 77]

        else:
            prompts_pos = torch.cat(
                [
                    # N(the number of template), 1, dim
                    prefix_pos,  # (n_cls, 1, dim)
                    ctx_pos,  # (n_cls, n_ctx, dim)
                    suffix_pos,  # (n_cls, *, dim)
                ],
                dim=2,
            )  # [1, 1, 77, 768]

            prompts_neg = torch.cat(
                [
                    prefix_neg,  # (n_cls, 1, dim)
                    ctx_neg,  # (n_cls, n_ctx, dim)
                    suffix_neg,  # (n_cls, *, dim)
                ],
                dim=2,
            )  # [1, 1, 77, 768]

            _, _, l, d = prompts_pos.shape
            prompts_pos = prompts_pos.reshape(-1, l, d)
            _, _, l, d = prompts_neg.shape
            prompts_neg = prompts_neg.reshape(-1, l, d)

            tokenized_prompts = torch.cat(
                (tokenized_prompts_pos, tokenized_prompts_neg), dim=0
            )  # [2, 77]

        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)  # [2 or 2*bs, 77, 768]

        """
        prompts: [2 or 2*bs, 77, 768], 2 is pos&neg, embeddings with 12 Xs replaced by learnables already

        tokenized_prompts: [2 or 2*bs, 77], prompts are tokenized, but NOT clip_model.token_embedding, with 12 Xs included. Used for finding EOT. No learnable parameters.

        learnable token/text embeddings: [4, 768] * 8
        """

        return prompts, tokenized_prompts, self.compound_prompts_text
