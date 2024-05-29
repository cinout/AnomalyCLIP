import torch
import math
import random
from tqdm import tqdm
from PIL import Image
import numpy as np

B, L, C = 4, 10, 5
tensor1 = torch.randint(0, 30, size=(B, L, C))
tensor2 = torch.tensor(
    [[8, 1, 3, 0, 5], [4, 1, 2, 3, 5], [7, 3, 6, 4, 8], [8, 7, 0, 2, 3]]
)

print(tensor1)

for img_idx, value in enumerate(tensor2):
    result = tensor1[img_idx, value]
    print(result)
exit()


H = 37
image_size = 518

path = "data/mvtec/cable/test/cut_inner_insulation/001.png"
image = Image.open(path).convert("RGBA")
image = image.resize((H, H))


values = random.sample(list(range(H * H - 1)), math.floor(H * H * 0.6))
values = [(math.floor(v / H), v % H) for v in values]


seg = np.full((H, H), False)
for tup in values:
    seg[tup] = True


mask = np.zeros(
    (H, H, 4),
    dtype=np.uint8,
)
mask[seg] = [
    238,
    79,
    38,
    89,
]
mask = Image.fromarray(mask)
image.paste(mask, (0, 0), mask)

image.save("test.png", "PNG")

AnomalyCLIP_keys = [
    "positional_embedding",
    "text_projection",
    "logit_scale",
    "visual.class_embedding",
    "visual.positional_embedding",
    "visual.proj",
    "visual.conv1.weight",
    "visual.ln_pre.weight",
    "visual.ln_pre.bias",
    "visual.transformer.resblocks.0.attn.in_proj_weight",
    "visual.transformer.resblocks.0.attn.in_proj_bias",
    "visual.transformer.resblocks.0.attn.out_proj.weight",
    "visual.transformer.resblocks.0.attn.out_proj.bias",
    "visual.transformer.resblocks.0.ln_1.weight",
    "visual.transformer.resblocks.0.ln_1.bias",
    "visual.transformer.resblocks.0.mlp.c_fc.weight",
    "visual.transformer.resblocks.0.mlp.c_fc.bias",
    "visual.transformer.resblocks.0.mlp.c_proj.weight",
    "visual.transformer.resblocks.0.mlp.c_proj.bias",
    "visual.transformer.resblocks.0.ln_2.weight",
    "visual.transformer.resblocks.0.ln_2.bias",
    "visual.transformer.resblocks.1.attn.in_proj_weight",
    "visual.transformer.resblocks.1.attn.in_proj_bias",
    "visual.transformer.resblocks.1.attn.out_proj.weight",
    "visual.transformer.resblocks.1.attn.out_proj.bias",
    "visual.transformer.resblocks.1.ln_1.weight",
    "visual.transformer.resblocks.1.ln_1.bias",
    "visual.transformer.resblocks.1.mlp.c_fc.weight",
    "visual.transformer.resblocks.1.mlp.c_fc.bias",
    "visual.transformer.resblocks.1.mlp.c_proj.weight",
    "visual.transformer.resblocks.1.mlp.c_proj.bias",
    "visual.transformer.resblocks.1.ln_2.weight",
    "visual.transformer.resblocks.1.ln_2.bias",
    "visual.transformer.resblocks.2.attn.in_proj_weight",
    "visual.transformer.resblocks.2.attn.in_proj_bias",
    "visual.transformer.resblocks.2.attn.out_proj.weight",
    "visual.transformer.resblocks.2.attn.out_proj.bias",
    "visual.transformer.resblocks.2.ln_1.weight",
    "visual.transformer.resblocks.2.ln_1.bias",
    "visual.transformer.resblocks.2.mlp.c_fc.weight",
    "visual.transformer.resblocks.2.mlp.c_fc.bias",
    "visual.transformer.resblocks.2.mlp.c_proj.weight",
    "visual.transformer.resblocks.2.mlp.c_proj.bias",
    "visual.transformer.resblocks.2.ln_2.weight",
    "visual.transformer.resblocks.2.ln_2.bias",
    "visual.transformer.resblocks.3.attn.in_proj_weight",
    "visual.transformer.resblocks.3.attn.in_proj_bias",
    "visual.transformer.resblocks.3.attn.out_proj.weight",
    "visual.transformer.resblocks.3.attn.out_proj.bias",
    "visual.transformer.resblocks.3.ln_1.weight",
    "visual.transformer.resblocks.3.ln_1.bias",
    "visual.transformer.resblocks.3.mlp.c_fc.weight",
    "visual.transformer.resblocks.3.mlp.c_fc.bias",
    "visual.transformer.resblocks.3.mlp.c_proj.weight",
    "visual.transformer.resblocks.3.mlp.c_proj.bias",
    "visual.transformer.resblocks.3.ln_2.weight",
    "visual.transformer.resblocks.3.ln_2.bias",
    "visual.transformer.resblocks.4.attn.in_proj_weight",
    "visual.transformer.resblocks.4.attn.in_proj_bias",
    "visual.transformer.resblocks.4.attn.out_proj.weight",
    "visual.transformer.resblocks.4.attn.out_proj.bias",
    "visual.transformer.resblocks.4.ln_1.weight",
    "visual.transformer.resblocks.4.ln_1.bias",
    "visual.transformer.resblocks.4.mlp.c_fc.weight",
    "visual.transformer.resblocks.4.mlp.c_fc.bias",
    "visual.transformer.resblocks.4.mlp.c_proj.weight",
    "visual.transformer.resblocks.4.mlp.c_proj.bias",
    "visual.transformer.resblocks.4.ln_2.weight",
    "visual.transformer.resblocks.4.ln_2.bias",
    "visual.transformer.resblocks.5.attn.in_proj_weight",
    "visual.transformer.resblocks.5.attn.in_proj_bias",
    "visual.transformer.resblocks.5.attn.out_proj.weight",
    "visual.transformer.resblocks.5.attn.out_proj.bias",
    "visual.transformer.resblocks.5.ln_1.weight",
    "visual.transformer.resblocks.5.ln_1.bias",
    "visual.transformer.resblocks.5.mlp.c_fc.weight",
    "visual.transformer.resblocks.5.mlp.c_fc.bias",
    "visual.transformer.resblocks.5.mlp.c_proj.weight",
    "visual.transformer.resblocks.5.mlp.c_proj.bias",
    "visual.transformer.resblocks.5.ln_2.weight",
    "visual.transformer.resblocks.5.ln_2.bias",
    "visual.transformer.resblocks.6.attn.in_proj_weight",
    "visual.transformer.resblocks.6.attn.in_proj_bias",
    "visual.transformer.resblocks.6.attn.out_proj.weight",
    "visual.transformer.resblocks.6.attn.out_proj.bias",
    "visual.transformer.resblocks.6.ln_1.weight",
    "visual.transformer.resblocks.6.ln_1.bias",
    "visual.transformer.resblocks.6.mlp.c_fc.weight",
    "visual.transformer.resblocks.6.mlp.c_fc.bias",
    "visual.transformer.resblocks.6.mlp.c_proj.weight",
    "visual.transformer.resblocks.6.mlp.c_proj.bias",
    "visual.transformer.resblocks.6.ln_2.weight",
    "visual.transformer.resblocks.6.ln_2.bias",
    "visual.transformer.resblocks.7.attn.in_proj_weight",
    "visual.transformer.resblocks.7.attn.in_proj_bias",
    "visual.transformer.resblocks.7.attn.out_proj.weight",
    "visual.transformer.resblocks.7.attn.out_proj.bias",
    "visual.transformer.resblocks.7.ln_1.weight",
    "visual.transformer.resblocks.7.ln_1.bias",
    "visual.transformer.resblocks.7.mlp.c_fc.weight",
    "visual.transformer.resblocks.7.mlp.c_fc.bias",
    "visual.transformer.resblocks.7.mlp.c_proj.weight",
    "visual.transformer.resblocks.7.mlp.c_proj.bias",
    "visual.transformer.resblocks.7.ln_2.weight",
    "visual.transformer.resblocks.7.ln_2.bias",
    "visual.transformer.resblocks.8.attn.in_proj_weight",
    "visual.transformer.resblocks.8.attn.in_proj_bias",
    "visual.transformer.resblocks.8.attn.out_proj.weight",
    "visual.transformer.resblocks.8.attn.out_proj.bias",
    "visual.transformer.resblocks.8.ln_1.weight",
    "visual.transformer.resblocks.8.ln_1.bias",
    "visual.transformer.resblocks.8.mlp.c_fc.weight",
    "visual.transformer.resblocks.8.mlp.c_fc.bias",
    "visual.transformer.resblocks.8.mlp.c_proj.weight",
    "visual.transformer.resblocks.8.mlp.c_proj.bias",
    "visual.transformer.resblocks.8.ln_2.weight",
    "visual.transformer.resblocks.8.ln_2.bias",
    "visual.transformer.resblocks.9.attn.in_proj_weight",
    "visual.transformer.resblocks.9.attn.in_proj_bias",
    "visual.transformer.resblocks.9.attn.out_proj.weight",
    "visual.transformer.resblocks.9.attn.out_proj.bias",
    "visual.transformer.resblocks.9.ln_1.weight",
    "visual.transformer.resblocks.9.ln_1.bias",
    "visual.transformer.resblocks.9.mlp.c_fc.weight",
    "visual.transformer.resblocks.9.mlp.c_fc.bias",
    "visual.transformer.resblocks.9.mlp.c_proj.weight",
    "visual.transformer.resblocks.9.mlp.c_proj.bias",
    "visual.transformer.resblocks.9.ln_2.weight",
    "visual.transformer.resblocks.9.ln_2.bias",
    "visual.transformer.resblocks.10.attn.in_proj_weight",
    "visual.transformer.resblocks.10.attn.in_proj_bias",
    "visual.transformer.resblocks.10.attn.out_proj.weight",
    "visual.transformer.resblocks.10.attn.out_proj.bias",
    "visual.transformer.resblocks.10.ln_1.weight",
    "visual.transformer.resblocks.10.ln_1.bias",
    "visual.transformer.resblocks.10.mlp.c_fc.weight",
    "visual.transformer.resblocks.10.mlp.c_fc.bias",
    "visual.transformer.resblocks.10.mlp.c_proj.weight",
    "visual.transformer.resblocks.10.mlp.c_proj.bias",
    "visual.transformer.resblocks.10.ln_2.weight",
    "visual.transformer.resblocks.10.ln_2.bias",
    "visual.transformer.resblocks.11.attn.in_proj_weight",
    "visual.transformer.resblocks.11.attn.in_proj_bias",
    "visual.transformer.resblocks.11.attn.out_proj.weight",
    "visual.transformer.resblocks.11.attn.out_proj.bias",
    "visual.transformer.resblocks.11.ln_1.weight",
    "visual.transformer.resblocks.11.ln_1.bias",
    "visual.transformer.resblocks.11.mlp.c_fc.weight",
    "visual.transformer.resblocks.11.mlp.c_fc.bias",
    "visual.transformer.resblocks.11.mlp.c_proj.weight",
    "visual.transformer.resblocks.11.mlp.c_proj.bias",
    "visual.transformer.resblocks.11.ln_2.weight",
    "visual.transformer.resblocks.11.ln_2.bias",
    "visual.transformer.resblocks.12.attn.in_proj_weight",
    "visual.transformer.resblocks.12.attn.in_proj_bias",
    "visual.transformer.resblocks.12.attn.out_proj.weight",
    "visual.transformer.resblocks.12.attn.out_proj.bias",
    "visual.transformer.resblocks.12.ln_1.weight",
    "visual.transformer.resblocks.12.ln_1.bias",
    "visual.transformer.resblocks.12.mlp.c_fc.weight",
    "visual.transformer.resblocks.12.mlp.c_fc.bias",
    "visual.transformer.resblocks.12.mlp.c_proj.weight",
    "visual.transformer.resblocks.12.mlp.c_proj.bias",
    "visual.transformer.resblocks.12.ln_2.weight",
    "visual.transformer.resblocks.12.ln_2.bias",
    "visual.transformer.resblocks.13.attn.in_proj_weight",
    "visual.transformer.resblocks.13.attn.in_proj_bias",
    "visual.transformer.resblocks.13.attn.out_proj.weight",
    "visual.transformer.resblocks.13.attn.out_proj.bias",
    "visual.transformer.resblocks.13.ln_1.weight",
    "visual.transformer.resblocks.13.ln_1.bias",
    "visual.transformer.resblocks.13.mlp.c_fc.weight",
    "visual.transformer.resblocks.13.mlp.c_fc.bias",
    "visual.transformer.resblocks.13.mlp.c_proj.weight",
    "visual.transformer.resblocks.13.mlp.c_proj.bias",
    "visual.transformer.resblocks.13.ln_2.weight",
    "visual.transformer.resblocks.13.ln_2.bias",
    "visual.transformer.resblocks.14.attn.in_proj_weight",
    "visual.transformer.resblocks.14.attn.in_proj_bias",
    "visual.transformer.resblocks.14.attn.out_proj.weight",
    "visual.transformer.resblocks.14.attn.out_proj.bias",
    "visual.transformer.resblocks.14.ln_1.weight",
    "visual.transformer.resblocks.14.ln_1.bias",
    "visual.transformer.resblocks.14.mlp.c_fc.weight",
    "visual.transformer.resblocks.14.mlp.c_fc.bias",
    "visual.transformer.resblocks.14.mlp.c_proj.weight",
    "visual.transformer.resblocks.14.mlp.c_proj.bias",
    "visual.transformer.resblocks.14.ln_2.weight",
    "visual.transformer.resblocks.14.ln_2.bias",
    "visual.transformer.resblocks.15.attn.in_proj_weight",
    "visual.transformer.resblocks.15.attn.in_proj_bias",
    "visual.transformer.resblocks.15.attn.out_proj.weight",
    "visual.transformer.resblocks.15.attn.out_proj.bias",
    "visual.transformer.resblocks.15.ln_1.weight",
    "visual.transformer.resblocks.15.ln_1.bias",
    "visual.transformer.resblocks.15.mlp.c_fc.weight",
    "visual.transformer.resblocks.15.mlp.c_fc.bias",
    "visual.transformer.resblocks.15.mlp.c_proj.weight",
    "visual.transformer.resblocks.15.mlp.c_proj.bias",
    "visual.transformer.resblocks.15.ln_2.weight",
    "visual.transformer.resblocks.15.ln_2.bias",
    "visual.transformer.resblocks.16.attn.in_proj_weight",
    "visual.transformer.resblocks.16.attn.in_proj_bias",
    "visual.transformer.resblocks.16.attn.out_proj.weight",
    "visual.transformer.resblocks.16.attn.out_proj.bias",
    "visual.transformer.resblocks.16.ln_1.weight",
    "visual.transformer.resblocks.16.ln_1.bias",
    "visual.transformer.resblocks.16.mlp.c_fc.weight",
    "visual.transformer.resblocks.16.mlp.c_fc.bias",
    "visual.transformer.resblocks.16.mlp.c_proj.weight",
    "visual.transformer.resblocks.16.mlp.c_proj.bias",
    "visual.transformer.resblocks.16.ln_2.weight",
    "visual.transformer.resblocks.16.ln_2.bias",
    "visual.transformer.resblocks.17.attn.in_proj_weight",
    "visual.transformer.resblocks.17.attn.in_proj_bias",
    "visual.transformer.resblocks.17.attn.out_proj.weight",
    "visual.transformer.resblocks.17.attn.out_proj.bias",
    "visual.transformer.resblocks.17.ln_1.weight",
    "visual.transformer.resblocks.17.ln_1.bias",
    "visual.transformer.resblocks.17.mlp.c_fc.weight",
    "visual.transformer.resblocks.17.mlp.c_fc.bias",
    "visual.transformer.resblocks.17.mlp.c_proj.weight",
    "visual.transformer.resblocks.17.mlp.c_proj.bias",
    "visual.transformer.resblocks.17.ln_2.weight",
    "visual.transformer.resblocks.17.ln_2.bias",
    "visual.transformer.resblocks.18.attn.in_proj_weight",
    "visual.transformer.resblocks.18.attn.in_proj_bias",
    "visual.transformer.resblocks.18.attn.out_proj.weight",
    "visual.transformer.resblocks.18.attn.out_proj.bias",
    "visual.transformer.resblocks.18.ln_1.weight",
    "visual.transformer.resblocks.18.ln_1.bias",
    "visual.transformer.resblocks.18.mlp.c_fc.weight",
    "visual.transformer.resblocks.18.mlp.c_fc.bias",
    "visual.transformer.resblocks.18.mlp.c_proj.weight",
    "visual.transformer.resblocks.18.mlp.c_proj.bias",
    "visual.transformer.resblocks.18.ln_2.weight",
    "visual.transformer.resblocks.18.ln_2.bias",
    "visual.transformer.resblocks.19.attn.in_proj_weight",
    "visual.transformer.resblocks.19.attn.in_proj_bias",
    "visual.transformer.resblocks.19.attn.out_proj.weight",
    "visual.transformer.resblocks.19.attn.out_proj.bias",
    "visual.transformer.resblocks.19.ln_1.weight",
    "visual.transformer.resblocks.19.ln_1.bias",
    "visual.transformer.resblocks.19.mlp.c_fc.weight",
    "visual.transformer.resblocks.19.mlp.c_fc.bias",
    "visual.transformer.resblocks.19.mlp.c_proj.weight",
    "visual.transformer.resblocks.19.mlp.c_proj.bias",
    "visual.transformer.resblocks.19.ln_2.weight",
    "visual.transformer.resblocks.19.ln_2.bias",
    "visual.transformer.resblocks.20.attn.in_proj_weight",
    "visual.transformer.resblocks.20.attn.in_proj_bias",
    "visual.transformer.resblocks.20.attn.out_proj.weight",
    "visual.transformer.resblocks.20.attn.out_proj.bias",
    "visual.transformer.resblocks.20.ln_1.weight",
    "visual.transformer.resblocks.20.ln_1.bias",
    "visual.transformer.resblocks.20.mlp.c_fc.weight",
    "visual.transformer.resblocks.20.mlp.c_fc.bias",
    "visual.transformer.resblocks.20.mlp.c_proj.weight",
    "visual.transformer.resblocks.20.mlp.c_proj.bias",
    "visual.transformer.resblocks.20.ln_2.weight",
    "visual.transformer.resblocks.20.ln_2.bias",
    "visual.transformer.resblocks.21.attn.in_proj_weight",
    "visual.transformer.resblocks.21.attn.in_proj_bias",
    "visual.transformer.resblocks.21.attn.out_proj.weight",
    "visual.transformer.resblocks.21.attn.out_proj.bias",
    "visual.transformer.resblocks.21.ln_1.weight",
    "visual.transformer.resblocks.21.ln_1.bias",
    "visual.transformer.resblocks.21.mlp.c_fc.weight",
    "visual.transformer.resblocks.21.mlp.c_fc.bias",
    "visual.transformer.resblocks.21.mlp.c_proj.weight",
    "visual.transformer.resblocks.21.mlp.c_proj.bias",
    "visual.transformer.resblocks.21.ln_2.weight",
    "visual.transformer.resblocks.21.ln_2.bias",
    "visual.transformer.resblocks.22.attn.in_proj_weight",
    "visual.transformer.resblocks.22.attn.in_proj_bias",
    "visual.transformer.resblocks.22.attn.out_proj.weight",
    "visual.transformer.resblocks.22.attn.out_proj.bias",
    "visual.transformer.resblocks.22.ln_1.weight",
    "visual.transformer.resblocks.22.ln_1.bias",
    "visual.transformer.resblocks.22.mlp.c_fc.weight",
    "visual.transformer.resblocks.22.mlp.c_fc.bias",
    "visual.transformer.resblocks.22.mlp.c_proj.weight",
    "visual.transformer.resblocks.22.mlp.c_proj.bias",
    "visual.transformer.resblocks.22.ln_2.weight",
    "visual.transformer.resblocks.22.ln_2.bias",
    "visual.transformer.resblocks.23.attn.in_proj_weight",
    "visual.transformer.resblocks.23.attn.in_proj_bias",
    "visual.transformer.resblocks.23.attn.out_proj.weight",
    "visual.transformer.resblocks.23.attn.out_proj.bias",
    "visual.transformer.resblocks.23.ln_1.weight",
    "visual.transformer.resblocks.23.ln_1.bias",
    "visual.transformer.resblocks.23.mlp.c_fc.weight",
    "visual.transformer.resblocks.23.mlp.c_fc.bias",
    "visual.transformer.resblocks.23.mlp.c_proj.weight",
    "visual.transformer.resblocks.23.mlp.c_proj.bias",
    "visual.transformer.resblocks.23.ln_2.weight",
    "visual.transformer.resblocks.23.ln_2.bias",
    "visual.ln_post.weight",
    "visual.ln_post.bias",
    "transformer.resblocks.0.attn.in_proj_weight",
    "transformer.resblocks.0.attn.in_proj_bias",
    "transformer.resblocks.0.attn.out_proj.weight",
    "transformer.resblocks.0.attn.out_proj.bias",
    "transformer.resblocks.0.ln_1.weight",
    "transformer.resblocks.0.ln_1.bias",
    "transformer.resblocks.0.mlp.c_fc.weight",
    "transformer.resblocks.0.mlp.c_fc.bias",
    "transformer.resblocks.0.mlp.c_proj.weight",
    "transformer.resblocks.0.mlp.c_proj.bias",
    "transformer.resblocks.0.ln_2.weight",
    "transformer.resblocks.0.ln_2.bias",
    "transformer.resblocks.1.attn.in_proj_weight",
    "transformer.resblocks.1.attn.in_proj_bias",
    "transformer.resblocks.1.attn.out_proj.weight",
    "transformer.resblocks.1.attn.out_proj.bias",
    "transformer.resblocks.1.ln_1.weight",
    "transformer.resblocks.1.ln_1.bias",
    "transformer.resblocks.1.mlp.c_fc.weight",
    "transformer.resblocks.1.mlp.c_fc.bias",
    "transformer.resblocks.1.mlp.c_proj.weight",
    "transformer.resblocks.1.mlp.c_proj.bias",
    "transformer.resblocks.1.ln_2.weight",
    "transformer.resblocks.1.ln_2.bias",
    "transformer.resblocks.2.attn.in_proj_weight",
    "transformer.resblocks.2.attn.in_proj_bias",
    "transformer.resblocks.2.attn.out_proj.weight",
    "transformer.resblocks.2.attn.out_proj.bias",
    "transformer.resblocks.2.ln_1.weight",
    "transformer.resblocks.2.ln_1.bias",
    "transformer.resblocks.2.mlp.c_fc.weight",
    "transformer.resblocks.2.mlp.c_fc.bias",
    "transformer.resblocks.2.mlp.c_proj.weight",
    "transformer.resblocks.2.mlp.c_proj.bias",
    "transformer.resblocks.2.ln_2.weight",
    "transformer.resblocks.2.ln_2.bias",
    "transformer.resblocks.3.attn.in_proj_weight",
    "transformer.resblocks.3.attn.in_proj_bias",
    "transformer.resblocks.3.attn.out_proj.weight",
    "transformer.resblocks.3.attn.out_proj.bias",
    "transformer.resblocks.3.ln_1.weight",
    "transformer.resblocks.3.ln_1.bias",
    "transformer.resblocks.3.mlp.c_fc.weight",
    "transformer.resblocks.3.mlp.c_fc.bias",
    "transformer.resblocks.3.mlp.c_proj.weight",
    "transformer.resblocks.3.mlp.c_proj.bias",
    "transformer.resblocks.3.ln_2.weight",
    "transformer.resblocks.3.ln_2.bias",
    "transformer.resblocks.4.attn.in_proj_weight",
    "transformer.resblocks.4.attn.in_proj_bias",
    "transformer.resblocks.4.attn.out_proj.weight",
    "transformer.resblocks.4.attn.out_proj.bias",
    "transformer.resblocks.4.ln_1.weight",
    "transformer.resblocks.4.ln_1.bias",
    "transformer.resblocks.4.mlp.c_fc.weight",
    "transformer.resblocks.4.mlp.c_fc.bias",
    "transformer.resblocks.4.mlp.c_proj.weight",
    "transformer.resblocks.4.mlp.c_proj.bias",
    "transformer.resblocks.4.ln_2.weight",
    "transformer.resblocks.4.ln_2.bias",
    "transformer.resblocks.5.attn.in_proj_weight",
    "transformer.resblocks.5.attn.in_proj_bias",
    "transformer.resblocks.5.attn.out_proj.weight",
    "transformer.resblocks.5.attn.out_proj.bias",
    "transformer.resblocks.5.ln_1.weight",
    "transformer.resblocks.5.ln_1.bias",
    "transformer.resblocks.5.mlp.c_fc.weight",
    "transformer.resblocks.5.mlp.c_fc.bias",
    "transformer.resblocks.5.mlp.c_proj.weight",
    "transformer.resblocks.5.mlp.c_proj.bias",
    "transformer.resblocks.5.ln_2.weight",
    "transformer.resblocks.5.ln_2.bias",
    "transformer.resblocks.6.attn.in_proj_weight",
    "transformer.resblocks.6.attn.in_proj_bias",
    "transformer.resblocks.6.attn.out_proj.weight",
    "transformer.resblocks.6.attn.out_proj.bias",
    "transformer.resblocks.6.ln_1.weight",
    "transformer.resblocks.6.ln_1.bias",
    "transformer.resblocks.6.mlp.c_fc.weight",
    "transformer.resblocks.6.mlp.c_fc.bias",
    "transformer.resblocks.6.mlp.c_proj.weight",
    "transformer.resblocks.6.mlp.c_proj.bias",
    "transformer.resblocks.6.ln_2.weight",
    "transformer.resblocks.6.ln_2.bias",
    "transformer.resblocks.7.attn.in_proj_weight",
    "transformer.resblocks.7.attn.in_proj_bias",
    "transformer.resblocks.7.attn.out_proj.weight",
    "transformer.resblocks.7.attn.out_proj.bias",
    "transformer.resblocks.7.ln_1.weight",
    "transformer.resblocks.7.ln_1.bias",
    "transformer.resblocks.7.mlp.c_fc.weight",
    "transformer.resblocks.7.mlp.c_fc.bias",
    "transformer.resblocks.7.mlp.c_proj.weight",
    "transformer.resblocks.7.mlp.c_proj.bias",
    "transformer.resblocks.7.ln_2.weight",
    "transformer.resblocks.7.ln_2.bias",
    "transformer.resblocks.8.attn.in_proj_weight",
    "transformer.resblocks.8.attn.in_proj_bias",
    "transformer.resblocks.8.attn.out_proj.weight",
    "transformer.resblocks.8.attn.out_proj.bias",
    "transformer.resblocks.8.ln_1.weight",
    "transformer.resblocks.8.ln_1.bias",
    "transformer.resblocks.8.mlp.c_fc.weight",
    "transformer.resblocks.8.mlp.c_fc.bias",
    "transformer.resblocks.8.mlp.c_proj.weight",
    "transformer.resblocks.8.mlp.c_proj.bias",
    "transformer.resblocks.8.ln_2.weight",
    "transformer.resblocks.8.ln_2.bias",
    "transformer.resblocks.9.attn.in_proj_weight",
    "transformer.resblocks.9.attn.in_proj_bias",
    "transformer.resblocks.9.attn.out_proj.weight",
    "transformer.resblocks.9.attn.out_proj.bias",
    "transformer.resblocks.9.ln_1.weight",
    "transformer.resblocks.9.ln_1.bias",
    "transformer.resblocks.9.mlp.c_fc.weight",
    "transformer.resblocks.9.mlp.c_fc.bias",
    "transformer.resblocks.9.mlp.c_proj.weight",
    "transformer.resblocks.9.mlp.c_proj.bias",
    "transformer.resblocks.9.ln_2.weight",
    "transformer.resblocks.9.ln_2.bias",
    "transformer.resblocks.10.attn.in_proj_weight",
    "transformer.resblocks.10.attn.in_proj_bias",
    "transformer.resblocks.10.attn.out_proj.weight",
    "transformer.resblocks.10.attn.out_proj.bias",
    "transformer.resblocks.10.ln_1.weight",
    "transformer.resblocks.10.ln_1.bias",
    "transformer.resblocks.10.mlp.c_fc.weight",
    "transformer.resblocks.10.mlp.c_fc.bias",
    "transformer.resblocks.10.mlp.c_proj.weight",
    "transformer.resblocks.10.mlp.c_proj.bias",
    "transformer.resblocks.10.ln_2.weight",
    "transformer.resblocks.10.ln_2.bias",
    "transformer.resblocks.11.attn.in_proj_weight",
    "transformer.resblocks.11.attn.in_proj_bias",
    "transformer.resblocks.11.attn.out_proj.weight",
    "transformer.resblocks.11.attn.out_proj.bias",
    "transformer.resblocks.11.ln_1.weight",
    "transformer.resblocks.11.ln_1.bias",
    "transformer.resblocks.11.mlp.c_fc.weight",
    "transformer.resblocks.11.mlp.c_fc.bias",
    "transformer.resblocks.11.mlp.c_proj.weight",
    "transformer.resblocks.11.mlp.c_proj.bias",
    "transformer.resblocks.11.ln_2.weight",
    "transformer.resblocks.11.ln_2.bias",
    "token_embedding.weight",
    "ln_final.weight",
    "ln_final.bias",
]
