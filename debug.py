import torch
import numpy as np
import pandas as pd

# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"


with open("mpdd_metanet_pag.t", "rb") as f:
    mpdd_metanet_pag = torch.load(f, map_location=device)

print(mpdd_metanet_pag)

exit()
p1dist = torch.nn.PairwiseDistance(p=1)
p2dist = torch.nn.PairwiseDistance(p=2)


"""
MVTEC checkpoints
"""
# baseline
pretrained_mvtec_baseline_path = (
    "./checkpoints/pretrained_mvtec_baseline_10/epoch_15.pth"
)
checkpoint_mvtec_baseline = torch.load(
    pretrained_mvtec_baseline_path, map_location=device
)
mvtec_baseline_ctx_pos = checkpoint_mvtec_baseline["prompt_learner"][
    "ctx_pos"
]  # [1, 1, 12, 768]
mvtec_baseline_ctx_pos = mvtec_baseline_ctx_pos.reshape(
    -1, mvtec_baseline_ctx_pos.shape[-1]
)  # [12, 768]
mvtec_baseline_ctx_neg = checkpoint_mvtec_baseline["prompt_learner"]["ctx_neg"]
mvtec_baseline_ctx_neg = mvtec_baseline_ctx_neg.reshape(
    -1, mvtec_baseline_ctx_neg.shape[-1]
)
p1dist_mvtec_baseline = p1dist(mvtec_baseline_ctx_pos, mvtec_baseline_ctx_neg)
p1dist_mvtec_baseline = torch.mean(p1dist_mvtec_baseline)
p2dist_mvtec_baseline = p2dist(mvtec_baseline_ctx_pos, mvtec_baseline_ctx_neg)
p2dist_mvtec_baseline = torch.mean(p2dist_mvtec_baseline)

print(f"p1dist_mvtec_baseline: {p1dist_mvtec_baseline}")
print(f"p2dist_mvtec_baseline: {p2dist_mvtec_baseline}")


# metanet_patchandglobal
pretrained_mvtec_metanet_patchandglobal_path = (
    "./checkpoints/pretrained_mvtec_metanet_patchandglobal_10/epoch_15.pth"
)
checkpoint_mvtec_metanet_patchandglobal = torch.load(
    pretrained_mvtec_metanet_patchandglobal_path, map_location=device
)
mvtec_metanet_ctx_pos = checkpoint_mvtec_metanet_patchandglobal["prompt_learner"][
    "ctx_pos"
]  # [1, 1, 12, 768]
mvtec_metanet_ctx_pos = mvtec_metanet_ctx_pos.reshape(
    -1, mvtec_metanet_ctx_pos.shape[-1]
)  # [12, 768]
mvtec_metanet_ctx_neg = checkpoint_mvtec_metanet_patchandglobal["prompt_learner"][
    "ctx_neg"
]
mvtec_metanet_ctx_neg = mvtec_metanet_ctx_neg.reshape(
    -1, mvtec_metanet_ctx_neg.shape[-1]
)
p1dist_mvtec_metanet = p1dist(mvtec_metanet_ctx_pos, mvtec_metanet_ctx_neg)
p1dist_mvtec_metanet = torch.mean(p1dist_mvtec_metanet)
p2dist_mvtec_metanet = p2dist(mvtec_metanet_ctx_pos, mvtec_metanet_ctx_neg)
p2dist_mvtec_metanet = torch.mean(p2dist_mvtec_metanet)
print(f"p1dist_mvtec_metanet: {p1dist_mvtec_metanet}")
print(f"p2dist_mvtec_metanet: {p2dist_mvtec_metanet}")


"""
VISA checkpoints
"""
# baseline
pretrained_visa_baseline_path = "./checkpoints/pretrained_visa_baseline_10/epoch_15.pth"
checkpoint_visa_baseline = torch.load(
    pretrained_visa_baseline_path, map_location=device
)
visa_baseline_ctx_pos = checkpoint_visa_baseline["prompt_learner"][
    "ctx_pos"
]  # [1, 1, 12, 768]
visa_baseline_ctx_pos = visa_baseline_ctx_pos.reshape(
    -1, visa_baseline_ctx_pos.shape[-1]
)  # [12, 768]
visa_baseline_ctx_neg = checkpoint_visa_baseline["prompt_learner"]["ctx_neg"]
visa_baseline_ctx_neg = visa_baseline_ctx_neg.reshape(
    -1, visa_baseline_ctx_neg.shape[-1]
)
p1dist_visa_baseline = p1dist(visa_baseline_ctx_pos, visa_baseline_ctx_neg)
p1dist_visa_baseline = torch.mean(p1dist_visa_baseline)
p2dist_visa_baseline = p2dist(visa_baseline_ctx_pos, visa_baseline_ctx_neg)
p2dist_visa_baseline = torch.mean(p2dist_visa_baseline)

print(f"p1dist_visa_baseline: {p1dist_visa_baseline}")
print(f"p2dist_visa_baseline: {p2dist_visa_baseline}")


# metanet_patchandglobal
pretrained_visa_metanet_patchandglobal_path = (
    "./checkpoints/pretrained_visa_metanet_patchandglobal_10/epoch_15.pth"
)
checkpoint_visa_metanet_patchandglobal = torch.load(
    pretrained_visa_metanet_patchandglobal_path, map_location=device
)
visa_metanet_ctx_pos = checkpoint_visa_metanet_patchandglobal["prompt_learner"][
    "ctx_pos"
]  # [1, 1, 12, 768]
visa_metanet_ctx_pos = visa_metanet_ctx_pos.reshape(
    -1, visa_metanet_ctx_pos.shape[-1]
)  # [12, 768]
visa_metanet_ctx_neg = checkpoint_visa_metanet_patchandglobal["prompt_learner"][
    "ctx_neg"
]
visa_metanet_ctx_neg = visa_metanet_ctx_neg.reshape(-1, visa_metanet_ctx_neg.shape[-1])
p1dist_visa_metanet = p1dist(visa_metanet_ctx_pos, visa_metanet_ctx_neg)
p1dist_visa_metanet = torch.mean(p1dist_visa_metanet)
p2dist_visa_metanet = p2dist(visa_metanet_ctx_pos, visa_metanet_ctx_neg)
p2dist_visa_metanet = torch.mean(p2dist_visa_metanet)
print(f"p1dist_visa_metanet: {p1dist_visa_metanet}")
print(f"p2dist_visa_metanet: {p2dist_mvtec_metanet}")


"""
p1dist_mvtec_baseline: 32.23704147338867
p1dist_mvtec_metanet: 50.8098258972168

p2dist_mvtec_baseline: 1.4645838737487793
p2dist_mvtec_metanet: 2.30246901512146

p1dist_visa_baseline: 35.094051361083984
p1dist_visa_metanet: 66.6487045288086

p2dist_visa_baseline: 1.5979334115982056
p2dist_visa_metanet: 2.30246901512146
"""
