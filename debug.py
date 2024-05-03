import torch

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (
    manifold,
    datasets,
    decomposition,
    ensemble,
    discriminant_analysis,
    random_projection,
)

## Loading and curating the data
digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target

n_samples, n_features = X.shape
n_neighbors = 30


## Function to Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(
            X[i, 0],
            X[i, 1],
            str(digits.target[i]),
            color=plt.cm.Set1(y[i] / 10.0),
            fontdict={"weight": "bold", "size": 9},
        )
    if hasattr(offsetbox, "AnnotationBbox"):
        ## only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1.0, 1.0]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                ## don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
            )
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# ----------------------------------------------------------------------
## Plot images of the digits
n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix : ix + 8, iy : iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))
plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title("A selection from the 64-dimensional digits dataset")
## Computing PCA
print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
plot_embedding(
    X_pca, "Principal Components projection of the digits (time %.2fs)" % (time() - t0)
)
## Computing t-SNE
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init="pca", random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne, "t-SNE embedding of the digits (time %.2fs)" % (time() - t0))
plt.show()

exit()
device = "cuda" if torch.cuda.is_available() else "cpu"


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
