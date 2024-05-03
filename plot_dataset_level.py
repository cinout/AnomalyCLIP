import torch
import numpy as np
import pandas as pd

# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: update
datasets = ["btad"]
categories_by_dataset = {"btad": ["01", "02", "03"]}


FIXED_LABELS = 4
LABELS_4each_CATEGORY = 4  # [i+, i-, t+, t-]

for dataset in datasets:
    categories = categories_by_dataset[dataset]

    with open(f"{dataset}_baseline.t", "rb") as f:
        baseline = torch.load(f, map_location=device)
    with open(f"{dataset}_metanet_pag.t", "rb") as f:
        metanet = torch.load(f, map_location=device)

    content_X = []
    content_Y = []

    # text features prior
    baseline_text_prior = (
        baseline["prior_text_feature"][0].detach().cpu().numpy()
    )  # [2, 768]
    metanet_text_prior = metanet["prior_text_feature"][0].detach().cpu().numpy()

    baseline_text_prior_pos = baseline_text_prior[0]  # [768]
    content_X.append(baseline_text_prior_pos)
    content_Y.append(0)
    baseline_text_prior_neg = baseline_text_prior[1]  # [768]
    content_X.append(baseline_text_prior_neg)
    content_Y.append(1)

    metanet_text_prior_pos = metanet_text_prior[0]  # [768]
    content_X.append(metanet_text_prior_pos)
    content_Y.append(2)

    metanet_text_prior_neg = metanet_text_prior[1]  # [768]
    content_X.append(metanet_text_prior_neg)
    content_Y.append(3)

    # individuals
    baseline_individuals = baseline["individual"]
    metanet_individuals = metanet["individual"]
    for baseline, metanet in zip(baseline_individuals, metanet_individuals):

        """
        Image Level Features
        """
        image_features = baseline["image_features"].detach().cpu().numpy()  # [768]
        content_X.append(image_features)

        gt_anomaly = baseline["gt_anomaly"].detach().cpu().numpy()  # 0 or 1
        class_name = baseline["class_name"]  # text
        category_id = (
            FIXED_LABELS
            + categories.index(class_name) * LABELS_4each_CATEGORY
            + gt_anomaly
        )
        content_Y.append(category_id)

        """
        Metanet Text Features
        """
        metanet_text = (
            metanet["text_features"].detach().cpu().numpy()
        )  # [2, 768] metanet adjusted

        metanet_text_pos = metanet_text[0]  # [768]
        content_X.append(metanet_text_pos)
        category_id = (
            FIXED_LABELS + categories.index(class_name) * LABELS_4each_CATEGORY + 2
        )
        content_Y.append(category_id)

        metanet_text_neg = metanet_text[1]  # [768]
        content_X.append(metanet_text_neg)
        category_id = (
            FIXED_LABELS + categories.index(class_name) * LABELS_4each_CATEGORY + 3
        )
        content_Y.append(category_id)

        """
        Patch Level Features
        """
        gt_mask = baseline["gt_mask"].detach().cpu().numpy()  # [518, 518]
        patch_features = (
            baseline["patch_features"].detach().cpu().numpy()
        )  # [37, 37, 768]

    # MAPPING: label -> color/label/shape
    category_to_label = {
        0: "[t] bs_+",
        1: "[t] bs_-",
        2: "[t] mt_+",
        3: "[t] mt_-",
    }
    for idx, category in enumerate(categories):
        legend_name = "[i] " + category + "_+"
        category_to_label[FIXED_LABELS + idx * LABELS_4each_CATEGORY] = legend_name
        legend_name = "[i] " + category + "_-"
        category_to_label[FIXED_LABELS + idx * LABELS_4each_CATEGORY + 1] = legend_name
        legend_name = "[t] " + category + "_+"
        category_to_label[FIXED_LABELS + idx * LABELS_4each_CATEGORY + 2] = legend_name
        legend_name = "[t] " + category + "_-"
        category_to_label[FIXED_LABELS + idx * LABELS_4each_CATEGORY + 3] = legend_name

    # TODO: update
    category_to_color = {
        0: "#0D0D04",
        1: "#0D0D04",
        2: "#F63110",
        3: "#F63110",
        4: "#20603D",
        5: "#20603D",
        6: "#20603D",
        7: "#20603D",
        8: "#C1876B",
        9: "#C1876B",
        10: "#C1876B",
        11: "#C1876B",
        12: "#3B83BD",
        13: "#3B83BD",
        14: "#3B83BD",
        15: "#3B83BD",
    }

    tsne = TSNE(
        n_components=2,
        init="pca",
    )
    content_X = np.array(content_X)
    content_Y = np.array(content_Y)
    transformed_x = tsne.fit_transform(content_X)

    fig, ax = plt.subplots()
    for category_id, label in category_to_label.items():
        mask = content_Y == category_id

        marker_choice = None
        if category_id in [0, 2] or category_id % 4 == 2:  # text, +
            marker_choice = 6
        elif category_id in [1, 3] or category_id % 4 == 3:  # text, baseline, prior, -
            marker_choice = 7
        elif category_id % 4 == 0:  # image, +
            marker_choice = "+"
        elif category_id % 4 == 1:  # image, -
            marker_choice = "*"

        scale = None
        if category_id in [0, 1, 2, 3]:
            scale = 128

        ax.scatter(
            transformed_x[mask, 0],
            transformed_x[mask, 1],
            label=label,
            c=category_to_color[category_id],
            marker=marker_choice,
            s=scale,
            # s=0.8 if category_id in [0, 1, 2, 3] else 0.3,
        )

    # legend = ax.legend(loc="lower right", shadow=True)
    # legend.get_frame()
    ax.legend()
    plt.show()
