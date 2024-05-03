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


datasets = ["btad", "dagm", "dtd", "mpdd", "mvtec", "sdd", "visa"]
categories_by_dataset = {
    "btad": ["01", "02", "03"],
    "dagm": [
        "Class1",
        "Class2",
        "Class3",
        "Class4",
        "Class5",
        "Class6",
        "Class7",
        "Class8",
        "Class9",
        "Class10",
    ],
    "dtd": [
        "Woven_001",
        "Woven_127",
        "Woven_104",
        "Stratified_154",
        "Blotchy_099",
        "Woven_068",
        "Woven_125",
        "Marbled_078",
        "Perforated_037",
        "Mesh_114",
        "Fibrous_183",
        "Matted_069",
    ],
    "mpdd": [
        "bracket_black",
        "bracket_brown",
        "bracket_white",
        "connector",
        "metal_plate",
        "tubes",
    ],
    "mvtec": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ],
    "sdd": [
        "SDD",
    ],
    "visa": [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ],
}


FIXED_LABELS = 4
LABELS_4each_CATEGORY = 4  # [i+, i-, t+, t-]

# FIRST FOR LOOP [dataset]
for dataset in datasets:
    # set up
    categories = categories_by_dataset[dataset]
    category_dict = {}
    for category in categories:
        category_dict[category] = []

    # open files
    with open(f"{dataset}_baseline.t", "rb") as f:
        baseline = torch.load(f, map_location=device)
    with open(f"{dataset}_metanet_pag.t", "rb") as f:
        metanet = torch.load(f, map_location=device)

    # text features prior
    baseline_text_prior = baseline["prior_text_feature"][0].detach().numpy()  # [2, 768]
    metanet_text_prior = metanet["prior_text_feature"][0].detach().numpy()
    baseline_text_prior_pos = baseline_text_prior[0]  # [768]
    baseline_text_prior_neg = baseline_text_prior[1]  # [768]
    metanet_text_prior_pos = metanet_text_prior[0]  # [768]
    metanet_text_prior_neg = metanet_text_prior[1]  # [768]

    # read individuals
    baseline_individuals = baseline["individual"]
    metanet_individuals = metanet["individual"]
    for baseline, metanet in zip(baseline_individuals, metanet_individuals):
        info = {}
        class_name = baseline["class_name"]  # text

        info["image_features"] = baseline["image_features"].detach().numpy()  # [768]
        info["gt_anomaly"] = baseline["gt_anomaly"].detach().numpy()  # 0 or 1
        metanet_text = (
            metanet["text_features"].detach().numpy()
        )  # [2, 768] metanet adjusted
        info["metanet_text_pos"] = metanet_text[0]  # [768] metanet adjusted
        info["metanet_text_neg"] = metanet_text[1]  # [768] metanet adjusted
        info["gt_mask"] = baseline["gt_mask"].detach().numpy()  # [518, 518]
        info["patch_features"] = (
            baseline["patch_features"].detach().numpy()
        )  # [37, 37, 768]
        category_dict[class_name].append(info)

    # SECOND FOR LOOP [category]
    for category in categories:
        content_X = []
        content_Y = []

        content_X.append(baseline_text_prior_pos)
        content_Y.append(0)
        content_X.append(baseline_text_prior_neg)
        content_Y.append(1)
        content_X.append(metanet_text_prior_pos)
        content_Y.append(2)
        content_X.append(metanet_text_prior_neg)
        content_Y.append(3)

        # read data
        for sample in category_dict[category]:
            image_features = sample["image_features"]
            gt_anomaly = sample["gt_anomaly"]
            content_X.append(image_features)
            content_Y.append(FIXED_LABELS + gt_anomaly)
            metanet_text_pos = sample["metanet_text_pos"]
            content_X.append(metanet_text_pos)
            content_Y.append(FIXED_LABELS + 2)
            metanet_text_neg = sample["metanet_text_neg"]
            content_X.append(metanet_text_neg)
            content_Y.append(FIXED_LABELS + 3)

        # MAPPING: label -> color/label/shape
        category_to_label = {
            0: "bs+",
            1: "bs-",
            2: "mt+",
            3: "mt-",
            4: "[I]+",
            5: "[I]-",
            6: "+",
            7: "-",
        }

        category_to_color = {
            0: "#0D0D04",
            1: "#0D0D04",
            2: "#F63110",
            3: "#F63110",
            4: "#20603D",
            5: "#6EC193",
            6: "#20603D",
            7: "#6EC193",
        }

        tsne = TSNE(
            n_components=2,
            init="pca",
        )
        content_X = np.array(content_X)
        content_Y = np.array(content_Y)
        transformed_x = tsne.fit_transform(content_X)

        fig, ax = plt.subplots()

        ax.set(title=f"IMAGE-LEVEL [Dataset] {dataset} [Category] {category}")
        for category_id, label in category_to_label.items():
            mask = content_Y == category_id

            marker_choice = None
            if category_id in [0, 2, 6]:  # text, +
                marker_choice = 6
            elif category_id in [1, 3, 7]:  # text, -
                marker_choice = 7
            elif category_id == 4:  # image, +
                marker_choice = "+"
            elif category_id == 5:  # image, -
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

        ax.legend()
        plt.savefig(f"Image_level_{dataset}_{category}.png")
