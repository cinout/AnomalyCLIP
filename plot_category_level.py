import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

tsne_types = ["patch", "image"]  # "patch", "image"


just_visual = False  # True, False
# TODO: UPDATE
auto_percentage = False
# TODO: UPDATE
# datasets = ["btad"]
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
    baseline_text_prior = (
        baseline["prior_text_feature"][0].detach().cpu().numpy()
    )  # [2, 768]
    metanet_text_prior = metanet["prior_text_feature"][0].detach().cpu().numpy()
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

        info["image_features"] = (
            baseline["image_features"].detach().cpu().numpy()
        )  # [768]
        info["gt_anomaly"] = baseline["gt_anomaly"].detach().cpu().numpy()  # 0 or 1
        metanet_text = (
            metanet["text_features"].detach().cpu().numpy()
        )  # [2, 768] metanet adjusted
        info["metanet_text_pos"] = metanet_text[0]  # [768] metanet adjusted
        info["metanet_text_neg"] = metanet_text[1]  # [768] metanet adjusted

        gt_mask = baseline["gt_mask"].detach()  # [518, 518]
        patch_features = baseline["patch_features"]  # [37, 37, 768]

        gt_mask = (
            torch.nn.functional.interpolate(
                gt_mask.unsqueeze(0).unsqueeze(0),
                (patch_features.shape[0], patch_features.shape[1]),
            )
            .flatten()
            .cpu()
            .numpy()
        )  # [1369=37x37,], values 0. or 1.
        info["gt_mask"] = gt_mask
        patch_features = (
            patch_features.view((-1, patch_features.shape[-1])).detach().cpu().numpy()
        )  # [1369, 768]
        info["patch_features"] = patch_features

        category_dict[class_name].append(info)

    # SECOND FOR LOOP [category]
    for category in categories:

        # THIRD FOR LOOP [tsne_type]
        for tsne_type in tsne_types:
            content_X = []
            content_Y = []

            if not just_visual:
                content_X.append(baseline_text_prior_pos)
                content_Y.append(0)
                content_X.append(baseline_text_prior_neg)
                content_Y.append(1)
                content_X.append(metanet_text_prior_pos)
                content_Y.append(2)
                content_X.append(metanet_text_prior_neg)
                content_Y.append(3)

            # read data
            all_samples_of_category = category_dict[category]

            if tsne_type == "patch":
                if auto_percentage:
                    total_normal_patches = 0
                    total_abnormal_patches = 0
                    for sample in all_samples_of_category:
                        gt_mask = sample["gt_mask"]  # [1369], values 0. or 1.
                        values, counts = np.unique(gt_mask, return_counts=True)
                        for value, count in zip(values, counts):
                            if int(value) == 1:
                                total_abnormal_patches += count
                            else:
                                total_normal_patches += count

                    normal_rate = total_normal_patches / (
                        total_normal_patches + total_abnormal_patches
                    )
                    abnormal_rate = 1 - normal_rate
                    TOTAL_SAMPLES = len(all_samples_of_category)
                    TOTAL_PATCH_TO_DRAW = 10000
                    TOTAL_PATCH_NORMAL = int(TOTAL_PATCH_TO_DRAW * normal_rate)
                    TOTAL_PATCH_ABNORMAL = TOTAL_PATCH_TO_DRAW - TOTAL_PATCH_NORMAL

                    PERCENT_PER_SAMPLE = TOTAL_PATCH_TO_DRAW / (TOTAL_SAMPLES * 1369)
                    NORMAL_PERCENT_PER_SAMPLE = PERCENT_PER_SAMPLE + 0.05
                    ABNORMAL_PERCENT_PER_SAMPLE = PERCENT_PER_SAMPLE + 0.1
                else:
                    TOTAL_PATCH_NORMAL = 7000
                    TOTAL_PATCH_ABNORMAL = 3000
                    NORMAL_PERCENT_PER_SAMPLE = 0.05
                    ABNORMAL_PERCENT_PER_SAMPLE = 0.15

                random.shuffle(all_samples_of_category)
                num_normal = 0
                num_abnormal = 0

            for sample in all_samples_of_category:
                if not just_visual:
                    # text features
                    metanet_text_pos = sample["metanet_text_pos"]
                    content_X.append(metanet_text_pos)
                    content_Y.append(FIXED_LABELS + 2)
                    metanet_text_neg = sample["metanet_text_neg"]
                    content_X.append(metanet_text_neg)
                    content_Y.append(FIXED_LABELS + 3)

                if tsne_type == "image":
                    # image-level
                    image_features = sample["image_features"]
                    gt_anomaly = sample["gt_anomaly"]
                    content_X.append(image_features)
                    content_Y.append(FIXED_LABELS + gt_anomaly)
                else:
                    # patch-level
                    patch_features = sample["patch_features"]  # [1369, 768]
                    gt_mask = sample["gt_mask"]  # [1369], values 0. or 1.

                    normal_patches = []
                    abnormal_patches = []

                    for gt, patch in zip(gt_mask, patch_features):
                        gt = int(gt)
                        if gt == 1:
                            abnormal_patches.append(patch)
                        else:
                            normal_patches.append(patch)

                    random.shuffle(normal_patches)
                    random.shuffle(abnormal_patches)

                    normal_patches = normal_patches[
                        : int(len(normal_patches) * NORMAL_PERCENT_PER_SAMPLE)
                    ]
                    abnormal_patches = abnormal_patches[
                        : int(len(abnormal_patches) * ABNORMAL_PERCENT_PER_SAMPLE)
                    ]

                    for patch in normal_patches:
                        content_X.append(patch)
                        content_Y.append(FIXED_LABELS)
                        num_normal += 1
                        if num_normal > TOTAL_PATCH_NORMAL:
                            break

                    for patch in abnormal_patches:
                        content_X.append(patch)
                        content_Y.append(FIXED_LABELS + 1)
                        num_abnormal += 1
                        if num_abnormal > TOTAL_PATCH_ABNORMAL:
                            break

                    # gt_patch_pairs = [
                    #     (gt, patch) for gt, patch in zip(gt_mask, patch_features)
                    # ]
                    # total_count = len(gt_patch_pairs)
                    # gt_patch_pairs = gt_patch_pairs[
                    #     : int(total_count * PERCENT_PER_SAMPLE)
                    # ]  # use a portion from each sample

                    # for gt, patch in gt_patch_pairs:
                    #     gt = int(gt)
                    #     if gt == 1:
                    #         if num_abnormal <= TOTAL_PATCH_ABNORMAL:
                    #             content_X.append(patch)
                    #             content_Y.append(FIXED_LABELS + gt)
                    #             num_abnormal += 1
                    #         else:
                    #             continue
                    #     else:
                    #         if num_normal <= TOTAL_PATCH_NORMAL:
                    #             content_X.append(patch)
                    #             content_Y.append(FIXED_LABELS + gt)
                    #             num_normal += 1
                    #         else:
                    #             continue

                    if (
                        num_abnormal > TOTAL_PATCH_ABNORMAL
                        and num_normal > TOTAL_PATCH_NORMAL
                    ):
                        break

            # MAPPING: label -> color/label/shape
            # category_to_label = {
            #     0: "bs+",
            #     1: "bs-",
            #     2: "mt+",
            #     3: "mt-",
            #     4: "[I]+" if tsne_type == "image" else "[P]+",
            #     5: "[I]-" if tsne_type == "image" else "[P]-",
            #     6: "+",
            #     7: "-",
            # }
            category_to_label = (
                [
                    (4, "[I]+" if tsne_type == "image" else "[P]+"),
                    (5, "[I]-" if tsne_type == "image" else "[P]-"),
                ]
                if just_visual
                else [
                    (4, "[I]+" if tsne_type == "image" else "[P]+"),
                    (5, "[I]-" if tsne_type == "image" else "[P]-"),
                    (6, "+"),
                    (7, "-"),
                    (0, "bs+"),
                    (1, "bs-"),
                    (2, "mt+"),
                    (3, "mt-"),
                ]
            )

            category_to_color = {
                0: "#0D0D04",
                1: "#0D0D04",
                2: "#F63110",
                3: "#F63110",
                4: "#6EC193",
                5: "#20603D",
                6: "#F6CB71",
                7: "#D1910B",
            }

            content_X = np.array(content_X)
            content_Y = np.array(content_Y)

            pca_fitter = PCA(n_components=50)
            content_X = pca_fitter.fit_transform(content_X)

            tsne = TSNE(
                n_components=2,
                init="pca",
            )
            content_X = tsne.fit_transform(content_X)

            fig, ax = plt.subplots()
            fig.set_figheight(9)
            fig.set_figwidth(12)

            ax.set(
                title=(
                    f"IMAGE-LEVEL [Dataset] {dataset} [Category] {category}"
                    if tsne_type == "image"
                    else f"PIXEL-LEVEL [Dataset] {dataset} [Category] {category}"
                )
            )

            for category_id, label in category_to_label:
                mask = content_Y == category_id

                marker_choice = None
                if category_id in [0, 2, 6]:  # text, +
                    marker_choice = 6
                elif category_id in [1, 3, 7]:  # text, -
                    marker_choice = 7
                elif category_id == 4:  # image/patch, +
                    marker_choice = "+"
                elif category_id == 5:  # image/patch, -
                    marker_choice = "*"

                scale = None
                if category_id in [0, 1, 2, 3]:
                    scale = 128

                ax.scatter(
                    content_X[mask, 0],
                    content_X[mask, 1],
                    label=label,
                    c=category_to_color[category_id],
                    marker=marker_choice,
                    s=scale,
                )

            ax.legend()
            file_name = (
                f"Image_level_{dataset}_{category}.png"
                if tsne_type == "image"
                else f"Pixel_{dataset}_{category}.png"
            )
            if just_visual:
                file_name = "Visual_Only_" + file_name
            plt.savefig(file_name)
