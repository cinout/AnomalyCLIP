import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm

import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from visualization import visualizer

from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


def generate_text_features(
    prompt_learner, model, image_features=None, patch_features=None
):
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(
        image_features=image_features, patch_features=patch_features
    )
    text_features = model.encode_text_learn(
        prompts, tokenized_prompts, compound_prompts_text
    ).float()  # [2 or 2*bs, 768]
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features  # [1 or bs, 2, 768]


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
        "maple": args.maple,
        # "maple_length": 2,
    }

    model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters
    )
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(
        root=args.data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False
    )
    obj_list = test_data.obj_list

    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]["gt_sp"] = []  # gt: ground-truth, sp: sample-level
        results[obj]["pr_sp"] = []  # pr: predicted, sp: sample-level
        results[obj]["imgs_masks"] = []  # ground-truth, pixel-level
        results[obj]["anomaly_maps"] = []  # predicted, pixel-level
        metrics[obj] = {}
        metrics[obj]["pixel-auroc"] = 0
        metrics[obj]["pixel-aupro"] = 0
        metrics[obj]["image-auroc"] = 0
        metrics[obj]["image-ap"] = 0

    prompt_learner = AnomalyCLIP_PromptLearner(
        model.to("cpu"), AnomalyCLIP_parameters, args=args
    )
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    prompt_learner.eval()
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    if not args.meta_net or args.debug_mode:
        text_features = generate_text_features(prompt_learner, model)

    model.to(device)

    if args.debug_mode:
        stored_features = dict()
        stored_features["prior_text_feature"] = text_features
        all_results = []

    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items["img"].to(device)
        cls_name = items["cls_name"]
        cls_id = items["cls_id"]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]["imgs_masks"].append(gt_mask)  # px
        results[cls_name[0]]["gt_sp"].extend(items["anomaly"].detach().cpu())

        if args.debug_mode:
            content = dict()
            if not args.meta_net:
                content["gt_anomaly"] = items["anomaly"][0]
                content["class_name"] = cls_name[0]

        with torch.no_grad():
            image_features, patch_features = model.encode_image(
                image,
                features_list,
                DPAM_layer=20,
                maple=args.maple,
                compound_deeper_prompts=prompt_learner.visual_deep_prompts,
            )
            if args.maple:
                patch_features = [
                    feature[:, : -args.t_n_ctx, :] for feature in patch_features
                ]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if args.meta_net:
                text_features = generate_text_features(
                    prompt_learner, model, image_features, patch_features
                )
                if args.debug_mode:
                    content["text_features"] = text_features[0]  # [2, 768]

            if args.debug_mode:
                if not args.meta_net:
                    content["image_features"] = image_features[0]  # [768]
                    content["gt_mask"] = gt_mask[0, 0]  # [518, 518]

            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            results[cls_name[0]]["pr_sp"].extend(text_probs.detach().cpu())

            anomaly_map_list = []

            if args.debug_mode:
                patch_features_norm = []

            for patch_idx, patch_feature in enumerate(patch_features):
                if patch_idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(
                        dim=-1, keepdim=True
                    )

                    if args.debug_mode:
                        patch_features_norm.append(patch_feature[0, 1:, :])

                    similarity, _ = AnomalyCLIP_lib.compute_similarity(
                        patch_feature, text_features
                    )
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(
                        similarity[:, 1:, :], args.image_size
                    )  # [1, 518, 518, 2]

                    anomaly_map = (
                        similarity_map[..., 1] + 1 - similarity_map[..., 0]
                    ) / 2.0
                    anomaly_map_list.append(anomaly_map)

            if args.debug_mode:
                patch_features_norm = torch.stack(patch_features_norm, dim=0)
                patch_features_norm = torch.mean(patch_features_norm, dim=0)
                side = int(patch_features_norm.shape[0] ** 0.5)
                patch_features_norm = torch.reshape(
                    patch_features_norm, (side, side, -1)
                )
                if not args.meta_net:
                    content["patch_features"] = patch_features_norm  # [37, 37, 768]

                all_results.append(content)

                continue

            anomaly_map = torch.stack(anomaly_map_list)  # [4, 1, 518, 518]
            anomaly_map = anomaly_map.sum(dim=0)  # [1, 518, 518]
            anomaly_map = torch.stack(
                [
                    torch.from_numpy(gaussian_filter(i, sigma=args.sigma))
                    for i in anomaly_map.detach().cpu()
                ],
                dim=0,
            )  # [1, 518, 518]

            results[cls_name[0]]["anomaly_maps"].append(anomaly_map)

            if args.seed == 10:
                visualizer(
                    items["img_path"],
                    anomaly_map.detach().cpu().numpy(),
                    args.image_size,
                    args.save_path,
                    cls_name,
                    gt_mask,
                )

    if args.debug_mode:
        stored_features["individual"] = all_results
        condition = "baseline"
        if args.meta_net and args.metanet_patch_and_global:
            condition = "metanet_pag"
        elif args.meta_net and args.metanet_patch_only:
            condition = "metanet_ponly"

        with open(f"{args.dataset}_{condition}.t", "wb") as f:
            torch.save(stored_features, f)
        return

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    for obj in obj_list:
        table = []
        table.append(obj)
        results[obj]["imgs_masks"] = torch.cat(results[obj]["imgs_masks"])
        results[obj]["anomaly_maps"] = (
            torch.cat(results[obj]["anomaly_maps"]).detach().cpu().numpy()
        )
        if args.metrics == "image-level":
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        elif args.metrics == "pixel-level":
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == "image-pixel-level":
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        table_ls.append(table)

    if args.metrics == "image-level":
        # logger
        table_ls.append(
            [
                "mean",
                str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                str(np.round(np.mean(image_ap_list) * 100, decimals=1)),
            ]
        )
        results = tabulate(
            table_ls, headers=["objects", "image_auroc", "image_ap"], tablefmt="pipe"
        )
    elif args.metrics == "pixel-level":
        # logger
        table_ls.append(
            [
                "mean",
                str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
            ]
        )
        results = tabulate(
            table_ls, headers=["objects", "pixel_auroc", "pixel_aupro"], tablefmt="pipe"
        )
    elif args.metrics == "image-pixel-level":
        # logger
        table_ls.append(
            [
                "mean",
                str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
                str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                str(np.round(np.mean(image_ap_list) * 100, decimals=1)),
            ]
        )
        results = tabulate(
            table_ls,
            headers=[
                "objects",
                "pixel_auroc",
                "pixel_aupro",
                "image_auroc",
                "image_ap",
            ],
            tablefmt="pipe",
        )
    logger.info("\n%s", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths
    parser.add_argument(
        "--data_path", type=str, default="./data/visa", help="path to test dataset"
    )
    parser.add_argument(
        "--save_path", type=str, default="./results/", help="path to save results"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoint/",
        help="path to checkpoint",
    )
    # model
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument(
        "--features_list",
        type=int,
        nargs="+",
        default=[6, 12, 18, 24],
        help="features used",
    )
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument(
        "--feature_map_layer",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="zero shot",
    )
    parser.add_argument("--metrics", type=str, default="image-pixel-level")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--meta_net", action="store_true")
    parser.add_argument("--maple", action="store_true")
    parser.add_argument(
        "--metanet_patch_and_global",
        action="store_true",
        help="use patch+image features in meta_net",
    )
    parser.add_argument(
        "--metanet_patch_only",
        action="store_true",
        help="use patch features only in meta_net",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="",
    )

    args = parser.parse_args()
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    setup_seed(args.seed)
    test(args)
