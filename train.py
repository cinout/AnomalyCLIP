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
import numpy as np
import os
import random
from utils import get_transform
from torch.utils.tensorboard import SummaryWriter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
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

    train_data = Dataset(
        root=args.train_data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )

    ##########################################################################################
    prompt_learner = AnomalyCLIP_PromptLearner(
        model.to("cpu"), AnomalyCLIP_parameters, args=args
    )
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    ##########################################################################################
    optimizer = torch.optim.Adam(
        list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999)
    )

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    model.eval()
    prompt_learner.train()
    writer = SummaryWriter(
        log_dir=args.save_path
    )  # Writer will output to ./runs/ directory by default. You can change log_dir in here

    global_step = 0
    for epoch in tqdm(range(args.epoch)):
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image = items["img"].to(device)
            label = items["anomaly"]

            gt = items["img_mask"].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                # Apply DPAM to the layer from 6 to 24
                # DPAM_layer represents the number of layer refined by DPAM from top to bottom
                # DPAM_layer = 1, no DPAM is used
                # DPAM_layer = 20 as default

                image_features, patch_features = model.encode_image(
                    image,
                    args.features_list,
                    DPAM_layer=20,
                    maple=args.maple,
                    compound_deeper_prompts=prompt_learner.visual_deep_prompts,
                )
                if args.maple:
                    patch_features = [
                        feature[:, : -args.t_n_ctx, :] for feature in patch_features
                    ]

                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )  # [8, 768]

            ####################################

            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(
                image_features=image_features,
                patch_features=patch_features,
                cls_id=None,
            )  # prompts: [2 or 2*bs, 77, 768]; tokenized_prompts: [2 or 2*bs, 77]; compound_prompts_text: [4, 768] * 8

            text_features = model.encode_text_learn(
                prompts, tokenized_prompts, compound_prompts_text
            ).float()  # [2 or 2*bs, 768]

            text_features = torch.stack(
                torch.chunk(text_features, dim=0, chunks=2), dim=1
            )  # [1 or bs, 2, 768]

            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )  # [1 or bs, 2, 768]

            # Apply DPAM surgery
            text_probs = image_features.unsqueeze(1) @ text_features.permute(
                0, 2, 1
            )  # [8, 1, 2], the same text_features are applied to 8 images

            text_probs = text_probs[:, 0, ...] / 0.07  # [8, 2]

            image_loss = F.cross_entropy(
                text_probs, label.long().to(device)
            )  #  no shape
            # image_loss = F.cross_entropy(
            #     text_probs.squeeze(), label.long().to(device)
            # )  #  no shape
            image_loss_list.append(image_loss.item())

            #########################################################################
            similarity_map_list = []

            for idx, patch_feature in enumerate(patch_features):  # 4*[bs, 1370, 768]
                if idx >= args.feature_map_layer[0]:  # >=0
                    patch_feature = patch_feature / patch_feature.norm(
                        dim=-1, keepdim=True
                    )  # [bs, 1370, 768]

                    if args.visual_ae:
                        patch_feature = prompt_learner.process_patch_features(
                            patch_feature, idx
                        )

                    # calculate patch-level similarity
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(
                        patch_feature, text_features
                    )  # [bs, 1370, 2]

                    # upsample anomaly map
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(
                        similarity[:, 1:, :], args.image_size
                    ).permute(
                        0, 3, 1, 2
                    )  # [bs, 2, 518, 518]

                    similarity_map_list.append(similarity_map)

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1 - gt)

            optimizer.zero_grad()
            total_loss = loss + image_loss
            total_loss.backward()
            writer.add_scalar("Loss/train", total_loss.item(), global_step)
            global_step += 1
            optimizer.step()
            loss_list.append(loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info(
                "epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}".format(
                    epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)
                )
            )

        # save model
        # if (epoch + 1) % args.save_freq == 0:
        if epoch + 1 == args.epoch:
            ckp_path = os.path.join(args.save_path, "epoch_" + str(args.epoch) + ".pth")
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)

    writer.flush()  # Call flush() method to make sure that all pending events have been written to disk
    writer.close()  # if you do not need the summary writer anymore, call close() method.


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument(
        "--train_data_path", type=str, default="./data/visa", help="train dataset path"
    )
    parser.add_argument(
        "--save_path", type=str, default="./checkpoint", help="path to save results"
    )

    parser.add_argument(
        "--dataset", type=str, default="mvtec", help="train dataset name"
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=9,
        help="learnabel_text_embedding_depth, The learnable token embeddings are attached to the first 9 layers of the text encoder",
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=12,
        help="Prompt_length, length of learnable word embeddings E",
    )
    parser.add_argument(
        "--t_n_ctx",
        type=int,
        default=4,
        help="learnabel_text_embedding_length, length of learnable token embeddings in each layer",
    )
    parser.add_argument(
        "--feature_map_layer",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="zero shot",
    )
    parser.add_argument(
        "--features_list",
        type=int,
        nargs="+",
        default=[6, 12, 18, 24],
        help="features used",
    )

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
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
    parser.add_argument(
        "--visual_ae",
        action="store_true",
        help="use AE after the four selected stages of visual encoder",
    )
    args = parser.parse_args()
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    setup_seed(args.seed)
    train(args)
