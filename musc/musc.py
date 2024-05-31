import os
import numpy as np
import torch
import torch.nn.functional as F
from musc._MSM import MSM
from musc._LNAMD import LNAMD
import math
from PIL import Image
from sklearn.cluster import KMeans


class MuSc:
    def __init__(
        self,
        args,
    ):
        self.device = args.device
        self.image_size = args.image_size
        self.features_list = args.features_list
        self.r_list = args.r_list
        self.normal_percent = 0.75  # TODO: does it matter?
        self.image_size = args.image_size
        self.show_musc_visual = args.show_musc_visual
        self.seed = args.seed
        self.save_path = args.save_path
        self.musc_cluster = args.musc_cluster

    def process_features(
        self, patch_features, img_path, cls_name, take_first_only=False
    ):
        # input patch_features.shape: 4*[bs, 1370, 768]

        feature_dim = patch_features[0].shape[-1]
        anomaly_maps_r = torch.tensor([]).double()

        for r in self.r_list:
            # LNAMD
            LNAMD_r = LNAMD(
                device=self.device,
                r=r,
                feature_dim=feature_dim,
                # feature_layer=self.features_list,
            )
            Z_layers = (
                {}
            )  # keys: [0,1,2,3], values: each is a list of features of size (B, L, C)

            patch_tokens = [
                p.to(self.device) for p in patch_features
            ]  # 4* [bs, 1370, 1024]

            # with torch.no_grad(), torch.cuda.amp.autocast():
            with torch.no_grad():
                features = LNAMD_r._embed(patch_tokens)  # (B=4, L=1369, layer=4, C)

                features /= features.norm(dim=-1, keepdim=True)
                for l in range(len(self.features_list)):
                    # save the aggregated features
                    if str(l) not in Z_layers.keys():
                        Z_layers[str(l)] = []
                    Z_layers[str(l)].append(features[:, :, l, :])

            # MSM: compare features
            anomaly_maps_l = torch.tensor([]).double()

            for l in Z_layers.keys():
                # different layers
                Z = torch.cat(Z_layers[l], dim=0).to(
                    self.device
                )  # (N=#test, L=1369, C=1024)

                anomaly_maps_msm = MSM(
                    Z=Z, device=self.device, topmin_min=0, topmin_max=0.3
                )  # (N=#test, L=1369)
                anomaly_maps_l = torch.cat(
                    (anomaly_maps_l, anomaly_maps_msm.unsqueeze(0).cpu()), dim=0
                )
                # torch.cuda.empty_cache()
            anomaly_maps_l = torch.mean(anomaly_maps_l, 0)  # [N, L=1369]

            anomaly_maps_r = torch.cat(
                (anomaly_maps_r, anomaly_maps_l.unsqueeze(0)), dim=0
            )

        anomaly_map = torch.mean(anomaly_maps_r, 0).to(self.device)  # [N=#test, L=1369]
        del anomaly_maps_r
        # torch.cuda.empty_cache()

        # only keep the first, as the rest are ref images from other batches
        if take_first_only:
            anomaly_map = anomaly_map[0].unsqueeze(0)
            patch_features = [f[0].unsqueeze(0) for f in patch_features]

        B, L = anomaly_map.shape
        H = int(np.sqrt(L))
        patch_features = [p[:, 1:, :] for p in patch_features]
        patch_features = torch.stack(patch_features, dim=0)
        patch_features = torch.mean(patch_features, dim=0)  # [bs, L, C]

        # TODO: how to better choose the normal patches to reveal more diverse patterns in the image?
        _, indices = torch.topk(
            anomaly_map, dim=1, k=math.floor(L * self.normal_percent), largest=False
        )  # indices: [bs, k]

        # the following code is for visualization purpose
        if self.show_musc_visual and self.seed == 10:
            save_dir = os.path.join(self.save_path, "musc", cls_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for img_idx, path in enumerate(img_path):
                image = Image.open(path).convert("RGBA")
                image = image.resize((H, H))

                pred_normal_pixels = indices[img_idx]
                pred_normal_pixels = [
                    (math.floor(v / H), v % H) for v in pred_normal_pixels
                ]  # 2D
                seg = np.full((H, H), False)
                for tup in pred_normal_pixels:
                    seg[tup] = True
                mask = np.zeros(
                    (H, H, 4),
                    dtype=np.uint8,
                )
                mask[seg] = [
                    238,
                    79,
                    38,
                    120,
                ]
                mask = Image.fromarray(mask)
                image.paste(mask, (0, 0), mask)

                file_path_split = path.split("/")
                anomaly_type = file_path_split[-2]
                file_name = file_path_split[-1]
                file_name = file_name.split(".")[0]

                image.save(
                    os.path.join(save_dir, f"{anomaly_type}_{file_name}.png"),
                    "PNG",
                )

        normal_features = []

        for img_idx, value in enumerate(indices):
            # for each image in the batch
            normal_patch_features = patch_features[img_idx, value]  # [k, C]

            if self.musc_cluster:
                # find the cluster centers
                normal_patch_features = normal_patch_features.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=8, n_init="auto").fit(normal_patch_features)
                cluster_centers = kmeans.cluster_centers_
                cluster_centers = torch.tensor(
                    cluster_centers, device=self.device
                )  # [#cluster=8, C]
                normal_patch_features = torch.mean(cluster_centers, dim=0)  # [C, ]
            else:
                normal_patch_features = torch.mean(
                    normal_patch_features, dim=0
                )  # [C, ]

            normal_features.append(normal_patch_features)

        normal_features = torch.stack(normal_features, dim=0)  # [bs, C]

        return normal_features
