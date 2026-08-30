import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from .RULE_Loss import CustomMultiLossLayer, RobustEvidentialAlignmentLoss
from .RULE_Encoder_ICEWS import RULE_Encoder


class RULE(nn.Module):
    """RULE ablation: 删除 uncertainty / consensus / DRF，保留 evidential alignment loss。"""

    def __init__(self, kgs, args, train_set, test_set, logger):
        super().__init__()

        self.kgs = kgs
        self.args = args
        self.logger = logger
        self.test_ill = test_set
        self.train_ill = train_set

        self.missing_img = self.kgs["missing_img"]
        self.img_features = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()

        self.input_idx = kgs["input_idx"].cuda()
        self.adj = kgs["adj"].cuda()

        self.rel_features = None
        self.att_features = None
        self.name_features = None
        self.char_features = None

        if kgs["name_features"] is not None:
            self.name_features = F.normalize(torch.FloatTensor(kgs["name_features"])).cuda()

        if kgs["char_features"] is not None:
            self.char_features = kgs["char_features"].cuda()

        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        else:
            img_dim = kgs["images_list"].shape[1]

        char_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100

        self.modal_keys = ["image", "structure", "relation", "attribute", "name", "char"]

        # 仅作为 availability mask；不再进行动态可靠性重加权
        self.loss_mask = {key: torch.ones(kgs["ent_num"], dtype=torch.float).cuda() for key in self.modal_keys}
        for entity_id in self.missing_img:
            self.loss_mask["image"][entity_id] = 0

        self.multimodal_encoder = RULE_Encoder(args=self.args, ent_num=kgs["ent_num"], img_feature_dim=img_dim, char_feature_dim=char_dim)

        self.modality_num = 2 + (1 if self.args.use_surface else 0)
        self.multi_loss_layer = CustomMultiLossLayer(self.modality_num)
        self.robust_loss = RobustEvidentialAlignmentLoss(tau=self.args.tau, top_k=self.args.topk, lambda2=self.args.lambda2)

    def emb_generat(self):
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb = self.multimodal_encoder(
            self.input_idx,
            self.adj,
            self.loss_mask,
            self.img_features,
            self.rel_features,
            self.att_features,
            self.name_features,
            self.char_features
        )
        return {"structure": gph_emb, "image": img_emb, "relation": rel_emb, "attribute": att_emb, "name": name_emb, "char_name": char_emb}

    def _normalized_emb_dict(self):
        raw = self.emb_generat()
        emb_dict = {
            "structure": raw.get("structure"),
            "image": raw.get("image"),
            "relation": raw.get("relation"),
            "attribute": raw.get("attribute"),
            "name": raw.get("name"),
            "char": raw.get("char_name"),
        }
        emb_dict = {key: F.normalize(value, dim=1) for key, value in emb_dict.items() if value is not None}
        return raw, emb_dict

    def joint_emb_generat(self):
        raw, emb_dict = self._normalized_emb_dict()
        joint_emb = self.multimodal_encoder.fusion(emb_dict, self.loss_mask)
        return {
            "structure": raw.get("structure"),
            "image": raw.get("image"),
            "relation": raw.get("relation"),
            "attribute": raw.get("attribute"),
            "name": raw.get("name"),
            "char": raw.get("char_name"),
            "joint": joint_emb,
        }

    def crossgraph_attribute_alignment(self, emb_dict, train_ill):
        losses = {}
        valid_losses = []

        for key, emb in emb_dict.items():
            mask = self.loss_mask[key] if key == "image" else None
            loss = self.robust_loss(emb, train_ill, mask=mask)
            losses[key] = loss
            valid_losses.append(loss)

        total_loss = self.multi_loss_layer(valid_losses)
        return total_loss, losses

    def forward(self, batch, epoch=None, batch_no=None):
        self.loss_dic = {}

        raw, emb_dict = self._normalized_emb_dict()

        loss_attribute, modality_losses = self.crossgraph_attribute_alignment(emb_dict, batch)

        joint_emb = self.multimodal_encoder.fusion(emb_dict, self.loss_mask)
        loss_entity = self.robust_loss(joint_emb, batch)

        loss_all = loss_attribute + loss_entity

        for key, value in modality_losses.items():
            self.loss_dic[f"{key}_loss"] = value
        self.loss_dic.update({"loss_all": loss_all, "loss_entity": loss_entity, "loss_attribute": loss_attribute})

        output = {
            "loss_dic": self.loss_dic,
            "loss_all": loss_all,
            "gph_emb": raw.get("structure"),
            "img_emb": raw.get("image"),
            "rel_emb": raw.get("relation"),
            "att_emb": raw.get("attribute"),
            "name_emb": raw.get("name"),
            "char_emb": raw.get("char_name"),
            "joint_emb": joint_emb,
        }

        return loss_all, output
