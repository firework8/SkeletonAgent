import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import linalg as LA
from abc import ABCMeta, abstractmethod

from ...core import top_k_accuracy
from ..builder import build_loss

import sys
from mmcv.runner import get_dist_info
from .agent import Interaction


def get_label_gt(labels):
    num = len(labels)
    gt = np.zeros(shape=(num, num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()
    return logits_per_x1, logits_per_x2

class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_metric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(reduction='sum')):
        super().__init__()
        self.error_metric = error_metric

    def forward(self, prediction, label):
        C = prediction.shape[1]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) / C
        return loss


"""
******************************************************************************************
******************************************************************************************
******************************************************************************************
"""


class BaseHead(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 joint_cfg,
                 num_classes,
                 in_channels,
                 work_dir,
                 interval_epoch,
                 weight_1,
                 weight_2,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0):
        super().__init__()
        self.joint_cfg = joint_cfg
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        
        self.cls_loss = build_loss(loss_cls)
        self.con_loss = build_loss(loss_cls)
        self.align_loss = KLLoss().cuda()
        
        self.Interaction = Interaction(self.num_classes, work_dir)
        
        self.change_state = False
        self.current_epoch = 1
        self.interval_epoch = interval_epoch
        self.class_list = []
        self.statistics = []
        for i in range(num_classes):
            count_dic = {}
            for j in range(num_classes):
                count_dic[j] = 0
            self.statistics.append(count_dic)
        

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def update_statistics(self, lbl, logit):
        pred = logit.max(1)[1]
        batch = len(lbl)
        for i in range(batch):
            if pred[i] != lbl[i]:
                class_1 = pred[i].item()
                self.statistics[lbl[i]][class_1] += 1
            for j in range(batch):
                if pred[j] == lbl[i] and lbl[j] != lbl[i]:
                    class_2 = lbl[j].item()
                    self.statistics[lbl[i]][class_2] += 1
        return True
    
    def update_class_list(self):
        for i in range(self.num_classes):
            counter = self.statistics[i]
            sorted_num = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            sorted_list = []
            if len(sorted_num) != 0:
                for sorted_i in range(len(sorted_num)):
                    sorted_list.append(sorted_num[sorted_i][0])
                if len(sorted_list) > 10:
                    sorted_list = sorted_list[:10]
                sorted_list = sorted(sorted_list)
            self.class_list.append(sorted_list)
        return True
    
    def check_state(self, llm_model):
        if self.change_state == True:
            rank, _ = get_dist_info()
            if rank != 0:
                return True
            check_list = [x for x in range(1, 150) if x % self.interval_epoch == 0]
            bool_class_list = self.update_class_list()
            
            if self.current_epoch in check_list:
                bool_update_text = self.Interaction.update_description(llm_model, self.class_list)
            
            self.change_state = False
            self.current_epoch += 1
            self.statistics = [{j: 0 for j in range(self.num_classes)} for _ in range(self.num_classes)]
            self.class_list.clear()
        return True
    
    def get_text(self, llm_model, label):
        text_list = self.Interaction.get_description(llm_model)
        texts = []
        for i in range(len(label)):
            texts.append(text_list[label[i]])
        texts = torch.cat(texts).cuda()
        return texts
    
    def loss(self, cls_score, cls_mix, x_1, clip_model, llm_model, x_2, model_text_dict, logit_scale, label, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``label``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            label (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            label = label.unsqueeze(0)

        if not self.multi_class and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            label = ((1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes)

        if self.change_state == False:
            self.change_state = True
        bool_statistics = self.update_statistics(label.detach(), cls_score.detach())
        
        """
        ************
        *** Loss ***
        ************
        """
        # Cls Loss
        loss_cls_0 = self.cls_loss(cls_score, label, **kwargs) + cls_mix
        
        # Constraint Loss
        loss_cls_1 = self.con_loss(x_1, label, **kwargs)
        
        # Alignment Calculation
        text_embedding = model_text_dict[clip_model](self.get_text(llm_model, label)).float()
        logits_per_skeleton, logits_per_text = create_logits(x_2, text_embedding, logit_scale[:, 0].mean())
        ground_truth = torch.tensor(get_label_gt(label), dtype=x_2.dtype, device='cuda')
        loss_skeleton = self.align_loss(logits_per_skeleton, ground_truth)
        loss_text = self.align_loss(logits_per_text, ground_truth)
        # Alignment Loss
        loss_cls_2 = (loss_skeleton + loss_text) / 2
        
        # Overall Loss
        loss_cls = loss_cls_0 + self.weight_1 * loss_cls_1 + self.weight_2 * loss_cls_2
        """
        ************
        """
        
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses


"""
******************************************************************************************
******************************************************************************************
******************************************************************************************
"""

"""
This version is a semi-finished release that does not require API setting. 
It achieves relatively satisfactory performance and can be used for tuning.
"""

class BaseHead_Semi(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 joint_cfg,
                 num_classes,
                 in_channels,
                 work_dir,
                 interval_epoch,
                 weight_1,
                 weight_2,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0):
        super().__init__()
        self.joint_cfg = joint_cfg
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        
        self.cls_loss = build_loss(loss_cls)
        self.con_loss = build_loss(loss_cls)
        self.align_loss = KLLoss().cuda()
        
        self.Interaction = Interaction(self.num_classes, work_dir)

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""
    
    def get_text(self, llm_model, label):
        text_list = self.Interaction.get_description(llm_model)
        texts = []
        for i in range(len(label)):
            texts.append(text_list[label[i]])
        texts = torch.cat(texts).cuda()
        return texts
    
    def loss(self, cls_score, cls_mix, x_1, clip_model, llm_model, x_2, model_text_dict, logit_scale, label, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``label``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            label (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            label = label.unsqueeze(0)

        if not self.multi_class and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            label = ((1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes)

        """
        ************
        *** Loss ***
        ************
        """
        # Cls Loss
        loss_cls_0 = self.cls_loss(cls_score, label, **kwargs) + cls_mix
        
        # Constraint Loss
        loss_cls_1 = self.con_loss(x_1, label, **kwargs)
        
        # Alignment Calculation
        text_embedding = model_text_dict[clip_model](self.get_text(llm_model, label)).float()
        logits_per_skeleton, logits_per_text = create_logits(x_2, text_embedding, logit_scale[:, 0].mean())
        ground_truth = torch.tensor(get_label_gt(label), dtype=x_2.dtype, device='cuda')
        loss_skeleton = self.align_loss(logits_per_skeleton, ground_truth)
        loss_text = self.align_loss(logits_per_text, ground_truth)
        # Alignment Loss
        loss_cls_2 = (loss_skeleton + loss_text) / 2
        
        # Overall Loss
        loss_cls = loss_cls_0 + self.weight_1 * loss_cls_1 + self.weight_2 * loss_cls_2
        """
        ************
        """
        
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses
    