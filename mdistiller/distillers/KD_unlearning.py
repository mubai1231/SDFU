import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from ._common import cc_loss_pearson


class KD_unlearning(Distiller):
    def __init__(self, student, teacher, cfg):
        super(KD_unlearning, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        batch_size, class_num = logits_student_strong.shape

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

        loss_ce = self.ce_loss_weight * (
                    F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))

        loss_cc_weak = self.kd_loss_weight * ((cc_loss_pearson(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * class_conf_mask).mean())

        losses_dict = {
            "loss_cc": 1/loss_cc_weak,
        }
        return logits_student_weak, losses_dict
