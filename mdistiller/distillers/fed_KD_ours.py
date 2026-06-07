import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from ._common import kd_loss, cc_loss, bc_loss, mixup_data, mixup_data_conf


class fed_KD_ours(Distiller):
    def __init__(self, student, teacher, cfg, clients_dataset_len):
        super(fed_KD_ours, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.clients_dataset_len = clients_dataset_len

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        for i in self.teacher:
            self.teacher[i].eval()
        return self

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        client_num = len(self.teacher)
        total_len = 0.0
        for i in range(client_num):
            total_len += self.clients_dataset_len[i]
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak = 0
            logits_teacher_strong = 0
            for i in range(client_num):
                logits_temp_weak, _ = self.teacher[i](image_weak)
                logits_temp_strong, _ = self.teacher[i](image_strong)
                logits_teacher_weak += (logits_temp_weak * self.clients_dataset_len[i])
                logits_teacher_strong += (logits_temp_strong * self.clients_dataset_len[i])
            logits_teacher_weak = logits_teacher_weak / total_len
            logits_teacher_strong = logits_teacher_strong / total_len

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

        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))

        loss_kd_weak = self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * mask).mean())

        loss_kd_strong = self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        )

        loss_cc_weak = self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * class_conf_mask).mean())
        loss_cc_strong = self.kd_loss_weight * cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
        ) + self.kd_loss_weight * cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
        ) + self.kd_loss_weight * cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
        ) + self.kd_loss_weight * cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) + self.kd_loss_weight * cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        )
        loss_bc_weak = self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * mask).mean())
        loss_bc_strong = self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            6.0,
        ) * mask).mean())
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd_weak + loss_kd_strong,
            "loss_cc": loss_cc_weak,
            "loss_bc": loss_bc_weak
        }
        return logits_student_weak, losses_dict
