from .trainer import BaseTrainer, CRDTrainer, AugTrainer,FebAugTrainer, UnlearningTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "ours": AugTrainer,
    "fed_ours": FebAugTrainer,
    "unlearning": UnlearningTrainer
}
