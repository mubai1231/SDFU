from ._base import Vanilla
from .KD_ours import KD_ours
from .fed_KD_ours import fed_KD_ours
from .KD_ours_plus import KD_ours_plus
from .KD_unlearning import KD_unlearning

distiller_dict = {
    "NONE": Vanilla,
    "KD_unlearning": KD_unlearning,
    "KD_ours_plus": KD_ours_plus,
    "fed_KD_ours": fed_KD_ours,
    "KD_ours": KD_ours,
}
