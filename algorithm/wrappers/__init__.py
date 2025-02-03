import torch
# constraint-based algorithms
# PC based algorithms
from .pc import PC
from .fci import FCI
from .cdnod import CDNOD
## Markov Blanket based algorithms
from .inter_iamb import InterIAMB
from .bamb import BAMB
from .hiton_mb import HITONMB
from .iambnpc import IAMBnPC
from .mbor import MBOR

# score-based algorithms
## Greedy search based algorithms
from .ges import GES
from .fges import FGES
from .x_ges import XGES
## permutation based algorithms
from .grasp import GRaSP

# functional-model based algorithms
from .direct_lingam import DirectLiNGAM
from .ica_lingam import ICALiNGAM

# hybrid algorithms
from .hybrid import Hybrid


constraint_based_algorithms = ['PC', 'FCI', 'CDNOD', 'InterIAMB', 'BAMB', 'HitonMB', 'IAMBnPC', 'MBOR']
score_based_algorithms = ['GES', 'FGES', 'XGES']
functional_model_based_algorithms = ['DirectLiNGAM', 'ICALiNGAM']
permutation_based_algorithms = ['GRaSP']
hybrid_algorithms = ['Hybrid']


if torch.cuda.is_available():
    # If GPU is available, use gpu-accelerated algorithms
    from .accelerated_lingam import AcceleratedDirectLiNGAM 
    from .accelerated_pc import AcceleratedPC
    accelerated_algorithms = ['AcceleratedDirectLiNGAM', 'AcceleratedPC']
    # If GPU is available, use continous optimization based algorithms
    from .notears_linear import NOTEARSLinear
    from .notears_mlp import NOTEARSNonlinear
    from .corl import CORL
    from .calm import CALM
    continuous_optimization_based_algorithms = ['NOTEARSLinear', 'NOTEARSNonlinear', 'CORL', 'CALM']

    __all__ = constraint_based_algorithms + score_based_algorithms + functional_model_based_algorithms + permutation_based_algorithms + hybrid_algorithms + accelerated_algorithms + continuous_optimization_based_algorithms
else:
    __all__ = constraint_based_algorithms + score_based_algorithms + functional_model_based_algorithms + permutation_based_algorithms + hybrid_algorithms