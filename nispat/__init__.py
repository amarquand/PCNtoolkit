import bayesreg
import trendsurf
import gp
from .bayesreg import BLR
from .gp import GPR, covSqExp
from .fileio import load_nifti, save_nifti, create_mask