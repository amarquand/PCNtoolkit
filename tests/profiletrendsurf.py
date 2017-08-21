# NOTE: must be run with kernprof (otherwise the inmports get screwed up)

#import nispat
import os
import sys

sys.path.append('/Users/andre/sfw/nispat/nispat')
from trendsurf import estimate
from line_profiler import LineProfiler
from bayesreg import BLR

# with test covariates
wdir = '/Users/andre/data/ismael/trendsurf_python'
maskfile = os.path.join(wdir, 'mask.nii.gz')
datafile = os.path.join(wdir, 'spect_data2_first5.nii.gz')
basis = os.path.join(wdir, 'bfs/icp_basis_s8.nii.gz')

lp = LineProfiler(BLR.loglik)
lp.add_function(BLR.dloglik)
lp.add_function(BLR.post)
lp.enable()

estimate(datafile, maskfile, basis)

lp.disable()
lp.print_stats()

# to profile, can also put the following code in trendsurf.py
# lp = LineProfiler(BLR.loglik)
# lp = LineProfiler(BLR.dloglik)
# lp.add_function(BLR.post)
# lp.enable()
# hyp[i, :] = breg.estimate(hyp0, Phi, Yz[:, i])
# lp.disable()
# lp.print_stats()