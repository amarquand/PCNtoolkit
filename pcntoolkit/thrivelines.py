from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.dataio.norm_data import NormData
import numpy as np

def thrivelines(model, data):
    pass

# """
# def sparse_correlation_matrices(subjects, age, responses, dc_max):
#     Create groups{age:df} dict
#     For ages i,j:
#         joint = inner merge groups[i] and groups[j] on subject
#         for column of responses:
#             correlation_matrix[responses,i,j] = corr(joint[responses_x], joint[responses_y])

# """


# def sparse_correlation(NormData, offset):

#     n_covs = np.floor(max(covariates))
#     for age_i, age_j in offset_indices(n_covs, max_offset):
#         print(age_i, age_j)


# def offset_indices(n_covs, offset):
#     acc = np.zeros((n_covs, n_covs))
#     acc[np.triu_indices(n_covs, 1)] = 1
#     acc[np.triu_indices(n_covs, offset + 1)] = 0
#     for pair in zip(*np.where(acc)):
#         yield pair