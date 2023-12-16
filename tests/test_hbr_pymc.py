import os
import pandas as pd
import pcntoolkit as ptk
import pymc as pm
import numpy as np
import pickle
from matplotlib import pyplot as plt
import arviz as az
processing_dir = "HBR_demo/"    # replace with a path to your working directory
if not os.path.isdir(processing_dir):
    os.makedirs(processing_dir)
os.chdir(processing_dir)
processing_dir = os.getcwd()


def main():
    fcon_tr = pd.read_csv(
        'https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000_tr.csv')
    fcon_te = pd.read_csv(
        'https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000_te.csv')

    idps = ['rh_MeanThickness_thickness']
    covs = ['age']
    batch_effects = ['sitenum', 'sex']

    X_train = fcon_tr[covs].to_numpy(dtype=float)
    Y_train = fcon_tr[idps].to_numpy(dtype=float)
    batch_effects_train = fcon_tr[batch_effects].to_numpy(dtype=int)

    X_test = fcon_te[covs].to_numpy(dtype=float)
    Y_test = fcon_te[idps].to_numpy(dtype=float)
    batch_effects_test = fcon_te[batch_effects].to_numpy(dtype=int)

    print(X_test.shape, Y_test.shape, batch_effects_test.shape)

    with open('X_train.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(X_train), file)
    with open('Y_train.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(Y_train), file)
    with open('trbefile.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(batch_effects_train), file)
    with open('X_test.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(X_test), file)
    with open('Y_test.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(Y_test), file)
    with open('tsbefile.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(batch_effects_test), file)

    # a simple function to quickly load pickle files
    def ldpkl(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    respfile = os.path.join(processing_dir, 'Y_train.pkl')
    covfile = os.path.join(processing_dir, 'X_train.pkl')

    testrespfile_path = os.path.join(processing_dir, 'Y_test.pkl')
    testcovfile_path = os.path.join(processing_dir, 'X_test.pkl')

    trbefile = os.path.join(processing_dir, 'trbefile.pkl')
    tsbefile = os.path.join(processing_dir, 'tsbefile.pkl')

    output_path = os.path.join(processing_dir, 'Models/')
    log_dir = os.path.join(processing_dir, 'log/')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    outputsuffix = '_estimate'
    nm = ptk.normative.estimate(covfile=covfile,
                                respfile=respfile,
                                trbefile=trbefile,
                                testcov=testcovfile_path,
                                testresp=testrespfile_path,
                                tsbefile=tsbefile,
                                alg='hbr',
                                likelihood='Normal',
                                # model_type='bspline',
                                linear_mu='False',
                                random_intercept_mu='False',
                                random_slope_mu='False',
                                random_intercept_sigma='False',
                                random_sigma='False',
                                random_mu='False',
                                log_path=log_dir,
                                binary='True',
                                n_samples=17,
                                n_tuning=13,
                                n_chains=1,
                                cores=1,
                                target_accept=0.99,
                                init='adapt_diag',
                                inscaler='standardize',
                                outscaler='standardize',
                                output_path=output_path,
                                outputsuffix=outputsuffix,
                                savemodel=True)


if __name__ == "__main__":
    main()
