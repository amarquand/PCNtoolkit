import os
import pandas as pd
import pcntoolkit as ptk
import numpy as np
import pickle
from matplotlib import pyplot as plt
processing_dir = "HBR_demo/"    # replace with a path to your working directory
if not os.path.isdir(processing_dir):
    os.makedirs(processing_dir)
os.chdir(processing_dir)
processing_dir = os.getcwd()


def main():
    # Optional
    fcon_tr = pd.read_csv('https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000_tr.csv')
    fcon_te = pd.read_csv('https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000_te.csv')
    # fcon_tr = pd.read_csv('https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000_icbm_tr.csv')
    # fcon_te = pd.read_csv('https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000_icbm_te.csv')
    idps = ['rh_MeanThickness_thickness']

    X_train = (fcon_tr['age']/100).to_numpy(dtype=float)
    Y_train = fcon_tr[idps].to_numpy(dtype=float)


    # fcon_tr.loc[fcon_tr['sitenum'] == 21,'sitenum'] = 13
    # fcon_te.loc[fcon_te['sitenum'] == 21,'sitenum'] = 13

    # configure batch effects for site and sex
    batch_effects_train = fcon_tr[['sitenum','sex']].to_numpy(dtype=int)


    # or only site
    # batch_effects_train = fcon_tr[['sitenum']].to_numpy(dtype=int)

    print(np.unique(batch_effects_train,axis=0))     # Here we see there are missing sites


    with open('X_train.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(X_train), file)
    with open('Y_train.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(Y_train), file) 
    with open('trbefile.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(batch_effects_train), file) 


    X_test = (fcon_te['age']/100).to_numpy(dtype=float)
    Y_test = fcon_te[idps].to_numpy(dtype=float)
    batch_effects_test = fcon_te[['sitenum','sex']].to_numpy(dtype=int)
    # batch_effects_test = fcon_te[['sitenum']].to_numpy(dtype=int)
        
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
        
    respfile = os.path.join(processing_dir, 'Y_train.pkl')       # measurements  (eg cortical thickness) of the training samples (columns: the various features/ROIs, rows: observations or subjects)
    covfile = os.path.join(processing_dir, 'X_train.pkl')        # covariates (eg age) the training samples (columns: covariates, rows: observations or subjects)

    testrespfile_path = os.path.join(processing_dir, 'Y_test.pkl')       # measurements  for the testing samples
    testcovfile_path = os.path.join(processing_dir, 'X_test.pkl')        # covariate file for the testing samples

    trbefile = os.path.join(processing_dir, 'trbefile.pkl')      # training batch effects file (eg scanner_id, gender)  (columns: the various batch effects, rows: observations or subjects)
    tsbefile = os.path.join(processing_dir, 'tsbefile.pkl')      # testing batch effects file

    output_path = os.path.join(processing_dir, 'Models/')    #  output path, where the models will be written
    log_dir = os.path.join(processing_dir, 'log/')           #
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    outputsuffix = '_estimate'      # a string to name the output files, of use only to you, so adapt it for your needs.`
    ptk.normative.estimate(covfile=covfile, 
                       respfile=respfile,
                       tsbefile=tsbefile, 
                       trbefile=trbefile, 
                       alg='hbr', 
                       linear_mu='True',
                       random_mu='True',
                       random_intercept_mu='True',
                       random_slope_mu='True',
                       random_sigma='True',
                       log_path=log_dir, 
                       binary=True,
                       output_path=output_path, 
                       testcov= testcovfile_path,
                       testresp = testrespfile_path,
                       outputsuffix=outputsuffix, 
                       savemodel=True)
    
if __name__=="__main__":
    main()