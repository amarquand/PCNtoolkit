import argparse
import os

import numpy as np
import pandas as pd

# Import train_test_split from sklearn
from sklearn.model_selection import train_test_split

# Import the StandardScaler from sklearn
from sklearn.preprocessing import StandardScaler

from pcntoolkit.util.utils import create_design_matrix


def main():

    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    infile=args.input_file.split("/")[-1]

    print(f"Splitting the data located at {args.input_file} into train and test covariates, responses and batch effects...")
    df = pd.read_csv(args.input_file)

    # Select the covariates, responses and batch effects
    cov = df['age']
    resp = df[['SubCortGrayVol','Left-Hippocampus','Brain-Stem','CSF']]
    be = df['site']

    # Standardize the covariates and responses
    cov = StandardScaler().fit_transform(cov.to_numpy()[:,np.newaxis])
    resp = StandardScaler().fit_transform(resp.to_numpy())

    # Map the batch effects to integers
    be_ids = np.unique(be, return_inverse=True)[1]
    
    # Split the data into training and test sets
    train_idx, test_idx = train_test_split(np.arange(len(cov)), test_size=0.2, stratify=be_ids)

    # Create the design matrices
    mean_basis = 'linear'
    var_basis = 'linear'
    Phi_tr = create_design_matrix(cov[train_idx], basis=mean_basis, intercept=False, site_ids=be_ids[train_idx])
    Phi_var_tr = create_design_matrix(cov[train_idx], basis=var_basis)
    Phi_te = create_design_matrix(cov[test_idx], basis=mean_basis, intercept=False, site_ids=be_ids[test_idx])
    Phi_var_te = create_design_matrix(cov[test_idx], basis=var_basis)

    print(f"{Phi_tr.shape=}")
    print(f"{Phi_var_tr.shape=}")
    print(f"{Phi_te.shape=}")
    print(f"{Phi_var_te.shape=}")

    # Save everything
    pd.to_pickle(pd.DataFrame(Phi_tr), os.path.join(args.output_dir, f'X_tr_{infile}.pkl'))
    pd.to_pickle(pd.DataFrame(Phi_var_tr), os.path.join(args.output_dir, f'X_var_tr_{infile}.pkl'))
    pd.to_pickle(pd.DataFrame(Phi_te), os.path.join(args.output_dir, f'X_te_{infile}.pkl'))
    pd.to_pickle(pd.DataFrame(Phi_var_te), os.path.join(args.output_dir, f'X_var_te_{infile}.pkl'))
    pd.to_pickle(pd.DataFrame(resp[train_idx]), os.path.join(args.output_dir, f'Y_tr_{infile}.pkl'))
    pd.to_pickle(pd.DataFrame(resp[test_idx]), os.path.join(args.output_dir, f'Y_te_{infile}.pkl'))
    pd.to_pickle(be[train_idx], os.path.join(args.output_dir, f'be_tr_{infile}.pkl'))
    pd.to_pickle(be[test_idx], os.path.join(args.output_dir, f'be_te_{infile}.pkl'))

    print(f"Done! The files can be found in: {args.output_dir}")

if __name__ == "__main__":
    main()
