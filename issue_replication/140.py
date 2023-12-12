import pcntoolkit as ptk
import pandas as pd
import numpy as np
from pcntoolkit.normative import fit, estimate, predict, transfer
import pickle
import os
import matplotlib.pyplot as plt
import scipy as sp

# Set seed ================================================================
np.random.seed(112358)

# Load data ===============================================================
fcon = pd.read_csv("/home/stijn/Downloads/fcon1000.csv")
covariates = ["age", "sex"]
responsevars = ["EstimatedTotalIntraCranialVol"]
batch_effects = ["site"]

unique_sites, idxs, unique_counts = np.unique(fcon["site"], return_inverse=True, return_counts=True)
sitemap = dict(zip(unique_sites, range(len(unique_sites))))

# Keep one (large) site out of the sample for transfer learning
transfer_site = 7
print(f"{unique_sites[transfer_site]=}")

# Split data into train/test
train_test_idxs = idxs != transfer_site
train_test_data = fcon[train_test_idxs]
train_part, test_part = 0.8, 0.2
train_data = []
test_data = []
for site in unique_sites:
    site_idxs = np.where(train_test_data["site"] == site)[0]
    train_mask = np.random.choice(
        [True, False], size=len(site_idxs), p=[train_part, 1 - train_part]
    )
    test_mask = ~train_mask
    train_idxs = site_idxs[np.where(train_mask)[0]]
    test_idxs = site_idxs[np.where(test_mask)[0]]
    train_data.append(train_test_data.iloc[train_idxs])
    test_data.append(train_test_data.iloc[test_idxs])

train_data = pd.concat(train_data)
test_data = pd.concat(test_data)

# Split transfer data into train/test
transfer_idxs = idxs == transfer_site
transfer_data = fcon[transfer_idxs]
transfer_train, transfer_test = 0.8, 0.2
transfer_train_data = []
transfer_test_data = []
for site in unique_sites:
    site_idxs = np.where(transfer_data["site"] == site)[0]
    train_mask = np.random.choice(
        [True, False], size=len(site_idxs), p=[transfer_train, 1 - transfer_train]
    )
    test_mask = ~train_mask
    train_idxs = site_idxs[np.where(train_mask)[0]]
    test_idxs = site_idxs[np.where(test_mask)[0]]
    transfer_train_data.append(transfer_data.iloc[train_idxs])
    transfer_test_data.append(transfer_data.iloc[test_idxs])

transfer_train_data = pd.concat(transfer_train_data)
transfer_test_data = pd.concat(transfer_test_data)

# Save data ===============================================================

data_store_path = "/home/stijn/temp/pcntoolkit/issue_140/data"

X_train = train_data[covariates]
y_train = train_data[responsevars]
be_train = train_data[batch_effects]
be_train['site'] = be_train['site'].map(sitemap)

X_test = test_data[covariates]
y_test = test_data[responsevars]
be_test = test_data[batch_effects]
be_test['site'] = be_test['site'].map(sitemap)

X_train_transfer = transfer_train_data[covariates]
y_train_transfer = transfer_train_data[responsevars]
be_train_transfer = transfer_train_data[batch_effects]
be_train_transfer['site'] = be_train_transfer['site'].map(sitemap)

X_test_transfer = transfer_test_data[covariates]
y_test_transfer = transfer_test_data[responsevars]
be_test_transfer = transfer_test_data[batch_effects]
be_test_transfer['site'] = be_test_transfer['site'].map(sitemap)


with open(data_store_path + "/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open(data_store_path + "/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open(data_store_path + "/be_train.pkl", "wb") as f:
    pickle.dump(be_train, f)
with open(data_store_path + "/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open(data_store_path + "/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)
with open(data_store_path + "/be_test.pkl", "wb") as f:
    pickle.dump(be_test, f)
with open(data_store_path + "/X_train_transfer.pkl", "wb") as f:
    pickle.dump(X_train_transfer, f)
with open(data_store_path + "/y_train_transfer.pkl", "wb") as f:
    pickle.dump(y_train_transfer, f)
with open(data_store_path + "/be_train_transfer.pkl", "wb") as f:
    pickle.dump(be_train_transfer, f)
with open(data_store_path + "/X_test_transfer.pkl", "wb") as f:
    pickle.dump(X_test_transfer, f)
with open(data_store_path + "/y_test_transfer.pkl", "wb") as f:
    pickle.dump(y_test_transfer, f)
with open(data_store_path + "/be_test_transfer.pkl", "wb") as f:
    pickle.dump(be_test_transfer, f)

# Train model =============================================================

issue_folder_path = "/home/stijn/temp/pcntoolkit/issue_140"

os.chdir(issue_folder_path)

nm = estimate(
    covfile=data_store_path + "/X_train.pkl",
    respfile=data_store_path + "/y_train.pkl",
    trbefile=data_store_path + "/be_train.pkl",
    testcov = data_store_path + "/X_test.pkl",
    testresp = data_store_path + "/y_test.pkl",
    tsbefile = data_store_path + "/be_test.pkl",
    alg='hbr',
    inscaler='standardize',
    outscaler='standardize',
    savemodel='True',
    n_samples=1000,
    n_tuning=500
)

# # Transfer model ==========================================================

model_path = "/home/stijn/temp/pcntoolkit/issue_140/Models"

transfer(
    model_path=model_path,
    covfile=data_store_path + "/X_train_transfer.pkl",
    respfile=data_store_path + "/y_train_transfer.pkl",
    trbefile=data_store_path + "/be_train_transfer.pkl",
    testcov = data_store_path + "/X_test_transfer.pkl",
    testresp = data_store_path + "/y_test_transfer.pkl",
    tsbefile = data_store_path + "/be_test_transfer.pkl",
    output_path = "/home/stijn/temp/pcntoolkit/issue_140",
    alg='hbr',
    savemodel='True',
    n_samples=1000,
    n_tuning=500
)

# show output =============================================================

## 1. Plot the distribution of the Z-scores for the transfer data

Z_scores_path = "/home/stijn/temp/pcntoolkit/issue_140/Z_transfer.pkl"

with open(Z_scores_path, "rb") as f:
    Z_scores = np.squeeze(pickle.load(f))

print(f"{Z_scores.shape=}")

z_dens = sp.stats.gaussian_kde(Z_scores)
x = np.linspace(-4, 4, 100)
plt.plot(x, z_dens(x))
plt.show()

## 2. Plot the qq-plot of the Z-scores for the training data

sp.stats.probplot(Z_scores, dist="norm", plot=plt)
plt.show()

## 3. Print the SMSE for the training data
SMSE_path = "/home/stijn/temp/pcntoolkit/issue_140/SMSE_transfer.pkl"

with open(SMSE_path, "rb") as f:
    SMSE = pickle.load(f)

print(f"{SMSE=}")