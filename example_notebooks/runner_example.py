import os

import pandas as pd

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_blr import NormBLR
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.regression_model.blr.blr_conf import BLRConf
from pcntoolkit.util.runner import Runner

os.makedirs("resources/data", exist_ok=True)

# If you are running this notebook for the first time, you need to download the dataset from github.
# If you have already downloaded the dataset, you can comment out the following line

pd.read_csv(
    "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
).to_csv("resources/data/fcon1000.csv", index=False)
data = pd.read_csv("resources/data/fcon1000.csv")
covariates = ["age"]
batch_effects = ["sex", "site"]
response_vars = "lh_G_cingul-Post-dorsal_thickness,lh_G_cingul-Post-ventral_thickness,lh_G_cuneus_thickness,lh_G_front_inf-Opercular_thickness,lh_G_front_inf-Orbital_thickness,lh_G_front_inf-Triangul_thickness,lh_G_front_middle_thickness,lh_G_front_sup_thickness,lh_G_Ins_lg&S_cent_ins_thickness,lh_G_insular_short_thickness,lh_G_occipital_middle_thickness,lh_G_occipital_sup_thickness,lh_G_oc-temp_lat-fusifor_thickness,lh_G_oc-temp_med-Lingual_thickness,lh_G_oc-temp_med-Parahip_thickness,lh_G_orbital_thickness,lh_G_pariet_inf-Angular_thickness,lh_G_pariet_inf-Supramar_thickness,lh_G_parietal_sup_thickness,lh_G_postcentral_thickness,lh_G_precentral_thickness,lh_G_precuneus_thickness,lh_G_rectus_thickness,lh_G_subcallosal_thickness,lh_G_temp_sup-G_T_transv_thickness,lh_G_temp_sup-Lateral_thickness,lh_G_temp_sup-Plan_polar_thickness,lh_G_temp_sup-Plan_tempo_thickness,lh_G_temporal_inf_thickness,lh_G_temporal_middle_thickness,lh_Lat_Fis-ant-Horizont_thickness,lh_Lat_Fis-ant-Vertical_thickness,lh_Lat_Fis-post_thickness,lh_Pole_occipital_thickness,lh_Pole_temporal_thickness,lh_S_calcarine_thickness".split(
    ","
)
norm_data = NormData.from_dataframe(
    name="full",
    dataframe=data,
    covariates=["age"],
    batch_effects=["sex", "site"],
    response_vars=response_vars,
)

# Leave two sites out for doing transfer and extend later
transfer_sites = ["Milwaukee_b", "Oulu"]
transfer_data, fit_data = norm_data.split_batch_effects({"site": transfer_sites}, names=("transfer", "fit"))

# Split into train and test sets
train, test = fit_data.train_test_split()
transfer_train, transfer_test = transfer_data.train_test_split()

# Create a NormConf object
norm_conf = NormConf(
    savemodel=True,
    saveresults=True,
    save_dir="resources/blr/save_dir",
    inscaler="standardize",
    outscaler="standardize",
    basis_function="bspline",
    basis_function_kwargs={"order": 3, "nknots": 5},
)

blr_conf = BLRConf(
    optimizer="l-bfgs-b",
    n_iter=200,
    heteroskedastic=False,
    intercept=True,
    fixed_effect=True,
    fixed_effect_var=False,
    warp="WarpSinhArcsinh",
    warp_reparam=True,
)


# Using the constructor
new_model = NormBLR(norm_conf=norm_conf, reg_conf=blr_conf)

runner = Runner(
    cross_validate=False,
    cv_folds=3,
    parallelize=True,
    environment="/opt/anaconda3/envs/dev_refactor_2",
    job_type="local",  # or "slurm" if you are on a slurm cluster
    n_jobs=10,
    log_dir="resources/runner_output/log_dir",
    temp_dir="resources/runner_output/temp_dir",
)
# Local parallelization might not work on login nodes due to resource restrictions

runner.fit_predict(new_model, train, test)