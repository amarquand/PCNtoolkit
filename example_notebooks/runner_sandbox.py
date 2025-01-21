import os

import pandas as pd

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.hbr.prior import make_prior
from pcntoolkit.util.runner import Runner

resources_dir = "/project/3022000.05/projects/stijdboe/Projects/PCNtoolkit/example_notebooks/resources"
data_dir = os.path.join(resources_dir, "data")
os.makedirs(data_dir, exist_ok=True)

pd.read_csv(
    "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
).to_csv(os.path.join(data_dir, "fcon1000.csv"), index=False)
data = pd.read_csv(os.path.join(data_dir, "fcon1000.csv"))
covariates = ["age"]
batch_effects = ["sex", "site"]
response_vars = ["rh_MeanThickness_thickness", "WM-hypointensities"]
norm_data = NormData.from_dataframe(
    name="full",
    dataframe=data,
    covariates=["age"],
    batch_effects=["sex", "site"],
    response_vars=["rh_MeanThickness_thickness", "WM-hypointensities"],
)

# Leave two sites out for doing transfer and extend later
transfer_sites = ["Milwaukee_b", "Oulu"]
transfer_data, fit_data = norm_data.split_batch_effects({"site": transfer_sites}, names=("transfer", "fit"))

# Split into train and test sets
train, test = fit_data.train_test_split()
transfer_train, transfer_test = transfer_data.train_test_split()

# Create a NormConf object
sandbox_dir = os.path.join(resources_dir, "hbr_runner_sandbox")
os.makedirs(sandbox_dir, exist_ok=True)
norm_conf = NormConf(
    savemodel=True,
    saveresults=True,
    save_dir=os.path.join(sandbox_dir, "save_dir"),
    inscaler="none",
    outscaler="none",
    basis_function="bspline",
    basis_function_kwargs={"order": 3, "nknots": 10},
)

mu = make_prior(
    name="mu",
    linear=True,
    slope=make_prior(dist_name="Normal", dist_params=(0.0, 10.0)),
    intercept=make_prior(
        random=True,
        sigma=make_prior(dist_name="HalfNormal", dist_params=(1.0,)),
        mu=make_prior(dist_name="Normal", dist_params=(0.0, 0.5)),
    ),
)
sigma = make_prior(
    name="sigma",
    linear=True,
    slope=make_prior(dist_name="Normal", dist_params=(0.0, 10.0)),
    intercept=make_prior(random=False),
    mapping="softplus",
    mapping_params=(0.0, 1.0),
)
epsilon = make_prior(linear=True, intercept=make_prior(random=True), slope=make_prior(dist="Normal", params=(0, 10)))
delta = make_prior(
    linear=True,
    intercept=make_prior(random=True),
    slope=make_prior(dist="Normal", params=(0, 10)),
    mapping="softplus",
    mapping_params=(0.0, 1.0, 0.0),
)

# Configure the HBRConf object
hbr_conf = HBRConf(
    draws=1500,
    tune=500,
    chains=4,
    pymc_cores=8,
    likelihood="Normal",
    mu=mu,
    sigma=sigma,
    # epsilon=epsilon,
    # delta=delta,
    nuts_sampler="nutpie",
)

new_hbr_model = NormHBR(norm_conf=norm_conf, reg_conf=hbr_conf)

runner = Runner(
    cross_validate=False,
    parallelize=True,
    time_limit="00:15:00",
    job_type="slurm",  # or "slurm" if you are on a slurm cluster
    n_jobs=2,
    log_dir=os.path.join(sandbox_dir, "log_dir"),
    temp_dir=os.path.join(sandbox_dir, "temp_dir"),
)

runner.fit_predict(new_hbr_model, train, test, observe=False)
