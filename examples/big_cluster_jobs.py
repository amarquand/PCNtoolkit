
import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import pcntoolkit.util.output
from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.math_functions.basis_function import BsplineBasisFunction
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR
from pcntoolkit.regression_model.hbr import HBR, NormalLikelihood, SHASHbLikelihood, make_prior
from pcntoolkit.regression_model.test_model import TestModel
from pcntoolkit.util.runner import Runner

# Get the conda environment path
conda_env_path = os.path.join(os.path.dirname(os.path.dirname(sys.executable)))

# Suppress some annoying warnings and logs
pymc_logger = logging.getLogger("pymc")
pymc_logger.setLevel(logging.WARNING)
pymc_logger.propagate = False
warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
pcntoolkit.util.output.Output.set_show_messages(True)

# Path to the dir of this file
this_file_path = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(this_file_path, "resources")

def load_fcon1000(n_response_vars=None,n_largest_sites=None):
    os.makedirs(os.path.join(resource_dir, "data"), exist_ok=True)
    if not os.path.exists(os.path.join(resource_dir, "data/fcon1000.csv")):
        pd.read_csv(
            "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
        ).to_csv(os.path.join(resource_dir, "data/fcon1000.csv"), index=False)

    data = pd.read_csv(os.path.join(resource_dir, "data/fcon1000.csv"))
    subject_ids = ['sub_id']
    covariates = ["age"]
    batch_effects = ["sex", "site"]
    response_vars = data.columns[3:220]

    if n_response_vars is not None:
        response_vars = response_vars[:n_response_vars]

    if n_largest_sites is not None:
        data = data[data['site'].isin(data['site'].value_counts().head(n_largest_sites).index)]

    norm_data = NormData.from_dataframe(
        name="fcon1000",
        dataframe=data,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
        subject_ids=subject_ids
    )
    return norm_data

def load_lifespan_big(n_response_vars=None, n_largest_sites=None, n_subjects=None):
    subject_ids = ['participant_id']
    covariates = ['age']
    batch_effects = ['sex', 'site']
    dtypes = {'participant_id': str, "group": str, "group2": str}
    for col in batch_effects:
        dtypes[col] = str
    for col in covariates:
        dtypes[col] = float
    data= pd.read_csv("/project_cephfs/3022017.06/projects/stijdboe/Data/sairut_data/lifespan_big.csv", dtype=dtypes)

    data = data.dropna(axis=0, how='all', inplace=False)
    data = data.dropna(axis=1, how='any', inplace=False)

    data["sex"] = data["sex"].map({"0.0": "Female", "1.0": "Male", "2.0": "Female"})
    data["site"] = data['site_ID']
    
    # Take only the n largest sites
    if n_largest_sites is not None:
        data = data[data['site_ID'].isin(data['site_ID'].value_counts().head(n_largest_sites).index)]

    # Take only n subjects
    if n_subjects is not None:
        data = data.sample(n=n_subjects, replace=False)

    def is_response_var(str):
        return str not in subject_ids and str not in covariates and str not in batch_effects and not str.startswith('site_') and not str.startswith('group') and not str.startswith('race') and data[str].var() > 0
    
    response_vars = [col for col in data.columns if is_response_var(col)]

    if n_response_vars is not None:
        response_vars = response_vars[:n_response_vars]

    norm_data = NormData.from_dataframe(
        name="lifespan_big",
        dataframe=data,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
        subject_ids=subject_ids
    )
    return norm_data

def main():
    data = "lifespan_big"
    # data = "fcon1000"
    if data == "lifespan_big": 
        normdata = load_lifespan_big(n_response_vars=4)
    else:
        normdata = load_fcon1000()
    train, test = normdata.train_test_split(splits = [0.8, 0.2])
    # Inspect the data
    df = train.to_dataframe()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(data=df, y=("batch_effects", "site"), hue=("batch_effects", "sex"), ax=ax[0], orient="h")
    ax[0].legend(title="Sex")
    ax[0].set_title("Count of sites")
    ax[0].set_xlabel("Site")
    ax[0].set_ylabel("Count")

    sns.scatterplot(
        data=df,
        x=("X", "age"),
        y=("Y", normdata.response_vars.values[0]),
        hue=("batch_effects", "site"),
        style=("batch_effects", "sex"),
        ax=ax[1],
    )
    ax[1].legend([], [])
    ax[1].set_title(f"Scatter plot of age vs {normdata.response_vars.values[0]}")
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel(normdata.response_vars.values[0])

    plt.savefig(os.path.join(resource_dir, f"test_plot_{data}.png"))
    plt.close()

     
    mu = make_prior(
        # Mu is linear because we want to allow the mean to vary as a function of the covariates.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 10.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 10.0)),
        # The intercept is random, because we expect the intercept to vary between sites and sexes.
        intercept=make_prior(
            random=True,
            # Mu is the mean of the intercept, which is normally distributed with a mean of 0 and a standard deviation of 1.
            mu=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
            # Sigma is the scale at which the intercepts vary. It is a positive parameter, so we use a half-normal distribution.
            sigma=make_prior(dist_name="Gamma", dist_params=(3.0, 1.0)),
        ),
        # We use a B-spline basis function to allow for non-linearity in the mean.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    sigma = make_prior(
        # Sigma is also linear, because we want to allow the standard deviation to vary as a function of the covariates: heteroskedasticity.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 2.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        # The intercept is random, because we expect the intercept to vary between sites and sexes.
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        # We use a B-spline basis function to allow for non-linearity in the standard deviation.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        # We use a softplus mapping to ensure that sigma is strictly positive.
        mapping="softplus",
        # We scale the softplus mapping by a factor of 3, to avoid spikes in the resulting density.
        # The parameters (a, b, c) provided to a mapping f are used as: f_abc(x) = f((x - a) / b) * b + c
        # This basically provides an affine transformation of the softplus function.
        # a -> horizontal shift
        # b -> scaling
        # c -> vertical shift
        # You can leave c out, and it will default to 0.
        mapping_params=(0.0, 3.0),
    )

    epsilon = make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        mapping="softplus",
        mapping_params=(0.0, 3.0),
    )
    delta = make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        mapping="softplus",
        mapping_params=(0.0, 3.0, 0.6),
    )

    # epsilon = make_prior(dist_name="Normal", dist_params=(0.0, 1.0))
    # delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0), mapping="softplus", mapping_params=(0.0, 3.0, 0.6))

    template_hbr = HBR(
        # The name of the model.
        name="template",
        # The number of cores to use for sampling.
        cores=16,
        # Whether to show a progress bar during the model fitting.
        progressbar=True,
        # The number of draws to sample from the posterior per chain.
        draws=1500,
        # The number of tuning steps to run.
        tune=500,
        # The number of MCMC chains to run.
        chains=4,
        # The implementation of NUTS to use for sampling.
        nuts_sampler="nutpie",
        # The likelihood function to use for the model.
        likelihood=SHASHbLikelihood(
            mu,
            sigma,
            epsilon,
            delta
        ),
    )


    model = NormativeModel(
        name = "SHASHb2",
        template_regression_model=template_hbr,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=True,
        # The directory to save the model, results, and plots.
        save_dir=os.path.join(resource_dir, f"testmodel_{data}/save_dir"),
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robustminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robustminmax", "none"
        outscaler="standardize",
    )

    runner = Runner(
        cross_validate=False,
        parallelize=True,
        environment=conda_env_path,
        job_type="slurm",  # or "torque" if you are on a torque cluster
        n_jobs=4,
        time_limit="50:00:00",
        memory="16GB",
        n_cores=16,
        max_retries=3,
        log_dir=os.path.join(resource_dir, "runner_output/log_dir"),
        temp_dir=os.path.join(resource_dir, "runner_output/temp_dir"),
    )

    runner.fit_predict(model, train, test, observe=False)


if __name__ == "__main__":
    main()
