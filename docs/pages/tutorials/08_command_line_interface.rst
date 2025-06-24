Command line interface
======================

The PCNtoolkit is a python package, but it can also be used from the
command line.

Here we show how to use the PCNtoolkit from the command line.

Furthermore, you can use this script to generate commands for the
command line interface. (Although if you are able to run this notebook,
why not just use it as a python package?)

.. code:: ipython3

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    import sys
    import pickle

BLR Example
-----------

Data preparation
~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Download and split data first
    # If you are running this notebook for the first time, you need to download the dataset from github.
    # If you have already downloaded the dataset, you can comment out the following line
    pd.read_csv(
        "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
    ).to_csv("resources/data/fcon1000.csv", index=False)

.. code:: ipython3

    data = pd.read_csv("resources/data/fcon1000.csv")

.. code:: ipython3

    # Inspect the data
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.scatterplot(data=data, x=("age"), y=("rh_MeanThickness_thickness"), hue=("site"), ax=ax[1])
    ax[1].legend([], [])
    ax[1].set_title("Scatter plot of age vs rh_MeanThickness_thickness")
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel("rh_MeanThickness_thickness")
    sns.countplot(data=data, y="site", hue="sex", ax=ax[0], orient="h")
    ax[0].legend(title="Sex")
    ax[0].set_title("Count of sites")
    ax[0].set_xlabel("Site")
    ax[0].set_ylabel("Count")
    plt.show()



.. image:: 08_command_line_interface_files/08_command_line_interface_6_0.png


.. code:: ipython3

    # Split into X, y, and batch effects
    covariate_columns = ["age"]
    batch_effect_columns = ["sex", "site"]
    response_columns = ["rh_MeanThickness_thickness", "WM-hypointensities"]
    
    X = data[covariate_columns]
    Y = data[response_columns]
    batch_effects = data[batch_effect_columns]
    
    batch_effects_strings = [str(b[0]) + " " + str(b[1]) for b in batch_effects.values]
    
    # Split into train and test set
    trainidx, testidx = train_test_split(data.index, test_size=0.2, random_state=42, stratify=batch_effects_strings)
    train_X = X.loc[trainidx]
    train_Y = Y.loc[trainidx]
    train_batch_effects = batch_effects.loc[trainidx]
    
    test_X = X.loc[testidx]
    test_Y = Y.loc[testidx]
    test_batch_effects = batch_effects.loc[testidx]

.. code:: ipython3

    # Save stuff
    root_dir = os.path.join("resources", "cli_example")
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    resp = os.path.abspath(os.path.join(data_dir, "responses.csv"))
    cov = os.path.abspath(os.path.join(data_dir, "covariates.csv"))
    be = os.path.abspath(os.path.join(data_dir, "batch_effects.csv"))
    
    t_resp = os.path.abspath(os.path.join(data_dir, "test_responses.csv"))
    t_cov = os.path.abspath(os.path.join(data_dir, "test_covariates.csv"))
    t_be = os.path.abspath(os.path.join(data_dir, "test_batch_effects.csv"))
    
    
    with open(cov, "wb") as f:
        pickle.dump(train_X, f)
    with open(resp, "wb") as f:
        pickle.dump(train_Y, f)
    with open(be, "wb") as f:
        pickle.dump(train_batch_effects, f)
    with open(t_cov, "wb") as f:
        pickle.dump(test_X, f)
    with open(t_resp, "wb") as f:
        pickle.dump(test_Y, f)
    with open(t_be, "wb") as f:
        pickle.dump(test_batch_effects, f)

BLR configuration
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    alg = "blr"
    func = "fit_predict"
    
    # normative model configuration
    save_dir = os.path.join(root_dir, "blr_cli", "save_dir")
    savemodel = True
    saveresults = True
    basis_function = "linear"
    inscaler = "standardize"
    outscaler = "standardize"
    
    # Regression model configuration
    optimizer = "l-bfgs-b"
    n_iter = 200
    heteroskedastic = True
    fixed_effect = True
    warp = "WarpSinhArcsinh"
    warp_reparam = True
    
    # runner configuration
    cross_validate = True
    cv_folds = 5
    parallelize = False
    job_type = "local"
    n_jobs = 2
    temp_dir = os.path.join(root_dir, "temp")
    log_dir = os.path.join(root_dir, "log")
    python_env = os.path.join(os.path.dirname(os.path.dirname(sys.executable)))

Constructing command
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    command = "normative"
    args = f"-a {alg} -f {func} -c {cov} -r {resp} -t {t_resp} -e {t_cov} -k {cv_folds}"
    kwargs = f"be={be} t_be={t_be}"
    normative_model_kwargs = f"save_dir={save_dir} savemodel={savemodel} saveresults={saveresults} basis_function={basis_function} inscaler={inscaler} outscaler={outscaler}"
    runner_kwargs = f"cross_validate={cross_validate} parallelize={parallelize} job_type={job_type} n_jobs={n_jobs} temp_dir={temp_dir} log_dir={log_dir} environment={python_env}"
    blr_kwargs = f"optimizer={optimizer} n_iter={n_iter} heteroskedastic={heteroskedastic} fixed_effect={fixed_effect} warp={warp} warp_reparam={warp_reparam}"
    full_command = f"{command} {args} {kwargs} {runner_kwargs} {normative_model_kwargs} {blr_kwargs}"

.. code:: ipython3

    print(full_command)


.. parsed-literal::

    normative -a blr -f fit_predict -c /Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/data/covariates.csv -r /Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/data/responses.csv -t /Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/data/test_responses.csv -e /Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/data/test_covariates.csv -k 5 be=/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/data/batch_effects.csv t_be=/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/data/test_batch_effects.csv cross_validate=True parallelize=False job_type=local n_jobs=2 temp_dir=resources/cli_example/temp log_dir=resources/cli_example/log environment=/opt/anaconda3/envs/uv_refactor save_dir=resources/cli_example/blr_cli/save_dir savemodel=True saveresults=True basis_function=linear inscaler=standardize outscaler=standardize optimizer=l-bfgs-b n_iter=200 heteroskedastic=True fixed_effect=True warp=WarpSinhArcsinh warp_reparam=True


Running command
~~~~~~~~~~~~~~~

.. code:: ipython3

    !{full_command}


.. parsed-literal::

    Process: 30436 - 2025-06-24 12:23:35 - Dataset "fit_data" created.
        - 862 observations
        - 862 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 30436 - 2025-06-24 12:23:35 - Dataset "predict_data" created.
        - 216 observations
        - 216 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 30436 - 2025-06-24 12:23:35 - Task ID created: fit_predict_fit_data__2025-06-24_12:23:35_271.957031
    Process: 30436 - 2025-06-24 12:23:35 - Temporary directory created:
    	/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/temp/fit_predict_fit_data__2025-06-24_12:23:35_271.957031
    Process: 30436 - 2025-06-24 12:23:35 - Log directory created:
    	/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/log/fit_predict_fit_data__2025-06-24_12:23:35_271.957031
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 30436 - 2025-06-24 12:23:35 - Predict data not used in k-fold cross-validation
      warnings.warn(message)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/sklearn/model_selection/_split.py:805: UserWarning: The least populated class in y has only 2 members, which is less than n_splits=5.
      warnings.warn(
    Process: 30436 - 2025-06-24 12:23:35 - Fitting models on 2 response variables.
    Process: 30436 - 2025-06-24 12:23:35 - Fitting model for response_var_0.
    Process: 30436 - 2025-06-24 12:23:36 - Fitting model for response_var_1.
    Process: 30436 - 2025-06-24 12:23:37 - Making predictions on 2 response variables.
    Process: 30436 - 2025-06-24 12:23:37 - Computing z-scores for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:37 - Computing z-scores for response_var_1.
    Process: 30436 - 2025-06-24 12:23:37 - Computing z-scores for response_var_0.
    Process: 30436 - 2025-06-24 12:23:37 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:37 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:23:37 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:23:37 - Computing log-probabilities for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:38 - Computing log-probabilities for response_var_1.
    Process: 30436 - 2025-06-24 12:23:38 - Computing log-probabilities for response_var_0.
    Process: 30436 - 2025-06-24 12:23:38 - Computing yhat for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:38 - Computing yhat for response_var_1.
    Process: 30436 - 2025-06-24 12:23:42 - Computing yhat for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 30436 - 2025-06-24 12:23:47 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Process: 30436 - 2025-06-24 12:23:47 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 30436 - 2025-06-24 12:23:47 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:47 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:23:47 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:23:47 - Harmonizing data on 2 response variables.
    Process: 30436 - 2025-06-24 12:23:47 - Harmonizing data for response_var_1.
    Process: 30436 - 2025-06-24 12:23:47 - Harmonizing data for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    Process: 30436 - 2025-06-24 12:23:47 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_0.
    Process: 30436 - 2025-06-24 12:23:47 - Making predictions on 2 response variables.
    Process: 30436 - 2025-06-24 12:23:47 - Computing z-scores for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:47 - Computing z-scores for response_var_1.
    Process: 30436 - 2025-06-24 12:23:47 - Computing z-scores for response_var_0.
    Process: 30436 - 2025-06-24 12:23:47 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:47 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:23:47 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:23:47 - Computing log-probabilities for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:47 - Computing log-probabilities for response_var_1.
    Process: 30436 - 2025-06-24 12:23:47 - Computing log-probabilities for response_var_0.
    Process: 30436 - 2025-06-24 12:23:47 - Computing yhat for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:47 - Computing yhat for response_var_1.
    Process: 30436 - 2025-06-24 12:23:48 - Computing yhat for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 30436 - 2025-06-24 12:23:50 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Process: 30436 - 2025-06-24 12:23:50 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 30436 - 2025-06-24 12:23:50 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:50 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:23:50 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:23:50 - Harmonizing data on 2 response variables.
    Process: 30436 - 2025-06-24 12:23:50 - Harmonizing data for response_var_1.
    Process: 30436 - 2025-06-24 12:23:50 - Harmonizing data for response_var_0.
    Process: 30436 - 2025-06-24 12:23:50 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_0.
    Process: 30436 - 2025-06-24 12:23:50 - Fitting models on 2 response variables.
    Process: 30436 - 2025-06-24 12:23:50 - Fitting model for response_var_0.
    Process: 30436 - 2025-06-24 12:23:51 - Fitting model for response_var_1.
    Process: 30436 - 2025-06-24 12:23:55 - Making predictions on 2 response variables.
    Process: 30436 - 2025-06-24 12:23:55 - Computing z-scores for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:55 - Computing z-scores for response_var_1.
    Process: 30436 - 2025-06-24 12:23:55 - Computing z-scores for response_var_0.
    Process: 30436 - 2025-06-24 12:23:55 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:55 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:23:55 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:23:55 - Computing log-probabilities for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:55 - Computing log-probabilities for response_var_1.
    Process: 30436 - 2025-06-24 12:23:56 - Computing log-probabilities for response_var_0.
    Process: 30436 - 2025-06-24 12:23:56 - Computing yhat for 2 response variables.
    Process: 30436 - 2025-06-24 12:23:56 - Computing yhat for response_var_1.
    Process: 30436 - 2025-06-24 12:23:59 - Computing yhat for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 30436 - 2025-06-24 12:24:04 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Process: 30436 - 2025-06-24 12:24:04 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 30436 - 2025-06-24 12:24:04 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:04 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:04 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:04 - Harmonizing data on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:04 - Harmonizing data for response_var_1.
    Process: 30436 - 2025-06-24 12:24:04 - Harmonizing data for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    Process: 30436 - 2025-06-24 12:24:05 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_1.
    Process: 30436 - 2025-06-24 12:24:05 - Making predictions on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:05 - Computing z-scores for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:05 - Computing z-scores for response_var_1.
    Process: 30436 - 2025-06-24 12:24:05 - Computing z-scores for response_var_0.
    Process: 30436 - 2025-06-24 12:24:05 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:05 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:05 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:05 - Computing log-probabilities for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:05 - Computing log-probabilities for response_var_1.
    Process: 30436 - 2025-06-24 12:24:05 - Computing log-probabilities for response_var_0.
    Process: 30436 - 2025-06-24 12:24:05 - Computing yhat for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:05 - Computing yhat for response_var_1.
    Process: 30436 - 2025-06-24 12:24:06 - Computing yhat for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 30436 - 2025-06-24 12:24:07 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Process: 30436 - 2025-06-24 12:24:07 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 30436 - 2025-06-24 12:24:07 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:07 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:07 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:07 - Harmonizing data on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:07 - Harmonizing data for response_var_1.
    Process: 30436 - 2025-06-24 12:24:07 - Harmonizing data for response_var_0.
    Process: 30436 - 2025-06-24 12:24:07 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_1.
    Process: 30436 - 2025-06-24 12:24:07 - Fitting models on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:07 - Fitting model for response_var_0.
    Process: 30436 - 2025-06-24 12:24:09 - Fitting model for response_var_1.
    Process: 30436 - 2025-06-24 12:24:11 - Making predictions on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:11 - Computing z-scores for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:11 - Computing z-scores for response_var_1.
    Process: 30436 - 2025-06-24 12:24:11 - Computing z-scores for response_var_0.
    Process: 30436 - 2025-06-24 12:24:11 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:11 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:11 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:12 - Computing log-probabilities for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:12 - Computing log-probabilities for response_var_1.
    Process: 30436 - 2025-06-24 12:24:12 - Computing log-probabilities for response_var_0.
    Process: 30436 - 2025-06-24 12:24:12 - Computing yhat for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:12 - Computing yhat for response_var_1.
    Process: 30436 - 2025-06-24 12:24:16 - Computing yhat for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 30436 - 2025-06-24 12:24:20 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Process: 30436 - 2025-06-24 12:24:20 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 30436 - 2025-06-24 12:24:20 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:20 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:20 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:20 - Harmonizing data on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:20 - Harmonizing data for response_var_1.
    Process: 30436 - 2025-06-24 12:24:20 - Harmonizing data for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    Process: 30436 - 2025-06-24 12:24:21 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_2.
    Process: 30436 - 2025-06-24 12:24:21 - Making predictions on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:21 - Computing z-scores for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:21 - Computing z-scores for response_var_1.
    Process: 30436 - 2025-06-24 12:24:21 - Computing z-scores for response_var_0.
    Process: 30436 - 2025-06-24 12:24:21 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:21 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:21 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:21 - Computing log-probabilities for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:21 - Computing log-probabilities for response_var_1.
    Process: 30436 - 2025-06-24 12:24:21 - Computing log-probabilities for response_var_0.
    Process: 30436 - 2025-06-24 12:24:21 - Computing yhat for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:21 - Computing yhat for response_var_1.
    Process: 30436 - 2025-06-24 12:24:22 - Computing yhat for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 30436 - 2025-06-24 12:24:23 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Process: 30436 - 2025-06-24 12:24:23 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 30436 - 2025-06-24 12:24:23 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:23 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:23 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:23 - Harmonizing data on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:23 - Harmonizing data for response_var_1.
    Process: 30436 - 2025-06-24 12:24:23 - Harmonizing data for response_var_0.
    Process: 30436 - 2025-06-24 12:24:23 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_2.
    Process: 30436 - 2025-06-24 12:24:23 - Fitting models on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:23 - Fitting model for response_var_0.
    Process: 30436 - 2025-06-24 12:24:24 - Fitting model for response_var_1.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.94207e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=4.43156e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 30436 - 2025-06-24 12:24:26 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:619: RuntimeWarning: overflow encountered in divide
      J_transposed[i] = df / dx
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=9.01735e-20): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.9526e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.94221e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.91067e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.91467e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.92745e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.94307e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.93861e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.94208e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.91452e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.9144e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.91584e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.91433e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.91442e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.94214e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.94331e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.94272e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:469: LinAlgWarning: Ill-conditioned matrix (rcond=1.94215e-19): result may not be accurate.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    Process: 30436 - 2025-06-24 12:24:26 - Making predictions on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:26 - Computing z-scores for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:26 - Computing z-scores for response_var_1.
    Process: 30436 - 2025-06-24 12:24:26 - Computing z-scores for response_var_0.
    Process: 30436 - 2025-06-24 12:24:26 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:26 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:26 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:26 - Computing log-probabilities for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:27 - Computing log-probabilities for response_var_1.
    Process: 30436 - 2025-06-24 12:24:27 - Computing log-probabilities for response_var_0.
    Process: 30436 - 2025-06-24 12:24:27 - Computing yhat for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:27 - Computing yhat for response_var_1.
    Process: 30436 - 2025-06-24 12:24:33 - Computing yhat for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 30436 - 2025-06-24 12:24:39 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Process: 30436 - 2025-06-24 12:24:39 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 30436 - 2025-06-24 12:24:39 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:39 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:39 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:39 - Harmonizing data on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:39 - Harmonizing data for response_var_1.
    Process: 30436 - 2025-06-24 12:24:39 - Harmonizing data for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    Process: 30436 - 2025-06-24 12:24:39 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_3.
    Process: 30436 - 2025-06-24 12:24:39 - Making predictions on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:39 - Computing z-scores for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:39 - Computing z-scores for response_var_1.
    Process: 30436 - 2025-06-24 12:24:39 - Computing z-scores for response_var_0.
    Process: 30436 - 2025-06-24 12:24:39 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:39 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:39 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:39 - Computing log-probabilities for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:39 - Computing log-probabilities for response_var_1.
    Process: 30436 - 2025-06-24 12:24:39 - Computing log-probabilities for response_var_0.
    Process: 30436 - 2025-06-24 12:24:39 - Computing yhat for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:39 - Computing yhat for response_var_1.
    Process: 30436 - 2025-06-24 12:24:41 - Computing yhat for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 30436 - 2025-06-24 12:24:42 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Process: 30436 - 2025-06-24 12:24:42 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 30436 - 2025-06-24 12:24:42 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:42 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:42 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:42 - Harmonizing data on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:42 - Harmonizing data for response_var_1.
    Process: 30436 - 2025-06-24 12:24:42 - Harmonizing data for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    Process: 30436 - 2025-06-24 12:24:42 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_3.
    Process: 30436 - 2025-06-24 12:24:42 - Fitting models on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:42 - Fitting model for response_var_0.
    Process: 30436 - 2025-06-24 12:24:43 - Fitting model for response_var_1.
    Process: 30436 - 2025-06-24 12:24:45 - Making predictions on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:45 - Computing z-scores for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:45 - Computing z-scores for response_var_1.
    Process: 30436 - 2025-06-24 12:24:45 - Computing z-scores for response_var_0.
    Process: 30436 - 2025-06-24 12:24:45 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:45 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:45 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:45 - Computing log-probabilities for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:46 - Computing log-probabilities for response_var_1.
    Process: 30436 - 2025-06-24 12:24:46 - Computing log-probabilities for response_var_0.
    Process: 30436 - 2025-06-24 12:24:46 - Computing yhat for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:46 - Computing yhat for response_var_1.
    Process: 30436 - 2025-06-24 12:24:50 - Computing yhat for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 30436 - 2025-06-24 12:24:55 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Process: 30436 - 2025-06-24 12:24:55 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 30436 - 2025-06-24 12:24:55 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:55 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:55 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:55 - Harmonizing data on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:55 - Harmonizing data for response_var_1.
    Process: 30436 - 2025-06-24 12:24:55 - Harmonizing data for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    Process: 30436 - 2025-06-24 12:24:56 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_4.
    Process: 30436 - 2025-06-24 12:24:56 - Making predictions on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:56 - Computing z-scores for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:56 - Computing z-scores for response_var_1.
    Process: 30436 - 2025-06-24 12:24:56 - Computing z-scores for response_var_0.
    Process: 30436 - 2025-06-24 12:24:56 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:56 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:56 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:56 - Computing log-probabilities for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:56 - Computing log-probabilities for response_var_1.
    Process: 30436 - 2025-06-24 12:24:56 - Computing log-probabilities for response_var_0.
    Process: 30436 - 2025-06-24 12:24:56 - Computing yhat for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:56 - Computing yhat for response_var_1.
    Process: 30436 - 2025-06-24 12:24:57 - Computing yhat for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 30436 - 2025-06-24 12:24:58 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Process: 30436 - 2025-06-24 12:24:58 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 30436 - 2025-06-24 12:24:58 - Computing centiles for 2 response variables.
    Process: 30436 - 2025-06-24 12:24:58 - Computing centiles for response_var_1.
    Process: 30436 - 2025-06-24 12:24:58 - Computing centiles for response_var_0.
    Process: 30436 - 2025-06-24 12:24:58 - Harmonizing data on 2 response variables.
    Process: 30436 - 2025-06-24 12:24:58 - Harmonizing data for response_var_1.
    Process: 30436 - 2025-06-24 12:24:58 - Harmonizing data for response_var_0.
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    Process: 30436 - 2025-06-24 12:24:59 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_4.


You can find the results in the resources/cli_example/blr/save_dir
folder.

.. code:: ipython3

    import pandas as pd
    
    a = pd.read_csv(
        "/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/blr_cli/save_dir/folds/fold_1/results/statistics_fit_data_fold_1_predict.csv",
        index_col=0,
    )

HBR example
-----------

.. code:: ipython3

    alg = "hbr"
    func = "fit_predict"
    
    # normative model configuration
    save_dir = os.path.join(root_dir, "hbr", "save_dir")
    savemodel = True
    saveresults = True
    basis_function = "bspline"
    inscaler = "standardize"
    outscaler = "standardize"
    
    
    # Regression model configuration
    draws = 1000
    tune = 500
    chains = 4
    nuts_sampler = "nutpie"
    
    likelihood = "Normal"
    linear_mu = "True"
    random_intercept_mu = "True"
    random_slope_mu = "False"
    linear_sigma = "True"
    random_intercept_sigma = "False"
    random_slope_sigma = "False"

Constructing command
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    command = "normative"
    args = f"-a {alg} -f {func} -c {cov} -r {resp} -t {t_resp} -e {t_cov}"
    kwargs = f"be={be} t_be={t_be}"
    normative_model_kwargs = f"save_dir={save_dir} savemodel={savemodel} saveresults={saveresults} basis_function={basis_function} inscaler={inscaler} outscaler={outscaler}"
    hbr_kwargs = f"draws={draws} tune={tune} chains={chains} nuts_sampler={nuts_sampler} likelihood={likelihood} linear_mu={linear_mu} random_intercept_mu={random_intercept_mu} random_slope_mu={random_slope_mu} linear_sigma={linear_sigma} random_intercept_sigma={random_intercept_sigma} random_slope_sigma={random_slope_sigma}"
    full_command = f"{command} {args} {kwargs} {normative_model_kwargs} {hbr_kwargs}"
    print(full_command)


.. parsed-literal::

    normative -a hbr -f fit_predict -c /Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/data/covariates.csv -r /Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/data/responses.csv -t /Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/data/test_responses.csv -e /Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/data/test_covariates.csv be=/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/data/batch_effects.csv t_be=/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/cli_example/data/test_batch_effects.csv save_dir=resources/cli_example/hbr/save_dir savemodel=True saveresults=True basis_function=bspline inscaler=standardize outscaler=standardize draws=1000 tune=500 chains=4 nuts_sampler=nutpie likelihood=Normal linear_mu=True random_intercept_mu=True random_slope_mu=False linear_sigma=True random_intercept_sigma=False random_slope_sigma=False


Running command
~~~~~~~~~~~~~~~

.. code:: ipython3

    !{full_command}


.. parsed-literal::

    Process: 31517 - 2025-06-24 12:26:40 - No log directory specified. Using default log directory: /Users/stijndeboer/.pcntoolkit/logs
    Process: 31517 - 2025-06-24 12:26:40 - No temporary directory specified. Using default temporary directory: /Users/stijndeboer/.pcntoolkit/temp
    Process: 31517 - 2025-06-24 12:26:40 - Dataset "fit_data" created.
        - 862 observations
        - 862 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 31517 - 2025-06-24 12:26:40 - Dataset "predict_data" created.
        - 216 observations
        - 216 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 31517 - 2025-06-24 12:26:40 - Task ID created: fit_predict_fit_data__2025-06-24_12:26:40_941.230957
    Process: 31517 - 2025-06-24 12:26:40 - Temporary directory created:
    	/Users/stijndeboer/.pcntoolkit/temp/fit_predict_fit_data__2025-06-24_12:26:40_941.230957
    Process: 31517 - 2025-06-24 12:26:40 - Log directory created:
    	/Users/stijndeboer/.pcntoolkit/logs/fit_predict_fit_data__2025-06-24_12:26:40_941.230957
    Process: 31517 - 2025-06-24 12:26:40 - Fitting models on 2 response variables.
    Process: 31517 - 2025-06-24 12:26:40 - Fitting model for response_var_0.
    [2K██████████████████████████████████████████████████████████████████████ 6000/6000Process: 31517 - 2025-06-24 12:26:50 - Fitting model for response_var_1.
    [2K██████████████████████████████████████████████████████████████████████ 6000/6000Process: 31517 - 2025-06-24 12:26:57 - Making predictions on 2 response variables.
    Process: 31517 - 2025-06-24 12:26:57 - Computing z-scores for 2 response variables.
    Process: 31517 - 2025-06-24 12:26:57 - Computing z-scores for response_var_1.
    Sampling: []
    Process: 31517 - 2025-06-24 12:26:59 - Computing z-scores for response_var_0.
    Sampling: []
    Process: 31517 - 2025-06-24 12:26:59 - Computing centiles for 2 response variables.
    Process: 31517 - 2025-06-24 12:26:59 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 31517 - 2025-06-24 12:27:00 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 31517 - 2025-06-24 12:27:01 - Computing log-probabilities for 2 response variables.
    Process: 31517 - 2025-06-24 12:27:02 - Computing log-probabilities for response_var_1.
    Process: 31517 - 2025-06-24 12:27:02 - Computing log-probabilities for response_var_0.
    Process: 31517 - 2025-06-24 12:27:03 - Computing yhat for 2 response variables.
    Sampling: []
    Sampling: []
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 31517 - 2025-06-24 12:27:03 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Process: 31517 - 2025-06-24 12:27:03 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 31517 - 2025-06-24 12:27:03 - Computing centiles for 2 response variables.
    Process: 31517 - 2025-06-24 12:27:03 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 31517 - 2025-06-24 12:27:04 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 31517 - 2025-06-24 12:27:05 - Harmonizing data on 2 response variables.
    Process: 31517 - 2025-06-24 12:27:05 - Harmonizing data for response_var_1.
    Sampling: []
    Sampling: []
    Process: 31517 - 2025-06-24 12:27:06 - Harmonizing data for response_var_0.
    Sampling: []
    Sampling: []
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    Process: 31517 - 2025-06-24 12:27:06 - Saving model to:
    	resources/cli_example/hbr/save_dir.
    Process: 31517 - 2025-06-24 12:27:07 - Making predictions on 2 response variables.
    Process: 31517 - 2025-06-24 12:27:07 - Computing z-scores for 2 response variables.
    Process: 31517 - 2025-06-24 12:27:07 - Computing z-scores for response_var_1.
    Sampling: []
    Process: 31517 - 2025-06-24 12:27:07 - Computing z-scores for response_var_0.
    Sampling: []
    Process: 31517 - 2025-06-24 12:27:07 - Computing centiles for 2 response variables.
    Process: 31517 - 2025-06-24 12:27:07 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 31517 - 2025-06-24 12:27:08 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 31517 - 2025-06-24 12:27:09 - Computing log-probabilities for 2 response variables.
    Process: 31517 - 2025-06-24 12:27:09 - Computing log-probabilities for response_var_1.
    Process: 31517 - 2025-06-24 12:27:09 - Computing log-probabilities for response_var_0.
    Process: 31517 - 2025-06-24 12:27:09 - Computing yhat for 2 response variables.
    Sampling: []
    Sampling: []
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 31517 - 2025-06-24 12:27:10 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Process: 31517 - 2025-06-24 12:27:10 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 31517 - 2025-06-24 12:27:10 - Computing centiles for 2 response variables.
    Process: 31517 - 2025-06-24 12:27:10 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 31517 - 2025-06-24 12:27:10 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 31517 - 2025-06-24 12:27:11 - Harmonizing data on 2 response variables.
    Process: 31517 - 2025-06-24 12:27:11 - Harmonizing data for response_var_1.
    Sampling: []
    Sampling: []
    Process: 31517 - 2025-06-24 12:27:12 - Harmonizing data for response_var_0.
    Sampling: []
    Sampling: []
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:295: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    Process: 31517 - 2025-06-24 12:27:12 - Saving model to:
    	resources/cli_example/hbr/save_dir.


