Normative Modelling: Bayesian Linear Regression
===============================================

Welcome to this tutorial notebook that will go through the fitting and
evaluation of Normative models with Bayesian Linear Regression.

Let’s jump right in.

Imports
~~~~~~~

.. code:: ipython3

    import warnings
    import logging
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    from pcntoolkit import (
        BLR,
        BsplineBasisFunction,
        NormativeModel,
        NormData,
        load_fcon1000,
        plot_centiles,
        plot_qq,
        plot_ridge,
    )
    
    import pcntoolkit.util.output
    import seaborn as sns
    
    sns.set_style("darkgrid")
    
    # Suppress some annoying warnings and logs
    pymc_logger = logging.getLogger("pymc")
    
    pymc_logger.setLevel(logging.WARNING)
    pymc_logger.propagate = False
    
    warnings.simplefilter(action="ignore", category=FutureWarning)
    pd.options.mode.chained_assignment = None  # default='warn'
    pcntoolkit.util.output.Output.set_show_messages(True)

Load data
---------

First we download a small example dataset from github.

.. code:: ipython3

    # Download an example dataset
    norm_data: NormData = load_fcon1000()
    
    # Select only a few features
    features_to_model = [
        "WM-hypointensities",
        "Right-Lateral-Ventricle",
        "Right-Amygdala",
        "CortexVol",
    ]
    norm_data = norm_data.sel({"response_vars": features_to_model})
    
    # Split into train and test sets
    train, test = norm_data.train_test_split()



.. parsed-literal::

    Process: 28764 - 2025-06-30 13:54:37 - Dataset "fcon1000" created.
        - 1078 observations
        - 1078 unique subjects
        - 1 covariates
        - 217 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        


.. code:: ipython3

    # Visualize the data
    feature_to_plot = features_to_model[1]
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
        y=("Y", feature_to_plot),
        hue=("batch_effects", "site"),
        style=("batch_effects", "sex"),
        ax=ax[1],
    )
    ax[1].legend([], [])
    ax[1].set_title(f"Scatter plot of age vs {feature_to_plot}")
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel(feature_to_plot)
    
    plt.show()



.. image:: 02_BLR_files/02_BLR_6_0.png


Creating a Normative model
--------------------------

A normative model has a regression model for each response variable. We
provide a template regression model which is copied for each response
variable.

A template regression model can be anything that extends the
``RegressionModel``. We provide a number of built-in regression models,
but you can also create your own.

Here we use the ``BLR`` class, which implements a Bayesian Linear
Regression model.

The ``BLR`` class has a number of parameters that can be set,
determining whether and how batch effects are modeled, which basis
expansion to use, and more.

.. code:: ipython3

    template_blr = BLR(
        name="template",
        basis_function_mean=BsplineBasisFunction(
            degree=3, nknots=5
        ),  # We use a B-spline basis expansion for the mean, so the predicted mean is a smooth function of the covariates
        fixed_effect=True,  # By setting fixed_effect=True, we \model offsets in the mean for each individual batch effect,
        heteroskedastic=True,  # We want the variance to be a function of the covariates too
        warp_name="warpsinharcsinh",  # We configure a sinh-arcsinh warp, so we can model flexible non-gaussian distributions
    )

After specifying the regression model, we can configure a normative
model.

A normative model has a number of configuration options: -
``savemodel``: Whether to save the model after fitting. -
``evaluate_model``: Whether to evaluate the model after fitting. -
``saveresults``: Whether to save the results after evaluation. -
``saveplots``: Whether to save the plots after fitting. - ``save_dir``:
The directory to save the model, results, and plots. - ``inscaler``: The
scaler to use for the input data. - ``outscaler``: The scaler to use for
the output data.

.. code:: ipython3

    model = NormativeModel(
        # The regression model to use for the normative model.
        template_regression_model=template_blr,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir="resources/blr/save_dir",
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
    )

Fit the model
-------------

With all that configured, we can fit the model.

The ``fit_predict`` function will fit the model, evaluate it, and save
the results and plots (if so configured).

After that, it will compute Z-scores and centiles for the test set.

All results can be found in the save directory.

.. code:: ipython3

    model.fit_predict(train, test)


.. parsed-literal::

    Process: 28764 - 2025-06-30 13:54:37 - Fitting models on 4 response variables.
    Process: 28764 - 2025-06-30 13:54:37 - Fitting model for WM-hypointensities.
    Process: 28764 - 2025-06-30 13:54:40 - Fitting model for Right-Lateral-Ventricle.
    Process: 28764 - 2025-06-30 13:54:42 - Fitting model for Right-Amygdala.
    Process: 28764 - 2025-06-30 13:54:44 - Fitting model for CortexVol.
    Process: 28764 - 2025-06-30 13:54:45 - Making predictions on 4 response variables.
    Process: 28764 - 2025-06-30 13:54:45 - Computing z-scores for 4 response variables.
    Process: 28764 - 2025-06-30 13:54:45 - Computing z-scores for CortexVol.
    Process: 28764 - 2025-06-30 13:54:45 - Computing z-scores for Right-Amygdala.
    Process: 28764 - 2025-06-30 13:54:45 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 28764 - 2025-06-30 13:54:45 - Computing z-scores for WM-hypointensities.
    Process: 28764 - 2025-06-30 13:54:45 - Computing centiles for 4 response variables.
    Process: 28764 - 2025-06-30 13:54:45 - Computing centiles for CortexVol.
    Process: 28764 - 2025-06-30 13:54:46 - Computing centiles for Right-Amygdala.
    Process: 28764 - 2025-06-30 13:54:46 - Computing centiles for Right-Lateral-Ventricle.
    Process: 28764 - 2025-06-30 13:54:46 - Computing centiles for WM-hypointensities.
    Process: 28764 - 2025-06-30 13:54:46 - Computing log-probabilities for 4 response variables.
    Process: 28764 - 2025-06-30 13:54:46 - Computing log-probabilities for CortexVol.
    Process: 28764 - 2025-06-30 13:54:46 - Computing log-probabilities for Right-Amygdala.
    Process: 28764 - 2025-06-30 13:54:46 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 28764 - 2025-06-30 13:54:46 - Computing log-probabilities for WM-hypointensities.
    Process: 28764 - 2025-06-30 13:54:46 - Computing yhat for 4 response variables.
    Process: 28764 - 2025-06-30 13:54:46 - Computing yhat for CortexVol.
    Process: 28764 - 2025-06-30 13:54:47 - Computing yhat for Right-Amygdala.
    Process: 28764 - 2025-06-30 13:54:47 - Computing yhat for Right-Lateral-Ventricle.
    Process: 28764 - 2025-06-30 13:54:47 - Computing yhat for WM-hypointensities.
    Process: 28764 - 2025-06-30 13:54:47 - Saving model to:
    	resources/blr/save_dir.
    Process: 28764 - 2025-06-30 13:54:47 - Making predictions on 4 response variables.
    Process: 28764 - 2025-06-30 13:54:47 - Computing z-scores for 4 response variables.
    Process: 28764 - 2025-06-30 13:54:47 - Computing z-scores for CortexVol.
    Process: 28764 - 2025-06-30 13:54:47 - Computing z-scores for Right-Amygdala.
    Process: 28764 - 2025-06-30 13:54:47 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 28764 - 2025-06-30 13:54:47 - Computing z-scores for WM-hypointensities.
    Process: 28764 - 2025-06-30 13:54:47 - Computing centiles for 4 response variables.
    Process: 28764 - 2025-06-30 13:54:47 - Computing centiles for CortexVol.
    Process: 28764 - 2025-06-30 13:54:47 - Computing centiles for Right-Amygdala.
    Process: 28764 - 2025-06-30 13:54:47 - Computing centiles for Right-Lateral-Ventricle.
    Process: 28764 - 2025-06-30 13:54:47 - Computing centiles for WM-hypointensities.
    Process: 28764 - 2025-06-30 13:54:47 - Computing log-probabilities for 4 response variables.
    Process: 28764 - 2025-06-30 13:54:47 - Computing log-probabilities for CortexVol.
    Process: 28764 - 2025-06-30 13:54:47 - Computing log-probabilities for Right-Amygdala.
    Process: 28764 - 2025-06-30 13:54:47 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 28764 - 2025-06-30 13:54:47 - Computing log-probabilities for WM-hypointensities.
    Process: 28764 - 2025-06-30 13:54:47 - Computing yhat for 4 response variables.
    Process: 28764 - 2025-06-30 13:54:47 - Computing yhat for CortexVol.
    Process: 28764 - 2025-06-30 13:54:47 - Computing yhat for Right-Amygdala.
    Process: 28764 - 2025-06-30 13:54:47 - Computing yhat for Right-Lateral-Ventricle.
    Process: 28764 - 2025-06-30 13:54:47 - Computing yhat for WM-hypointensities.
    Process: 28764 - 2025-06-30 13:54:47 - Saving model to:
    	resources/blr/save_dir.




.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
    <defs>
    <symbol id="icon-database" viewBox="0 0 32 32">
    <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
    <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    </symbol>
    <symbol id="icon-file-text2" viewBox="0 0 32 32">
    <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
    <path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    </symbol>
    </defs>
    </svg>
    <style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
     *
     */
    
    :root {
      --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
      --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
      --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
      --xr-border-color: var(--jp-border-color2, #e0e0e0);
      --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
      --xr-background-color: var(--jp-layout-color0, white);
      --xr-background-color-row-even: var(--jp-layout-color1, white);
      --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
    }
    
    html[theme="dark"],
    html[data-theme="dark"],
    body[data-theme="dark"],
    body.vscode-dark {
      --xr-font-color0: rgba(255, 255, 255, 1);
      --xr-font-color2: rgba(255, 255, 255, 0.54);
      --xr-font-color3: rgba(255, 255, 255, 0.38);
      --xr-border-color: #1f1f1f;
      --xr-disabled-color: #515151;
      --xr-background-color: #111111;
      --xr-background-color-row-even: #111111;
      --xr-background-color-row-odd: #313131;
    }
    
    .xr-wrap {
      display: block !important;
      min-width: 300px;
      max-width: 700px;
    }
    
    .xr-text-repr-fallback {
      /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
      display: none;
    }
    
    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
      margin-bottom: 4px;
      border-bottom: solid 1px var(--xr-border-color);
    }
    
    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }
    
    .xr-obj-type,
    .xr-array-name {
      margin-left: 2px;
      margin-right: 10px;
    }
    
    .xr-obj-type {
      color: var(--xr-font-color2);
    }
    
    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
    }
    
    .xr-section-item {
      display: contents;
    }
    
    .xr-section-item input {
      display: inline-block;
      opacity: 0;
      height: 0;
    }
    
    .xr-section-item input + label {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-item input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }
    
    .xr-section-item input:focus + label {
      border: 2px solid var(--xr-font-color0);
    }
    
    .xr-section-item input:enabled + label:hover {
      color: var(--xr-font-color0);
    }
    
    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
    }
    
    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.5em;
    }
    
    .xr-section-summary-in:disabled + label {
      color: var(--xr-font-color2);
    }
    
    .xr-section-summary-in + label:before {
      display: inline-block;
      content: "►";
      font-size: 11px;
      width: 15px;
      text-align: center;
    }
    
    .xr-section-summary-in:disabled + label:before {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-summary-in:checked + label:before {
      content: "▼";
    }
    
    .xr-section-summary-in:checked + label > span {
      display: none;
    }
    
    .xr-section-summary,
    .xr-section-inline-details {
      padding-top: 4px;
      padding-bottom: 4px;
    }
    
    .xr-section-inline-details {
      grid-column: 2 / -1;
    }
    
    .xr-section-details {
      display: none;
      grid-column: 1 / -1;
      margin-bottom: 5px;
    }
    
    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }
    
    .xr-array-wrap {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 20px auto;
    }
    
    .xr-array-wrap > label {
      grid-column: 1;
      vertical-align: top;
    }
    
    .xr-preview {
      color: var(--xr-font-color3);
    }
    
    .xr-array-preview,
    .xr-array-data {
      padding: 0 5px !important;
      grid-column: 2;
    }
    
    .xr-array-data,
    .xr-array-in:checked ~ .xr-array-preview {
      display: none;
    }
    
    .xr-array-in:checked ~ .xr-array-data,
    .xr-array-preview {
      display: inline-block;
    }
    
    .xr-dim-list {
      display: inline-block !important;
      list-style: none;
      padding: 0 !important;
      margin: 0;
    }
    
    .xr-dim-list li {
      display: inline-block;
      padding: 0;
      margin: 0;
    }
    
    .xr-dim-list:before {
      content: "(";
    }
    
    .xr-dim-list:after {
      content: ")";
    }
    
    .xr-dim-list li:not(:last-child):after {
      content: ",";
      padding-right: 5px;
    }
    
    .xr-has-index {
      font-weight: bold;
    }
    
    .xr-var-list,
    .xr-var-item {
      display: contents;
    }
    
    .xr-var-item > div,
    .xr-var-item label,
    .xr-var-item > .xr-var-name span {
      background-color: var(--xr-background-color-row-even);
      margin-bottom: 0;
    }
    
    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }
    
    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
    }
    
    .xr-var-name {
      grid-column: 1;
    }
    
    .xr-var-dims {
      grid-column: 2;
    }
    
    .xr-var-dtype {
      grid-column: 3;
      text-align: right;
      color: var(--xr-font-color2);
    }
    
    .xr-var-preview {
      grid-column: 4;
    }
    
    .xr-index-preview {
      grid-column: 2 / 5;
      color: var(--xr-font-color2);
    }
    
    .xr-var-name,
    .xr-var-dims,
    .xr-var-dtype,
    .xr-preview,
    .xr-attrs dt {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-right: 10px;
    }
    
    .xr-var-name:hover,
    .xr-var-dims:hover,
    .xr-var-dtype:hover,
    .xr-attrs dt:hover {
      overflow: visible;
      width: auto;
      z-index: 1;
    }
    
    .xr-var-attrs,
    .xr-var-data,
    .xr-index-data {
      display: none;
      background-color: var(--xr-background-color) !important;
      padding-bottom: 5px !important;
    }
    
    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data,
    .xr-index-data-in:checked ~ .xr-index-data {
      display: block;
    }
    
    .xr-var-data > table {
      float: right;
    }
    
    .xr-var-name span,
    .xr-var-data,
    .xr-index-name div,
    .xr-index-data,
    .xr-attrs {
      padding-left: 25px !important;
    }
    
    .xr-attrs,
    .xr-var-attrs,
    .xr-var-data,
    .xr-index-data {
      grid-column: 1 / -1;
    }
    
    dl.xr-attrs {
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 125px auto;
    }
    
    .xr-attrs dt,
    .xr-attrs dd {
      padding: 0;
      margin: 0;
      float: left;
      padding-right: 10px;
      width: auto;
    }
    
    .xr-attrs dt {
      font-weight: normal;
      grid-column: 1;
    }
    
    .xr-attrs dt:hover span {
      display: inline-block;
      background: var(--xr-background-color);
      padding-right: 10px;
    }
    
    .xr-attrs dd {
      grid-column: 2;
      white-space: pre-wrap;
      word-break: break-all;
    }
    
    .xr-icon-database,
    .xr-icon-file-text2,
    .xr-no-icon {
      display: inline-block;
      vertical-align: middle;
      width: 1em;
      height: 1.5em !important;
      stroke-width: 0;
      stroke: currentColor;
      fill: currentColor;
    }
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 98kB
    Dimensions:            (observations: 216, response_vars: 4, covariates: 1,
                            batch_effect_dims: 2, centile: 5, statistic: 11)
    Coordinates:
      * observations       (observations) int64 2kB 756 769 692 616 ... 751 470 1043
      * response_vars      (response_vars) &lt;U23 368B &#x27;WM-hypointensities&#x27; ... &#x27;Co...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) &lt;U8 352B &#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;
    Data variables:
        subjects           (observations) object 2kB &#x27;Munchen_sub96752&#x27; ... &#x27;Quee...
        Y                  (observations, response_vars) float64 7kB 2.721e+03 .....
        X                  (observations, covariates) float64 2kB 63.0 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 29kB &#x27;F&#x27; ... &#x27;Q...
        Z                  (observations, response_vars) float64 7kB 0.7016 ... -...
        centiles           (centile, observations, response_vars) float64 35kB 60...
        logp               (observations, response_vars) float64 7kB -1.104 ... -...
        Yhat               (observations, response_vars) float64 7kB 1.371e+03 .....
        statistics         (response_vars, statistic) float64 352B -0.662 ... 0.9941
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {&#x27;sex&#x27;: [&#x27;F&#x27;, &#x27;M&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}
        batch_effect_covariate_ranges:  {&#x27;sex&#x27;: {&#x27;F&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-70ff47ed-712a-4523-a02d-2a16de440269' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-70ff47ed-712a-4523-a02d-2a16de440269' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 4</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>centile</span>: 5</li><li><span class='xr-has-index'>statistic</span>: 11</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-1a69aec7-361a-4fe2-89d2-3bcb36725be1' class='xr-section-summary-in' type='checkbox'  checked><label for='section-1a69aec7-361a-4fe2-89d2-3bcb36725be1' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-d308c306-70eb-4b6e-8afc-d9180f751038' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d308c306-70eb-4b6e-8afc-d9180f751038' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dfaa3e61-7db0-4aeb-8765-6c1fac654568' class='xr-var-data-in' type='checkbox'><label for='data-dfaa3e61-7db0-4aeb-8765-6c1fac654568' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Cortex...</div><input id='attrs-eba68b8c-2ee8-4204-934f-5c23064c9275' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-eba68b8c-2ee8-4204-934f-5c23064c9275' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-647b09a7-ceef-432b-9953-0087e43daa2a' class='xr-var-data-in' type='checkbox'><label for='data-647b09a7-ceef-432b-9953-0087e43daa2a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-03b50f4f-f5f2-472c-a517-33c3d18dd6b4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-03b50f4f-f5f2-472c-a517-33c3d18dd6b4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ea08c468-2b8f-43ce-b85b-b419ec55ac84' class='xr-var-data-in' type='checkbox'><label for='data-ea08c468-2b8f-43ce-b85b-b419ec55ac84' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-b3a2dd59-e875-4d8b-94d6-9a52f61c21a3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b3a2dd59-e875-4d8b-94d6-9a52f61c21a3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2c0d92c7-e616-4987-9eba-bb0c18dfb17a' class='xr-var-data-in' type='checkbox'><label for='data-2c0d92c7-e616-4987-9eba-bb0c18dfb17a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-603f5fb3-9abf-4e75-87a0-4de3cfb9d3ce' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-603f5fb3-9abf-4e75-87a0-4de3cfb9d3ce' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b274283e-600a-4d9c-90df-b034d6cd8a0a' class='xr-var-data-in' type='checkbox'><label for='data-b274283e-600a-4d9c-90df-b034d6cd8a0a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-01bf0933-ca18-40cc-95f5-94c8681b73e3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-01bf0933-ca18-40cc-95f5-94c8681b73e3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a46e528d-1416-45d4-911c-e40a5e1ba63f' class='xr-var-data-in' type='checkbox'><label for='data-a46e528d-1416-45d4-911c-e40a5e1ba63f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-2853e704-b441-46cf-8862-9b90725ee28e' class='xr-section-summary-in' type='checkbox'  checked><label for='section-2853e704-b441-46cf-8862-9b90725ee28e' class='xr-section-summary' >Data variables: <span>(9)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subjects</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Munchen_sub96752&#x27; ... &#x27;Queensla...</div><input id='attrs-b97fb03b-ade5-4bd8-802e-d72d6c445852' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b97fb03b-ade5-4bd8-802e-d72d6c445852' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dd740a9b-2413-4424-b756-25a35773b239' class='xr-var-data-in' type='checkbox'><label for='data-dd740a9b-2413-4424-b756-25a35773b239' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Munchen_sub96752&#x27;, &#x27;NewYork_a_sub18638&#x27;, &#x27;Leiden_2200_sub87320&#x27;,
           &#x27;ICBM_sub47658&#x27;, &#x27;AnnArbor_b_sub45569&#x27;, &#x27;Beijing_Zang_sub18960&#x27;,
           &#x27;Leiden_2200_sub18456&#x27;, &#x27;Berlin_Margulies_sub27711&#x27;,
           &#x27;Beijing_Zang_sub87776&#x27;, &#x27;Milwaukee_b_sub63196&#x27;,
           &#x27;Beijing_Zang_sub07144&#x27;, &#x27;Atlanta_sub76280&#x27;,
           &#x27;Beijing_Zang_sub40037&#x27;, &#x27;Cambridge_Buckner_sub17737&#x27;,
           &#x27;ICBM_sub89049&#x27;, &#x27;ICBM_sub55656&#x27;, &#x27;Oulu_sub45566&#x27;,
           &#x27;Beijing_Zang_sub89088&#x27;, &#x27;Atlanta_sub16563&#x27;,
           &#x27;Cambridge_Buckner_sub51172&#x27;, &#x27;Oulu_sub98739&#x27;,
           &#x27;Queensland_sub49845&#x27;, &#x27;Cambridge_Buckner_sub84256&#x27;,
           &#x27;Cleveland_sub80263&#x27;, &#x27;ICBM_sub16607&#x27;, &#x27;Newark_sub46570&#x27;,
           &#x27;NewYork_a_sub88286&#x27;, &#x27;Cambridge_Buckner_sub02591&#x27;,
           &#x27;Oulu_sub66467&#x27;, &#x27;Beijing_Zang_sub74386&#x27;, &#x27;Newark_sub55760&#x27;,
           &#x27;ICBM_sub30623&#x27;, &#x27;Oulu_sub68752&#x27;, &#x27;Leiden_2180_sub19281&#x27;,
           &#x27;Beijing_Zang_sub50972&#x27;, &#x27;Beijing_Zang_sub85030&#x27;,
           &#x27;Milwaukee_b_sub36386&#x27;, &#x27;Baltimore_sub31837&#x27;, &#x27;PaloAlto_sub84978&#x27;,
           &#x27;Oulu_sub01077&#x27;, &#x27;NewYork_a_ADHD_sub54828&#x27;, &#x27;PaloAlto_sub96705&#x27;,
           &#x27;Cambridge_Buckner_sub40635&#x27;, &#x27;ICBM_sub66794&#x27;,
           &#x27;Beijing_Zang_sub46541&#x27;, &#x27;Beijing_Zang_sub87089&#x27;,
           &#x27;Pittsburgh_sub97823&#x27;, &#x27;Beijing_Zang_sub98617&#x27;, &#x27;ICBM_sub92028&#x27;,
    ...
           &#x27;Leiden_2200_sub04484&#x27;, &#x27;Beijing_Zang_sub80163&#x27;, &#x27;ICBM_sub02382&#x27;,
           &#x27;Cambridge_Buckner_sub77435&#x27;, &#x27;NewYork_a_sub54887&#x27;,
           &#x27;Oulu_sub85532&#x27;, &#x27;Baltimore_sub73823&#x27;, &#x27;Beijing_Zang_sub29590&#x27;,
           &#x27;Oulu_sub99718&#x27;, &#x27;Beijing_Zang_sub08455&#x27;, &#x27;Beijing_Zang_sub85543&#x27;,
           &#x27;Cambridge_Buckner_sub45354&#x27;, &#x27;Beijing_Zang_sub07717&#x27;,
           &#x27;Baltimore_sub76160&#x27;, &#x27;Beijing_Zang_sub17093&#x27;,
           &#x27;AnnArbor_b_sub90127&#x27;, &#x27;SaintLouis_sub73002&#x27;,
           &#x27;Queensland_sub93238&#x27;, &#x27;Cleveland_sub34189&#x27;,
           &#x27;Cambridge_Buckner_sub89107&#x27;, &#x27;Atlanta_sub75153&#x27;,
           &#x27;NewYork_a_ADHD_sub73035&#x27;, &#x27;Cambridge_Buckner_sub59434&#x27;,
           &#x27;Milwaukee_b_sub44912&#x27;, &#x27;Cleveland_sub46739&#x27;, &#x27;Oulu_sub20495&#x27;,
           &#x27;SaintLouis_sub28304&#x27;, &#x27;Cambridge_Buckner_sub35430&#x27;,
           &#x27;Oulu_sub86362&#x27;, &#x27;Newark_sub58526&#x27;, &#x27;Leiden_2180_sub12255&#x27;,
           &#x27;ICBM_sub48210&#x27;, &#x27;Cambridge_Buckner_sub77989&#x27;,
           &#x27;Berlin_Margulies_sub75506&#x27;, &#x27;NewYork_a_sub29216&#x27;,
           &#x27;Beijing_Zang_sub05267&#x27;, &#x27;AnnArbor_b_sub18546&#x27;, &#x27;Oulu_sub75620&#x27;,
           &#x27;AnnArbor_b_sub30250&#x27;, &#x27;Berlin_Margulies_sub86111&#x27;,
           &#x27;Beijing_Zang_sub89592&#x27;, &#x27;Beijing_Zang_sub68012&#x27;,
           &#x27;NewYork_a_sub50559&#x27;, &#x27;Munchen_sub66933&#x27;,
           &#x27;Cambridge_Buckner_sub59729&#x27;, &#x27;Queensland_sub86245&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.721e+03 1.289e+04 ... 5.035e+05</div><input id='attrs-6d9cd696-caa4-42a8-aa2d-a589c84c16da' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6d9cd696-caa4-42a8-aa2d-a589c84c16da' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-32a42c2a-1afb-4251-8460-383b63b47878' class='xr-var-data-in' type='checkbox'><label for='data-32a42c2a-1afb-4251-8460-383b63b47878' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[2.72140000e+03, 1.28916000e+04, 1.43940000e+03, 4.57858328e+05],
           [1.14310000e+03, 9.91910000e+03, 1.64970000e+03, 5.26780362e+05],
           [9.55800000e+02, 7.47730000e+03, 1.83850000e+03, 4.95744471e+05],
           [1.47390000e+03, 1.43021000e+04, 1.86770000e+03, 5.85303839e+05],
           [7.57800000e+02, 4.11930000e+03, 1.32500000e+03, 3.33111552e+05],
           [8.71100000e+02, 5.03090000e+03, 1.90730000e+03, 5.10794940e+05],
           [1.20730000e+03, 1.78664000e+04, 2.02220000e+03, 5.50533325e+05],
           [5.95000000e+02, 5.00790000e+03, 2.01070000e+03, 4.67673977e+05],
           [6.82400000e+02, 7.28660000e+03, 1.45630000e+03, 4.60129533e+05],
           [4.45100000e+02, 5.74290000e+03, 1.47450000e+03, 4.44494817e+05],
           [1.62000000e+03, 3.71370000e+03, 2.00110000e+03, 5.59424624e+05],
           [6.02800000e+02, 5.30120000e+03, 1.36100000e+03, 4.21551234e+05],
           [1.43250000e+03, 4.42970000e+03, 1.65080000e+03, 5.19842763e+05],
           [1.90820000e+03, 3.57810000e+03, 1.88370000e+03, 5.06679262e+05],
           [1.83400000e+03, 3.27190000e+03, 2.05120000e+03, 5.35569987e+05],
           [4.59600000e+02, 3.98580000e+03, 1.45470000e+03, 4.67607555e+05],
           [1.21000000e+03, 8.72130000e+03, 1.71430000e+03, 5.30904612e+05],
           [8.45900000e+02, 6.59310000e+03, 1.61900000e+03, 5.09371867e+05],
           [9.95200000e+02, 7.04020000e+03, 1.99490000e+03, 4.60068379e+05],
           [1.73470000e+03, 4.01480000e+03, 1.51620000e+03, 4.87269373e+05],
    ...
           [7.85800000e+02, 5.70900000e+03, 1.47480000e+03, 4.53982166e+05],
           [2.24010000e+03, 4.36660000e+03, 2.04210000e+03, 5.58453123e+05],
           [7.58100000e+02, 6.52980000e+03, 1.56730000e+03, 4.73575183e+05],
           [1.44050000e+03, 6.70530000e+03, 1.20540000e+03, 3.82788491e+05],
           [8.18600000e+02, 9.38330000e+03, 1.96740000e+03, 5.02713911e+05],
           [3.76990000e+03, 1.58644000e+04, 1.79170000e+03, 5.12490348e+05],
           [8.80200000e+02, 4.37020000e+03, 1.75520000e+03, 4.37300069e+05],
           [8.23900000e+02, 6.37900000e+03, 1.57650000e+03, 5.67331908e+05],
           [2.11390000e+03, 1.07225000e+04, 1.84380000e+03, 5.12273764e+05],
           [7.41900000e+02, 8.80170000e+03, 1.60640000e+03, 4.91973562e+05],
           [1.33390000e+03, 6.98000000e+03, 1.74850000e+03, 4.78907154e+05],
           [7.07300000e+02, 5.68070000e+03, 1.53450000e+03, 4.74077083e+05],
           [1.13410000e+03, 5.59220000e+03, 1.62620000e+03, 4.54163909e+05],
           [4.38600000e+02, 6.33000000e+03, 1.59670000e+03, 4.68067037e+05],
           [9.66300000e+02, 9.21550000e+03, 1.78250000e+03, 5.09199708e+05],
           [4.24300000e+02, 4.51110000e+03, 1.70200000e+03, 5.26635258e+05],
           [6.04700000e+02, 7.59080000e+03, 1.69930000e+03, 5.20499663e+05],
           [2.34320000e+03, 1.71923000e+04, 1.79380000e+03, 4.86680791e+05],
           [2.72170000e+03, 6.08600000e+03, 2.32470000e+03, 6.10402006e+05],
           [7.03500000e+02, 1.07003000e+04, 1.67620000e+03, 5.03535771e+05]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 23.27 22.0 ... 72.0 23.0 23.0</div><input id='attrs-cb687e97-bccd-4340-b6fc-ecac9098808c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cb687e97-bccd-4340-b6fc-ecac9098808c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6fa551f4-5ef9-4ef3-bc22-ddf061e49a53' class='xr-var-data-in' type='checkbox'><label for='data-6fa551f4-5ef9-4ef3-bc22-ddf061e49a53' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[63.  ],
           [23.27],
           [22.  ],
           [42.  ],
           [63.  ],
           [23.  ],
           [21.  ],
           [26.  ],
           [21.  ],
           [49.  ],
           [20.  ],
           [23.  ],
           [20.  ],
           [26.  ],
           [35.  ],
           [21.  ],
           [22.  ],
           [19.  ],
           [34.  ],
           [18.  ],
    ...
           [21.  ],
           [20.  ],
           [22.  ],
           [25.  ],
           [25.  ],
           [73.  ],
           [22.  ],
           [28.  ],
           [29.06],
           [19.  ],
           [20.  ],
           [22.  ],
           [19.  ],
           [24.  ],
           [21.  ],
           [24.  ],
           [22.79],
           [72.  ],
           [23.  ],
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-2a63a5b2-241c-421b-ae00-63929405438b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2a63a5b2-241c-421b-ae00-63929405438b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fbab20a9-9357-4640-bf13-b2eac1831dae' class='xr-var-data-in' type='checkbox'><label for='data-fbab20a9-9357-4640-bf13-b2eac1831dae' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;F&#x27;, &#x27;Leiden_2200&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Leiden_2200&#x27;],
           [&#x27;F&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Milwaukee_b&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Atlanta&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;ICBM&#x27;],
           [&#x27;M&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Atlanta&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
    ...
           [&#x27;F&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;F&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;Newark&#x27;],
           [&#x27;M&#x27;, &#x27;Leiden_2180&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;F&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;F&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;M&#x27;, &#x27;Munchen&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.7016 0.4511 ... -1.294 -1.16</div><input id='attrs-b0adba9d-1a36-404a-a392-0fe727feed48' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b0adba9d-1a36-404a-a392-0fe727feed48' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-90f9f65f-b42e-4d03-81f5-9bbc78628c92' class='xr-var-data-in' type='checkbox'><label for='data-90f9f65f-b42e-4d03-81f5-9bbc78628c92' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.70162225,  0.45108228, -0.53539626,  0.29388233],
           [ 0.09560252,  1.21664329, -1.32660684, -0.3074337 ],
           [ 0.244679  ,  0.37958917,  0.59968993,  0.28750188],
           [ 0.28042107,  1.67183607, -0.15935201,  1.45471063],
           [-1.40121344, -1.37798141, -0.25357932, -0.95836225],
           [-0.81397524, -0.74436443,  0.34799242, -0.47493403],
           [ 0.57160882,  2.52009926,  0.27372894,  0.31866583],
           [-0.00961869, -0.67104066,  2.01518006, -0.77740609],
           [-0.91007481,  0.8159425 , -0.99965942, -0.74994733],
           [-1.23273137, -0.50207175, -0.76437823,  0.10862625],
           [ 1.33296633, -1.21489282,  0.79518393,  0.59333855],
           [-0.76348397, -0.3632876 , -1.22406449, -1.18933154],
           [ 1.19735953, -0.44641872,  0.10194602,  1.02909785],
           [ 1.69374608, -0.86756211,  1.2153382 ,  1.85707245],
           [ 1.26045026, -1.62028738,  0.75980913,  0.22212517],
           [-1.29053416, -0.92749277, -1.09769158, -1.08622092],
           [-0.55208825,  0.69857574, -0.60585674, -0.35877223],
           [-0.51134877,  0.59748936, -0.06987092,  0.64787779],
           [-0.38911232, -0.11688521,  0.85645987, -1.05405892],
           [ 1.56801446, -0.45949074, -0.84137104,  0.74471017],
    ...
           [ 0.17445929, -0.16228994, -1.09208539, -1.52449248],
           [ 2.28011385, -0.87118395,  0.74021136,  1.43757284],
           [-1.3893788 ,  0.20910281, -0.28314258, -0.66700356],
           [-0.63087978,  0.17376966, -1.47802761, -0.70411349],
           [-0.88122935,  0.72946772,  0.55984811, -0.61240111],
           [ 0.76885148,  0.25313774,  0.65712374,  0.59194952],
           [-0.91162704, -0.47487633,  0.50314931, -0.44231304],
           [ 0.09302237, -0.52912213, -1.56351687,  0.63930993],
           [ 2.19159939,  1.30561632, -0.2116823 , -0.39862223],
           [-0.78620473,  1.576848  , -0.14463102,  0.0997611 ],
           [-0.30459462,  0.38844484,  0.46917117, -0.58409444],
           [-1.51277298, -0.18081118, -0.46660057, -0.65147609],
           [-0.50244357,  0.26111862,  0.93663544,  0.02685182],
           [-0.29295236, -0.06626632, -0.21885197, -0.88218412],
           [-0.56661466,  1.15133284, -0.31615507, -0.65758914],
           [-1.44556458, -0.53545461,  0.39023864,  1.47225268],
           [-1.2645671 ,  0.34570429, -1.0190337 , -0.51270613],
           [-0.37521858,  0.30340593,  0.90688349,  0.19720961],
           [ 2.94218704, -0.20190964,  1.63342122,  2.63175641],
           [-1.02510519,  1.41445932, -1.29436637, -1.15977053]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>602.5 4.332e+03 ... 6.364e+05</div><input id='attrs-acef19a4-33a2-4fa0-ae3e-2fbb04da2017' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-acef19a4-33a2-4fa0-ae3e-2fbb04da2017' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1c320974-7778-4710-bd81-d070cd00bd4e' class='xr-var-data-in' type='checkbox'><label for='data-1c320974-7778-4710-bd81-d070cd00bd4e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[6.02499877e+02, 4.33236193e+03, 1.18769509e+03, 3.59271959e+05],
            [4.21281375e+02, 2.95673695e+03, 1.59655152e+03, 4.82558820e+05],
            [5.86584804e+01, 2.62568526e+03, 1.43734596e+03, 4.26917640e+05],
            ...,
            [1.09376757e+03, 6.03029980e+03, 1.36040485e+03, 4.08785425e+05],
            [7.35319731e+02, 2.60271436e+03, 1.60336262e+03, 4.43576057e+05],
            [4.09103673e+02, 2.89259240e+03, 1.61722555e+03, 4.87740504e+05]],
    
           [[1.37202714e+03, 7.59905979e+03, 1.41150985e+03, 4.14843101e+05],
            [8.53056440e+02, 5.34531340e+03, 1.75605712e+03, 5.13839529e+05],
            [5.73080209e+02, 5.09543383e+03, 1.62027545e+03, 4.64114601e+05],
            ...,
            [1.97912356e+03, 9.81433702e+03, 1.54459222e+03, 4.53767599e+05],
            [1.10752279e+03, 5.03375722e+03, 1.76185277e+03, 4.77021361e+05],
            [8.51494319e+02, 5.33962216e+03, 1.78024206e+03, 5.20236225e+05]],
    
           [[1.91774220e+03, 1.02946473e+04, 1.53858482e+03, 4.45907038e+05],
            [1.10845319e+03, 6.84376782e+03, 1.87590285e+03, 5.38512882e+05],
            [8.60690018e+02, 6.61578632e+03, 1.73321964e+03, 4.86466591e+05],
            ...,
            [2.93040632e+03, 1.41601353e+04, 1.65314049e+03, 4.79553045e+05],
            [1.34657416e+03, 6.52778071e+03, 1.88199840e+03, 4.98310725e+05],
            [1.11214343e+03, 6.87056322e+03, 1.90770903e+03, 5.46673967e+05]],
    
           [[2.68303759e+03, 1.45898420e+04, 1.64969900e+03, 4.72285064e+05],
            [1.35029905e+03, 8.42027236e+03, 2.02413428e+03, 5.67966086e+05],
            [1.11529802e+03, 8.18244062e+03, 1.85267173e+03, 5.08496167e+05],
            ...,
            [4.55257784e+03, 2.23630644e+04, 1.75681551e+03, 5.03969654e+05],
            [1.59135936e+03, 8.05410003e+03, 2.03107368e+03, 5.20642801e+05],
            [1.35897925e+03, 8.48735976e+03, 2.06936660e+03, 5.78591454e+05]],
    
           [[4.62918131e+03, 2.76306569e+04, 1.80356263e+03, 5.07119540e+05],
            [1.71420463e+03, 1.13785953e+04, 2.32062172e+03, 6.21257829e+05],
            [1.46385904e+03, 1.10598853e+04, 2.07559759e+03, 5.44811169e+05],
            ...,
            [9.03590039e+03, 4.84560680e+04, 1.92541656e+03, 5.43818813e+05],
            [1.99292595e+03, 1.08056977e+04, 2.32890427e+03, 5.59164341e+05],
            [1.73216225e+03, 1.15609968e+04, 2.39562940e+03, 6.36400440e+05]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.104 -0.8605 ... -1.377 -1.117</div><input id='attrs-11f47521-3180-40be-b528-a11bd5300607' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-11f47521-3180-40be-b528-a11bd5300607' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fa113839-eee1-45a4-bb9d-748155cd1aca' class='xr-var-data-in' type='checkbox'><label for='data-fa113839-eee1-45a4-bb9d-748155cd1aca' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.10411658e+00, -8.60518393e-01, -7.16784795e-01,
            -6.17653736e-01],
           [-9.71677438e-02, -1.11022013e+00, -1.40382354e+00,
            -5.02859512e-01],
           [-1.26379491e-01, -4.86884770e-01, -7.18522393e-01,
            -4.79452276e-01],
           [-4.52422793e-01, -1.93333229e+00, -5.00455434e-01,
            -1.68288890e+00],
           [-1.77582129e+00, -1.81675928e+00, -6.45761505e-01,
            -1.26822491e+00],
           [-4.19236257e-01, -7.36725304e-01, -5.88288846e-01,
            -5.35883968e-01],
           [-2.38022803e-01, -3.55809330e+00, -6.40125338e-01,
            -5.74950263e-01],
           [-1.73817820e-01, -7.34598141e-01, -2.61159334e+00,
            -7.64042342e-01],
           [-4.73905984e-01, -7.13539688e-01, -1.10285485e+00,
            -7.24913376e-01],
           [-1.32395779e+00, -8.11761680e-01, -8.46011808e-01,
            -5.51032758e-01],
    ...
           [-1.01592890e-01, -4.79814690e-01, -6.35199304e-01,
            -6.05242163e-01],
           [-1.22292032e+00, -4.50207145e-01, -6.76787810e-01,
            -6.37484674e-01],
           [-1.67351201e-01, -4.67010341e-01, -9.91646098e-01,
            -4.74594397e-01],
           [-1.92083052e-01, -4.58259135e-01, -5.80156839e-01,
            -8.47241055e-01],
           [-2.10829032e-01, -1.01189770e+00, -5.54408585e-01,
            -6.32997915e-01],
           [-1.16844431e+00, -6.26967330e-01, -5.80030112e-01,
            -1.53446235e+00],
           [-9.01672683e-01, -4.57860990e-01, -1.03169522e+00,
            -5.74647516e-01],
           [-1.05251060e+00, -8.92423114e-01, -8.59350508e-01,
            -5.68212417e-01],
           [-4.45119561e+00, -4.50278191e-01, -2.02793908e+00,
            -4.09393596e+00],
           [-6.52068347e-01, -1.38718584e+00, -1.37661021e+00,
            -1.11698385e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.371e+03 1.626e+04 ... 5.219e+05</div><input id='attrs-c82dcebd-7d36-4e54-9243-ccef9ea35a4b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c82dcebd-7d36-4e54-9243-ccef9ea35a4b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-efa1decd-45e1-4f90-bb0c-3224cdae0bf3' class='xr-var-data-in' type='checkbox'><label for='data-efa1decd-45e1-4f90-bb0c-3224cdae0bf3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1.37093283e+03,  1.62610717e+04,  1.22252436e+03,
             4.92413286e+05],
           [ 7.18954011e+02,  6.41936535e+03,  1.73089758e+03,
             5.41840858e+05],
           [ 1.42288839e+03,  5.86124207e+03,  1.21167926e+03,
             4.81501762e+05],
           [ 4.96798433e+02,  5.15502416e+03,  1.75871042e+03,
             5.48912148e+05],
           [ 2.45875065e+03,  7.28806757e+03,  1.41248647e+03,
             3.86459410e+05],
           [ 7.03885482e+02,  5.93726819e+03,  2.10124944e+03,
             5.50482363e+05],
           [ 1.30897512e+03,  9.10039662e+03,  1.83353648e+03,
             4.91187304e+05],
           [ 7.91543823e+02,  6.97046355e+03,  1.69178189e+03,
             4.99982793e+05],
           [ 2.75738858e+02,  5.84372661e+03,  1.59670939e+03,
             4.76456072e+05],
           [ 1.10740496e+03,  6.32563590e+03,  1.86695120e+03,
             4.26195492e+05],
    ...
           [ 8.47308207e+02,  5.72722867e+03,  1.54513861e+03,
             5.16100864e+05],
           [ 8.26677813e+02,  9.91639701e+03,  1.71886358e+03,
             5.79630324e+05],
           [ 1.16924548e+03,  8.14833472e+03,  1.05621363e+03,
             5.10752476e+05],
           [ 5.76475317e+02,  1.23108711e+04,  1.49581793e+03,
             5.17296956e+05],
           [ 7.20133686e+02,  7.49538804e+03,  1.69174686e+03,
             5.36802446e+05],
           [ 9.83860419e+02,  6.04248702e+03,  1.81976226e+03,
             5.06954660e+05],
           [ 1.34394429e+03,  6.51059096e+03,  2.49614155e+03,
             5.54627984e+05],
           [ 5.68503434e+03,  2.77422025e+04,  1.78712363e+03,
             4.66062308e+05],
           [ 1.04548087e+03,  8.90636014e+03,  1.72762572e+03,
             4.93311337e+05],
           [ 8.80848128e+02,  7.92779721e+03,  1.72491281e+03,
             5.21915084e+05]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.662 0.03426 ... 1.296 0.9941</div><input id='attrs-fa757bae-ff94-499a-bfa6-7ac278059162' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fa757bae-ff94-499a-bfa6-7ac278059162' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-26c6ab36-a7ba-40ab-81eb-bad8dde9c231' class='xr-var-data-in' type='checkbox'><label for='data-26c6ab36-a7ba-40ab-81eb-bad8dde9c231' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-6.62042384e-01,  3.42592593e-02,  2.91385952e+00,
             4.60967210e-01,  6.58301830e-01, -6.63371969e-01,
             9.55762269e-01,  2.78790185e-01,  3.23455290e-05,
             1.66337197e+00,  9.68022156e-01],
           [-1.07522462e-01,  4.25925926e-02,  5.72550179e+00,
             3.15209849e-01,  1.11941089e+00, -1.10384238e-01,
             1.07040305e+00,  1.94983270e-01,  4.01868600e-03,
             1.11038424e+00,  9.51151796e-01],
           [-5.21851252e-01,  1.66666667e-02,  2.79072694e+00,
             2.84366630e-01,  1.09690254e+00, -5.22676966e-01,
             1.18834979e+00,  3.08213455e-01,  3.90471862e-06,
             1.52267697e+00,  9.91162660e-01],
           [-2.87061837e-01,  2.31481481e-02,  5.45107151e+00,
             3.63966176e-01,  9.91960143e-01, -2.96306752e-01,
             1.06902543e+00,  4.16065921e-01,  1.88276544e-10,
             1.29630675e+00,  9.94067486e-01]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-2603e909-0e6d-4c4f-a23e-62a22300d216' class='xr-section-summary-in' type='checkbox'  ><label for='section-2603e909-0e6d-4c4f-a23e-62a22300d216' class='xr-section-summary' >Indexes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-7e2a41b2-c054-4420-b6f6-b0bbd760b392' class='xr-index-data-in' type='checkbox'/><label for='index-7e2a41b2-c054-4420-b6f6-b0bbd760b392' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([ 756,  769,  692,  616,   35,  164,  680,  331,  299,  727,
           ...
             27,  959,   29,  346,  304,  264,  798,  751,  470, 1043],
          dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;, length=216))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-d7c0b759-e3a4-499f-b21c-83f1a0395008' class='xr-index-data-in' type='checkbox'/><label for='index-d7c0b759-e3a4-499f-b21c-83f1a0395008' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-e2359c85-15f7-4d62-948c-70f9c5fce2c8' class='xr-index-data-in' type='checkbox'/><label for='index-e2359c85-15f7-4d62-948c-70f9c5fce2c8' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-fd998e92-958c-4f2f-9fbb-902bdbda23c9' class='xr-index-data-in' type='checkbox'/><label for='index-fd998e92-958c-4f2f-9fbb-902bdbda23c9' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>centile</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-475632d6-0b71-4cd7-9404-8c1296839329' class='xr-index-data-in' type='checkbox'/><label for='index-475632d6-0b71-4cd7-9404-8c1296839329' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0.05, 0.25, 0.5, 0.75, 0.95], dtype=&#x27;float64&#x27;, name=&#x27;centile&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-c641f371-0f90-4169-b300-45b7a95f13b4' class='xr-index-data-in' type='checkbox'/><label for='index-c641f371-0f90-4169-b300-45b7a95f13b4' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-42e2fce7-7357-4934-94e8-31fb78b7a836' class='xr-section-summary-in' type='checkbox'  checked><label for='section-42e2fce7-7357-4934-94e8-31fb78b7a836' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{&#x27;sex&#x27;: [&#x27;F&#x27;, &#x27;M&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x317943f60&gt;, {&#x27;sex&#x27;: {&#x27;F&#x27;: 589, &#x27;M&#x27;: 489}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;F&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}, &#x27;M&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



Plot the results
----------------

The PCNtoolkit offers are a number of different plotting functions: 1.
plot_centiles: Plot the predicted centiles for a model 2. plot_qq: Plot
the QQ-plot of the predicted Z-scores 3. plot_ridge: Plot density plots
of the predicted Z-scores

Let’s start with the centiles.

.. code:: ipython3

    plot_centiles(
        model,
        centiles=[0.05, 0.5, 0.95],  # Plot these centiles, the default is [0.05, 0.25, 0.5, 0.75, 0.95]
        scatter_data=train,  # Scatter this data along with the centiles
        batch_effects={"site": ["Beijing_Zang", "AnnArbor_a"], "sex": ["M"]},  # Highlight these groups
        show_other_data=True,  # scatter data not in those groups as smaller black circles
        harmonize=True,  # harmonize the scatterdata, this means that we 'remove' the batch effects from the data, by simulating what the data would have looked like if all data was from the same batch.
        show_yhat=True,
    )


.. parsed-literal::

    Process: 28764 - 2025-06-30 13:54:47 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	site (1)
    	sex (1)
        
    Process: 28764 - 2025-06-30 13:54:47 - Computing centiles for 4 response variables.
    Process: 28764 - 2025-06-30 13:54:47 - Computing centiles for CortexVol.
    Process: 28764 - 2025-06-30 13:54:47 - Computing centiles for Right-Amygdala.
    Process: 28764 - 2025-06-30 13:54:47 - Computing centiles for Right-Lateral-Ventricle.
    Process: 28764 - 2025-06-30 13:54:47 - Computing centiles for WM-hypointensities.
    Process: 28764 - 2025-06-30 13:54:47 - Computing yhat for 4 response variables.
    Process: 28764 - 2025-06-30 13:54:48 - Computing yhat for CortexVol.
    Process: 28764 - 2025-06-30 13:54:48 - Computing yhat for Right-Amygdala.
    Process: 28764 - 2025-06-30 13:54:48 - Computing yhat for Right-Lateral-Ventricle.
    Process: 28764 - 2025-06-30 13:54:48 - Computing yhat for WM-hypointensities.


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 28764 - 2025-06-30 13:54:47 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)


.. parsed-literal::

    Process: 28764 - 2025-06-30 13:54:48 - Harmonizing data on 4 response variables.
    Process: 28764 - 2025-06-30 13:54:48 - Harmonizing data for CortexVol.
    Process: 28764 - 2025-06-30 13:54:48 - Harmonizing data for Right-Amygdala.
    Process: 28764 - 2025-06-30 13:54:48 - Harmonizing data for Right-Lateral-Ventricle.
    Process: 28764 - 2025-06-30 13:54:48 - Harmonizing data for WM-hypointensities.



.. image:: 02_BLR_files/02_BLR_14_3.png



.. image:: 02_BLR_files/02_BLR_14_4.png



.. image:: 02_BLR_files/02_BLR_14_5.png



.. image:: 02_BLR_files/02_BLR_14_6.png


Now let’s see the qq plots

.. code:: ipython3

    plot_qq(test, plot_id_line=True)



.. image:: 02_BLR_files/02_BLR_16_0.png



.. image:: 02_BLR_files/02_BLR_16_1.png



.. image:: 02_BLR_files/02_BLR_16_2.png



.. image:: 02_BLR_files/02_BLR_16_3.png


We can also split the QQ plots by batch effects:

.. code:: ipython3

    plot_qq(test, plot_id_line=True, hue_data="sex", split_data="sex")
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": (0, 0, 0, 0)})



.. image:: 02_BLR_files/02_BLR_18_0.png



.. image:: 02_BLR_files/02_BLR_18_1.png



.. image:: 02_BLR_files/02_BLR_18_2.png



.. image:: 02_BLR_files/02_BLR_18_3.png


And finally the ridge plot:

.. code:: ipython3

    plot_ridge(
        train, "Z", split_by="sex"
    )  # We can also show the 'Y' variable, and that will show the marginal distribution of the response variable, per batch effect.


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:574: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      plt.tight_layout()



.. image:: 02_BLR_files/02_BLR_20_1.png


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:574: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      plt.tight_layout()



.. image:: 02_BLR_files/02_BLR_20_3.png


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:574: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      plt.tight_layout()



.. image:: 02_BLR_files/02_BLR_20_5.png


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:574: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      plt.tight_layout()



.. image:: 02_BLR_files/02_BLR_20_7.png


Model evaluation statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluation statistcs are stored in the NormData object:

.. code:: ipython3

    display(train.get_statistics_df())
    display(test.get_statistics_df())



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>statistic</th>
          <th>MACE</th>
          <th>MAPE</th>
          <th>MSLL</th>
          <th>NLL</th>
          <th>R2</th>
          <th>RMSE</th>
          <th>Rho</th>
          <th>Rho_p</th>
          <th>SMSE</th>
          <th>ShapiroW</th>
        </tr>
        <tr>
          <th>response_vars</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>CortexVol</th>
          <td>0.01</td>
          <td>2.70</td>
          <td>0.47</td>
          <td>0.95</td>
          <td>0.52</td>
          <td>0.69</td>
          <td>0.71</td>
          <td>0.0</td>
          <td>0.48</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>Right-Amygdala</th>
          <td>0.01</td>
          <td>2.51</td>
          <td>0.38</td>
          <td>1.03</td>
          <td>0.39</td>
          <td>0.78</td>
          <td>0.60</td>
          <td>0.0</td>
          <td>0.61</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>Right-Lateral-Ventricle</th>
          <td>0.04</td>
          <td>2.55</td>
          <td>0.43</td>
          <td>0.99</td>
          <td>0.23</td>
          <td>0.88</td>
          <td>0.39</td>
          <td>0.0</td>
          <td>0.77</td>
          <td>0.96</td>
        </tr>
        <tr>
          <th>WM-hypointensities</th>
          <td>0.04</td>
          <td>2.62</td>
          <td>0.77</td>
          <td>0.65</td>
          <td>0.36</td>
          <td>0.80</td>
          <td>0.52</td>
          <td>0.0</td>
          <td>0.64</td>
          <td>0.94</td>
        </tr>
      </tbody>
    </table>
    </div>



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>statistic</th>
          <th>MACE</th>
          <th>MAPE</th>
          <th>MSLL</th>
          <th>NLL</th>
          <th>R2</th>
          <th>RMSE</th>
          <th>Rho</th>
          <th>Rho_p</th>
          <th>SMSE</th>
          <th>ShapiroW</th>
        </tr>
        <tr>
          <th>response_vars</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>CortexVol</th>
          <td>0.02</td>
          <td>2.27</td>
          <td>0.36</td>
          <td>0.99</td>
          <td>0.41</td>
          <td>0.72</td>
          <td>0.64</td>
          <td>0.0</td>
          <td>0.59</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>Right-Amygdala</th>
          <td>0.02</td>
          <td>1.50</td>
          <td>0.28</td>
          <td>1.10</td>
          <td>0.29</td>
          <td>0.81</td>
          <td>0.51</td>
          <td>0.0</td>
          <td>0.71</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>Right-Lateral-Ventricle</th>
          <td>0.04</td>
          <td>3.87</td>
          <td>0.32</td>
          <td>1.12</td>
          <td>0.20</td>
          <td>0.91</td>
          <td>0.28</td>
          <td>0.0</td>
          <td>0.80</td>
          <td>0.95</td>
        </tr>
        <tr>
          <th>WM-hypointensities</th>
          <td>0.03</td>
          <td>1.73</td>
          <td>0.46</td>
          <td>0.66</td>
          <td>0.32</td>
          <td>0.61</td>
          <td>0.50</td>
          <td>0.0</td>
          <td>0.68</td>
          <td>0.97</td>
        </tr>
      </tbody>
    </table>
    </div>


What’s next?
------------

Now we have a normative Bayesian linear regression model, we can use it
to:

- Make predictions on new data
- Harmonize data, this means that we ‘remove’ the batch effects from the
  data, by simulating what the data would have looked like if all data
  was from the same batch.
- Synthesize new data
- Extend the model using data from new batches

Predicting
~~~~~~~~~~

.. code:: ipython3

    model.predict(test)


.. parsed-literal::

    Process: 27518 - 2025-06-24 12:16:24 - Making predictions on 4 response variables.
    Process: 27518 - 2025-06-24 12:16:24 - Computing z-scores for 4 response variables.
    Process: 27518 - 2025-06-24 12:16:24 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 27518 - 2025-06-24 12:16:24 - Computing z-scores for CortexVol.
    Process: 27518 - 2025-06-24 12:16:24 - Computing z-scores for WM-hypointensities.
    Process: 27518 - 2025-06-24 12:16:24 - Computing z-scores for Right-Amygdala.
    Process: 27518 - 2025-06-24 12:16:24 - Computing centiles for 4 response variables.
    Process: 27518 - 2025-06-24 12:16:24 - Computing centiles for Right-Lateral-Ventricle.
    Process: 27518 - 2025-06-24 12:16:24 - Computing centiles for CortexVol.
    Process: 27518 - 2025-06-24 12:16:24 - Computing centiles for WM-hypointensities.
    Process: 27518 - 2025-06-24 12:16:24 - Computing centiles for Right-Amygdala.
    Process: 27518 - 2025-06-24 12:16:24 - Computing log-probabilities for 4 response variables.
    Process: 27518 - 2025-06-24 12:16:24 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 27518 - 2025-06-24 12:16:24 - Computing log-probabilities for CortexVol.
    Process: 27518 - 2025-06-24 12:16:24 - Computing log-probabilities for WM-hypointensities.
    Process: 27518 - 2025-06-24 12:16:24 - Computing log-probabilities for Right-Amygdala.
    Process: 27518 - 2025-06-24 12:16:24 - Computing yhat for 4 response variables.
    Process: 27518 - 2025-06-24 12:16:24 - Computing yhat for Right-Lateral-Ventricle.
    Process: 27518 - 2025-06-24 12:16:26 - Computing yhat for CortexVol.
    Process: 27518 - 2025-06-24 12:16:27 - Computing yhat for WM-hypointensities.
    Process: 27518 - 2025-06-24 12:16:29 - Computing yhat for Right-Amygdala.




.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
    <defs>
    <symbol id="icon-database" viewBox="0 0 32 32">
    <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
    <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    </symbol>
    <symbol id="icon-file-text2" viewBox="0 0 32 32">
    <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
    <path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    </symbol>
    </defs>
    </svg>
    <style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
     *
     */
    
    :root {
      --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
      --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
      --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
      --xr-border-color: var(--jp-border-color2, #e0e0e0);
      --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
      --xr-background-color: var(--jp-layout-color0, white);
      --xr-background-color-row-even: var(--jp-layout-color1, white);
      --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
    }
    
    html[theme="dark"],
    html[data-theme="dark"],
    body[data-theme="dark"],
    body.vscode-dark {
      --xr-font-color0: rgba(255, 255, 255, 1);
      --xr-font-color2: rgba(255, 255, 255, 0.54);
      --xr-font-color3: rgba(255, 255, 255, 0.38);
      --xr-border-color: #1f1f1f;
      --xr-disabled-color: #515151;
      --xr-background-color: #111111;
      --xr-background-color-row-even: #111111;
      --xr-background-color-row-odd: #313131;
    }
    
    .xr-wrap {
      display: block !important;
      min-width: 300px;
      max-width: 700px;
    }
    
    .xr-text-repr-fallback {
      /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
      display: none;
    }
    
    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
      margin-bottom: 4px;
      border-bottom: solid 1px var(--xr-border-color);
    }
    
    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }
    
    .xr-obj-type,
    .xr-array-name {
      margin-left: 2px;
      margin-right: 10px;
    }
    
    .xr-obj-type {
      color: var(--xr-font-color2);
    }
    
    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
    }
    
    .xr-section-item {
      display: contents;
    }
    
    .xr-section-item input {
      display: inline-block;
      opacity: 0;
      height: 0;
    }
    
    .xr-section-item input + label {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-item input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }
    
    .xr-section-item input:focus + label {
      border: 2px solid var(--xr-font-color0);
    }
    
    .xr-section-item input:enabled + label:hover {
      color: var(--xr-font-color0);
    }
    
    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
    }
    
    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.5em;
    }
    
    .xr-section-summary-in:disabled + label {
      color: var(--xr-font-color2);
    }
    
    .xr-section-summary-in + label:before {
      display: inline-block;
      content: "►";
      font-size: 11px;
      width: 15px;
      text-align: center;
    }
    
    .xr-section-summary-in:disabled + label:before {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-summary-in:checked + label:before {
      content: "▼";
    }
    
    .xr-section-summary-in:checked + label > span {
      display: none;
    }
    
    .xr-section-summary,
    .xr-section-inline-details {
      padding-top: 4px;
      padding-bottom: 4px;
    }
    
    .xr-section-inline-details {
      grid-column: 2 / -1;
    }
    
    .xr-section-details {
      display: none;
      grid-column: 1 / -1;
      margin-bottom: 5px;
    }
    
    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }
    
    .xr-array-wrap {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 20px auto;
    }
    
    .xr-array-wrap > label {
      grid-column: 1;
      vertical-align: top;
    }
    
    .xr-preview {
      color: var(--xr-font-color3);
    }
    
    .xr-array-preview,
    .xr-array-data {
      padding: 0 5px !important;
      grid-column: 2;
    }
    
    .xr-array-data,
    .xr-array-in:checked ~ .xr-array-preview {
      display: none;
    }
    
    .xr-array-in:checked ~ .xr-array-data,
    .xr-array-preview {
      display: inline-block;
    }
    
    .xr-dim-list {
      display: inline-block !important;
      list-style: none;
      padding: 0 !important;
      margin: 0;
    }
    
    .xr-dim-list li {
      display: inline-block;
      padding: 0;
      margin: 0;
    }
    
    .xr-dim-list:before {
      content: "(";
    }
    
    .xr-dim-list:after {
      content: ")";
    }
    
    .xr-dim-list li:not(:last-child):after {
      content: ",";
      padding-right: 5px;
    }
    
    .xr-has-index {
      font-weight: bold;
    }
    
    .xr-var-list,
    .xr-var-item {
      display: contents;
    }
    
    .xr-var-item > div,
    .xr-var-item label,
    .xr-var-item > .xr-var-name span {
      background-color: var(--xr-background-color-row-even);
      margin-bottom: 0;
    }
    
    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }
    
    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
    }
    
    .xr-var-name {
      grid-column: 1;
    }
    
    .xr-var-dims {
      grid-column: 2;
    }
    
    .xr-var-dtype {
      grid-column: 3;
      text-align: right;
      color: var(--xr-font-color2);
    }
    
    .xr-var-preview {
      grid-column: 4;
    }
    
    .xr-index-preview {
      grid-column: 2 / 5;
      color: var(--xr-font-color2);
    }
    
    .xr-var-name,
    .xr-var-dims,
    .xr-var-dtype,
    .xr-preview,
    .xr-attrs dt {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-right: 10px;
    }
    
    .xr-var-name:hover,
    .xr-var-dims:hover,
    .xr-var-dtype:hover,
    .xr-attrs dt:hover {
      overflow: visible;
      width: auto;
      z-index: 1;
    }
    
    .xr-var-attrs,
    .xr-var-data,
    .xr-index-data {
      display: none;
      background-color: var(--xr-background-color) !important;
      padding-bottom: 5px !important;
    }
    
    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data,
    .xr-index-data-in:checked ~ .xr-index-data {
      display: block;
    }
    
    .xr-var-data > table {
      float: right;
    }
    
    .xr-var-name span,
    .xr-var-data,
    .xr-index-name div,
    .xr-index-data,
    .xr-attrs {
      padding-left: 25px !important;
    }
    
    .xr-attrs,
    .xr-var-attrs,
    .xr-var-data,
    .xr-index-data {
      grid-column: 1 / -1;
    }
    
    dl.xr-attrs {
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 125px auto;
    }
    
    .xr-attrs dt,
    .xr-attrs dd {
      padding: 0;
      margin: 0;
      float: left;
      padding-right: 10px;
      width: auto;
    }
    
    .xr-attrs dt {
      font-weight: normal;
      grid-column: 1;
    }
    
    .xr-attrs dt:hover span {
      display: inline-block;
      background: var(--xr-background-color);
      padding-right: 10px;
    }
    
    .xr-attrs dd {
      grid-column: 2;
      white-space: pre-wrap;
      word-break: break-all;
    }
    
    .xr-icon-database,
    .xr-icon-file-text2,
    .xr-no-icon {
      display: inline-block;
      vertical-align: middle;
      width: 1em;
      height: 1.5em !important;
      stroke-width: 0;
      stroke: currentColor;
      fill: currentColor;
    }
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 98kB
    Dimensions:            (observations: 216, response_vars: 4, covariates: 1,
                            batch_effect_dims: 2, centile: 5, statistic: 10)
    Coordinates:
      * observations       (observations) int64 2kB 756 769 692 616 ... 751 470 1043
      * response_vars      (response_vars) &lt;U23 368B &#x27;WM-hypointensities&#x27; ... &#x27;Co...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) &lt;U8 320B &#x27;MACE&#x27; &#x27;MAPE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;
    Data variables:
        subjects           (observations) object 2kB &#x27;Munchen_sub96752&#x27; ... &#x27;Quee...
        Y                  (observations, response_vars) float64 7kB 2.721e+03 .....
        X                  (observations, covariates) float64 2kB 63.0 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 29kB &#x27;F&#x27; ... &#x27;Q...
        Z                  (observations, response_vars) float64 7kB 0.7016 ... -...
        centiles           (centile, observations, response_vars) float64 35kB 60...
        logp               (observations, response_vars) float64 7kB -1.104 ... -...
        Yhat               (observations, response_vars) float64 7kB 2.182e+03 .....
        statistics         (response_vars, statistic) float64 320B 0.03426 ... 0....
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {&#x27;sex&#x27;: [&#x27;F&#x27;, &#x27;M&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;...
        batch_effect_counts:            {&#x27;sex&#x27;: {&#x27;F&#x27;: 589, &#x27;M&#x27;: 489}, &#x27;site&#x27;: {&#x27;A...
        batch_effect_covariate_ranges:  {&#x27;sex&#x27;: {&#x27;F&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;...
        covariate_ranges:               {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-0c5fe867-6819-4f46-823f-c5c72662ac13' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-0c5fe867-6819-4f46-823f-c5c72662ac13' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 4</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>centile</span>: 5</li><li><span class='xr-has-index'>statistic</span>: 10</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-43a735ba-33ea-4e20-8e27-50b742c17772' class='xr-section-summary-in' type='checkbox'  checked><label for='section-43a735ba-33ea-4e20-8e27-50b742c17772' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-fc88fa70-7708-45de-a4e8-ebe21a26190c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fc88fa70-7708-45de-a4e8-ebe21a26190c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f2a6f821-a17e-49e0-a6fc-32fb6b8e2421' class='xr-var-data-in' type='checkbox'><label for='data-f2a6f821-a17e-49e0-a6fc-32fb6b8e2421' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Cortex...</div><input id='attrs-ccd2b3bb-586b-4e86-90ff-e2b400d78b6e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ccd2b3bb-586b-4e86-90ff-e2b400d78b6e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6a251e31-b04f-4a1d-80e0-4951d8278ef0' class='xr-var-data-in' type='checkbox'><label for='data-6a251e31-b04f-4a1d-80e0-4951d8278ef0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-e67c4461-fa0f-479c-8cc9-d24b761d58d1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e67c4461-fa0f-479c-8cc9-d24b761d58d1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-88e1196e-6ffd-4a51-8249-87542e88dcb6' class='xr-var-data-in' type='checkbox'><label for='data-88e1196e-6ffd-4a51-8249-87542e88dcb6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-e425e731-c252-4564-b755-c8ad7c7c0cfe' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e425e731-c252-4564-b755-c8ad7c7c0cfe' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6809c537-a916-4e7f-99bb-457e10cd7af4' class='xr-var-data-in' type='checkbox'><label for='data-6809c537-a916-4e7f-99bb-457e10cd7af4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-b3db2238-078d-4dc3-b8ff-5c9b43c27588' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b3db2238-078d-4dc3-b8ff-5c9b43c27588' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1b09b2ca-2e33-4887-992c-794e6b72e849' class='xr-var-data-in' type='checkbox'><label for='data-1b09b2ca-2e33-4887-992c-794e6b72e849' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;MACE&#x27; &#x27;MAPE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-8cdcfa31-4b4a-42aa-a51a-026ff582915c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8cdcfa31-4b4a-42aa-a51a-026ff582915c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fc290e80-d872-41ff-88dd-dd5225bd0b79' class='xr-var-data-in' type='checkbox'><label for='data-fc290e80-d872-41ff-88dd-dd5225bd0b79' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;,
           &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d5d562a7-231e-494e-aa5d-bbb6d26f5e7c' class='xr-section-summary-in' type='checkbox'  checked><label for='section-d5d562a7-231e-494e-aa5d-bbb6d26f5e7c' class='xr-section-summary' >Data variables: <span>(9)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subjects</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Munchen_sub96752&#x27; ... &#x27;Queensla...</div><input id='attrs-bfa71bd5-3ac5-44c1-b895-6fc9eccf4789' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bfa71bd5-3ac5-44c1-b895-6fc9eccf4789' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-42b70f56-ce93-44bf-926c-e6e86610edc2' class='xr-var-data-in' type='checkbox'><label for='data-42b70f56-ce93-44bf-926c-e6e86610edc2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Munchen_sub96752&#x27;, &#x27;NewYork_a_sub18638&#x27;, &#x27;Leiden_2200_sub87320&#x27;,
           &#x27;ICBM_sub47658&#x27;, &#x27;AnnArbor_b_sub45569&#x27;, &#x27;Beijing_Zang_sub18960&#x27;,
           &#x27;Leiden_2200_sub18456&#x27;, &#x27;Berlin_Margulies_sub27711&#x27;,
           &#x27;Beijing_Zang_sub87776&#x27;, &#x27;Milwaukee_b_sub63196&#x27;,
           &#x27;Beijing_Zang_sub07144&#x27;, &#x27;Atlanta_sub76280&#x27;,
           &#x27;Beijing_Zang_sub40037&#x27;, &#x27;Cambridge_Buckner_sub17737&#x27;,
           &#x27;ICBM_sub89049&#x27;, &#x27;ICBM_sub55656&#x27;, &#x27;Oulu_sub45566&#x27;,
           &#x27;Beijing_Zang_sub89088&#x27;, &#x27;Atlanta_sub16563&#x27;,
           &#x27;Cambridge_Buckner_sub51172&#x27;, &#x27;Oulu_sub98739&#x27;,
           &#x27;Queensland_sub49845&#x27;, &#x27;Cambridge_Buckner_sub84256&#x27;,
           &#x27;Cleveland_sub80263&#x27;, &#x27;ICBM_sub16607&#x27;, &#x27;Newark_sub46570&#x27;,
           &#x27;NewYork_a_sub88286&#x27;, &#x27;Cambridge_Buckner_sub02591&#x27;,
           &#x27;Oulu_sub66467&#x27;, &#x27;Beijing_Zang_sub74386&#x27;, &#x27;Newark_sub55760&#x27;,
           &#x27;ICBM_sub30623&#x27;, &#x27;Oulu_sub68752&#x27;, &#x27;Leiden_2180_sub19281&#x27;,
           &#x27;Beijing_Zang_sub50972&#x27;, &#x27;Beijing_Zang_sub85030&#x27;,
           &#x27;Milwaukee_b_sub36386&#x27;, &#x27;Baltimore_sub31837&#x27;, &#x27;PaloAlto_sub84978&#x27;,
           &#x27;Oulu_sub01077&#x27;, &#x27;NewYork_a_ADHD_sub54828&#x27;, &#x27;PaloAlto_sub96705&#x27;,
           &#x27;Cambridge_Buckner_sub40635&#x27;, &#x27;ICBM_sub66794&#x27;,
           &#x27;Beijing_Zang_sub46541&#x27;, &#x27;Beijing_Zang_sub87089&#x27;,
           &#x27;Pittsburgh_sub97823&#x27;, &#x27;Beijing_Zang_sub98617&#x27;, &#x27;ICBM_sub92028&#x27;,
    ...
           &#x27;Leiden_2200_sub04484&#x27;, &#x27;Beijing_Zang_sub80163&#x27;, &#x27;ICBM_sub02382&#x27;,
           &#x27;Cambridge_Buckner_sub77435&#x27;, &#x27;NewYork_a_sub54887&#x27;,
           &#x27;Oulu_sub85532&#x27;, &#x27;Baltimore_sub73823&#x27;, &#x27;Beijing_Zang_sub29590&#x27;,
           &#x27;Oulu_sub99718&#x27;, &#x27;Beijing_Zang_sub08455&#x27;, &#x27;Beijing_Zang_sub85543&#x27;,
           &#x27;Cambridge_Buckner_sub45354&#x27;, &#x27;Beijing_Zang_sub07717&#x27;,
           &#x27;Baltimore_sub76160&#x27;, &#x27;Beijing_Zang_sub17093&#x27;,
           &#x27;AnnArbor_b_sub90127&#x27;, &#x27;SaintLouis_sub73002&#x27;,
           &#x27;Queensland_sub93238&#x27;, &#x27;Cleveland_sub34189&#x27;,
           &#x27;Cambridge_Buckner_sub89107&#x27;, &#x27;Atlanta_sub75153&#x27;,
           &#x27;NewYork_a_ADHD_sub73035&#x27;, &#x27;Cambridge_Buckner_sub59434&#x27;,
           &#x27;Milwaukee_b_sub44912&#x27;, &#x27;Cleveland_sub46739&#x27;, &#x27;Oulu_sub20495&#x27;,
           &#x27;SaintLouis_sub28304&#x27;, &#x27;Cambridge_Buckner_sub35430&#x27;,
           &#x27;Oulu_sub86362&#x27;, &#x27;Newark_sub58526&#x27;, &#x27;Leiden_2180_sub12255&#x27;,
           &#x27;ICBM_sub48210&#x27;, &#x27;Cambridge_Buckner_sub77989&#x27;,
           &#x27;Berlin_Margulies_sub75506&#x27;, &#x27;NewYork_a_sub29216&#x27;,
           &#x27;Beijing_Zang_sub05267&#x27;, &#x27;AnnArbor_b_sub18546&#x27;, &#x27;Oulu_sub75620&#x27;,
           &#x27;AnnArbor_b_sub30250&#x27;, &#x27;Berlin_Margulies_sub86111&#x27;,
           &#x27;Beijing_Zang_sub89592&#x27;, &#x27;Beijing_Zang_sub68012&#x27;,
           &#x27;NewYork_a_sub50559&#x27;, &#x27;Munchen_sub66933&#x27;,
           &#x27;Cambridge_Buckner_sub59729&#x27;, &#x27;Queensland_sub86245&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.721e+03 1.289e+04 ... 5.035e+05</div><input id='attrs-bfe06ed4-8216-474a-aab9-313be348372e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bfe06ed4-8216-474a-aab9-313be348372e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6067aa8a-21a3-4865-8d55-eb0a6ee4ddd0' class='xr-var-data-in' type='checkbox'><label for='data-6067aa8a-21a3-4865-8d55-eb0a6ee4ddd0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[2.72140000e+03, 1.28916000e+04, 1.43940000e+03, 4.57858328e+05],
           [1.14310000e+03, 9.91910000e+03, 1.64970000e+03, 5.26780362e+05],
           [9.55800000e+02, 7.47730000e+03, 1.83850000e+03, 4.95744471e+05],
           [1.47390000e+03, 1.43021000e+04, 1.86770000e+03, 5.85303839e+05],
           [7.57800000e+02, 4.11930000e+03, 1.32500000e+03, 3.33111552e+05],
           [8.71100000e+02, 5.03090000e+03, 1.90730000e+03, 5.10794940e+05],
           [1.20730000e+03, 1.78664000e+04, 2.02220000e+03, 5.50533325e+05],
           [5.95000000e+02, 5.00790000e+03, 2.01070000e+03, 4.67673977e+05],
           [6.82400000e+02, 7.28660000e+03, 1.45630000e+03, 4.60129533e+05],
           [4.45100000e+02, 5.74290000e+03, 1.47450000e+03, 4.44494817e+05],
           [1.62000000e+03, 3.71370000e+03, 2.00110000e+03, 5.59424624e+05],
           [6.02800000e+02, 5.30120000e+03, 1.36100000e+03, 4.21551234e+05],
           [1.43250000e+03, 4.42970000e+03, 1.65080000e+03, 5.19842763e+05],
           [1.90820000e+03, 3.57810000e+03, 1.88370000e+03, 5.06679262e+05],
           [1.83400000e+03, 3.27190000e+03, 2.05120000e+03, 5.35569987e+05],
           [4.59600000e+02, 3.98580000e+03, 1.45470000e+03, 4.67607555e+05],
           [1.21000000e+03, 8.72130000e+03, 1.71430000e+03, 5.30904612e+05],
           [8.45900000e+02, 6.59310000e+03, 1.61900000e+03, 5.09371867e+05],
           [9.95200000e+02, 7.04020000e+03, 1.99490000e+03, 4.60068379e+05],
           [1.73470000e+03, 4.01480000e+03, 1.51620000e+03, 4.87269373e+05],
    ...
           [7.85800000e+02, 5.70900000e+03, 1.47480000e+03, 4.53982166e+05],
           [2.24010000e+03, 4.36660000e+03, 2.04210000e+03, 5.58453123e+05],
           [7.58100000e+02, 6.52980000e+03, 1.56730000e+03, 4.73575183e+05],
           [1.44050000e+03, 6.70530000e+03, 1.20540000e+03, 3.82788491e+05],
           [8.18600000e+02, 9.38330000e+03, 1.96740000e+03, 5.02713911e+05],
           [3.76990000e+03, 1.58644000e+04, 1.79170000e+03, 5.12490348e+05],
           [8.80200000e+02, 4.37020000e+03, 1.75520000e+03, 4.37300069e+05],
           [8.23900000e+02, 6.37900000e+03, 1.57650000e+03, 5.67331908e+05],
           [2.11390000e+03, 1.07225000e+04, 1.84380000e+03, 5.12273764e+05],
           [7.41900000e+02, 8.80170000e+03, 1.60640000e+03, 4.91973562e+05],
           [1.33390000e+03, 6.98000000e+03, 1.74850000e+03, 4.78907154e+05],
           [7.07300000e+02, 5.68070000e+03, 1.53450000e+03, 4.74077083e+05],
           [1.13410000e+03, 5.59220000e+03, 1.62620000e+03, 4.54163909e+05],
           [4.38600000e+02, 6.33000000e+03, 1.59670000e+03, 4.68067037e+05],
           [9.66300000e+02, 9.21550000e+03, 1.78250000e+03, 5.09199708e+05],
           [4.24300000e+02, 4.51110000e+03, 1.70200000e+03, 5.26635258e+05],
           [6.04700000e+02, 7.59080000e+03, 1.69930000e+03, 5.20499663e+05],
           [2.34320000e+03, 1.71923000e+04, 1.79380000e+03, 4.86680791e+05],
           [2.72170000e+03, 6.08600000e+03, 2.32470000e+03, 6.10402006e+05],
           [7.03500000e+02, 1.07003000e+04, 1.67620000e+03, 5.03535771e+05]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 23.27 22.0 ... 72.0 23.0 23.0</div><input id='attrs-a71e231b-de55-47c3-a12b-de2fb75fd0e9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a71e231b-de55-47c3-a12b-de2fb75fd0e9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-37463c58-c81c-46ce-a068-d2f586ad752a' class='xr-var-data-in' type='checkbox'><label for='data-37463c58-c81c-46ce-a068-d2f586ad752a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[63.  ],
           [23.27],
           [22.  ],
           [42.  ],
           [63.  ],
           [23.  ],
           [21.  ],
           [26.  ],
           [21.  ],
           [49.  ],
           [20.  ],
           [23.  ],
           [20.  ],
           [26.  ],
           [35.  ],
           [21.  ],
           [22.  ],
           [19.  ],
           [34.  ],
           [18.  ],
    ...
           [21.  ],
           [20.  ],
           [22.  ],
           [25.  ],
           [25.  ],
           [73.  ],
           [22.  ],
           [28.  ],
           [29.06],
           [19.  ],
           [20.  ],
           [22.  ],
           [19.  ],
           [24.  ],
           [21.  ],
           [24.  ],
           [22.79],
           [72.  ],
           [23.  ],
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-8a3001df-17dd-45d7-bbdb-2c9c97c259a9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8a3001df-17dd-45d7-bbdb-2c9c97c259a9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2175ac59-9671-4cb5-904c-405316b4440e' class='xr-var-data-in' type='checkbox'><label for='data-2175ac59-9671-4cb5-904c-405316b4440e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;F&#x27;, &#x27;Leiden_2200&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Leiden_2200&#x27;],
           [&#x27;F&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Milwaukee_b&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Atlanta&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;ICBM&#x27;],
           [&#x27;M&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Atlanta&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
    ...
           [&#x27;F&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;F&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;Newark&#x27;],
           [&#x27;M&#x27;, &#x27;Leiden_2180&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;F&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;F&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;M&#x27;, &#x27;Munchen&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.7016 0.4511 ... -1.294 -1.16</div><input id='attrs-5850f16d-7a44-4869-9357-dc4e76170f03' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5850f16d-7a44-4869-9357-dc4e76170f03' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5a8a7171-9ccc-4f0f-83fc-a09d80fcddac' class='xr-var-data-in' type='checkbox'><label for='data-5a8a7171-9ccc-4f0f-83fc-a09d80fcddac' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.70162225,  0.45108228, -0.53539626,  0.29388233],
           [ 0.09560252,  1.21664329, -1.32660684, -0.3074337 ],
           [ 0.244679  ,  0.37958917,  0.59968993,  0.28750188],
           [ 0.28042107,  1.67183607, -0.15935201,  1.45471063],
           [-1.40121344, -1.37798141, -0.25357932, -0.95836225],
           [-0.81397524, -0.74436443,  0.34799242, -0.47493403],
           [ 0.57160882,  2.52009926,  0.27372894,  0.31866583],
           [-0.00961869, -0.67104066,  2.01518006, -0.77740609],
           [-0.91007481,  0.8159425 , -0.99965942, -0.74994733],
           [-1.23273137, -0.50207175, -0.76437823,  0.10862625],
           [ 1.33296633, -1.21489282,  0.79518393,  0.59333855],
           [-0.76348397, -0.3632876 , -1.22406449, -1.18933154],
           [ 1.19735953, -0.44641872,  0.10194602,  1.02909785],
           [ 1.69374608, -0.86756211,  1.2153382 ,  1.85707245],
           [ 1.26045026, -1.62028738,  0.75980913,  0.22212517],
           [-1.29053416, -0.92749277, -1.09769158, -1.08622092],
           [-0.55208825,  0.69857574, -0.60585674, -0.35877223],
           [-0.51134877,  0.59748936, -0.06987092,  0.64787779],
           [-0.38911232, -0.11688521,  0.85645987, -1.05405892],
           [ 1.56801446, -0.45949074, -0.84137104,  0.74471017],
    ...
           [ 0.17445929, -0.16228994, -1.09208539, -1.52449248],
           [ 2.28011385, -0.87118395,  0.74021136,  1.43757284],
           [-1.3893788 ,  0.20910281, -0.28314258, -0.66700356],
           [-0.63087978,  0.17376966, -1.47802761, -0.70411349],
           [-0.88122935,  0.72946772,  0.55984811, -0.61240111],
           [ 0.76885148,  0.25313774,  0.65712374,  0.59194952],
           [-0.91162704, -0.47487633,  0.50314931, -0.44231304],
           [ 0.09302237, -0.52912213, -1.56351687,  0.63930993],
           [ 2.19159939,  1.30561632, -0.2116823 , -0.39862223],
           [-0.78620473,  1.576848  , -0.14463102,  0.0997611 ],
           [-0.30459462,  0.38844484,  0.46917117, -0.58409444],
           [-1.51277298, -0.18081118, -0.46660057, -0.65147609],
           [-0.50244357,  0.26111862,  0.93663544,  0.02685182],
           [-0.29295236, -0.06626632, -0.21885197, -0.88218412],
           [-0.56661466,  1.15133284, -0.31615507, -0.65758914],
           [-1.44556458, -0.53545461,  0.39023864,  1.47225268],
           [-1.2645671 ,  0.34570429, -1.0190337 , -0.51270613],
           [-0.37521858,  0.30340593,  0.90688349,  0.19720961],
           [ 2.94218704, -0.20190964,  1.63342122,  2.63175641],
           [-1.02510519,  1.41445932, -1.29436637, -1.15977053]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>602.5 4.332e+03 ... 6.364e+05</div><input id='attrs-0ae3aca2-fd80-4766-b223-fabc67d52aed' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0ae3aca2-fd80-4766-b223-fabc67d52aed' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2e2011da-fc74-4e70-9e5b-5c5c205daeb8' class='xr-var-data-in' type='checkbox'><label for='data-2e2011da-fc74-4e70-9e5b-5c5c205daeb8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[6.02499877e+02, 4.33236193e+03, 1.18769509e+03, 3.59271959e+05],
            [4.21281375e+02, 2.95673695e+03, 1.59655152e+03, 4.82558820e+05],
            [5.86584804e+01, 2.62568526e+03, 1.43734596e+03, 4.26917640e+05],
            ...,
            [1.09376757e+03, 6.03029980e+03, 1.36040485e+03, 4.08785425e+05],
            [7.35319731e+02, 2.60271436e+03, 1.60336262e+03, 4.43576057e+05],
            [4.09103673e+02, 2.89259240e+03, 1.61722555e+03, 4.87740504e+05]],
    
           [[1.37202714e+03, 7.59905979e+03, 1.41150985e+03, 4.14843101e+05],
            [8.53056440e+02, 5.34531340e+03, 1.75605712e+03, 5.13839529e+05],
            [5.73080209e+02, 5.09543383e+03, 1.62027545e+03, 4.64114601e+05],
            ...,
            [1.97912356e+03, 9.81433702e+03, 1.54459222e+03, 4.53767599e+05],
            [1.10752279e+03, 5.03375722e+03, 1.76185277e+03, 4.77021361e+05],
            [8.51494319e+02, 5.33962216e+03, 1.78024206e+03, 5.20236225e+05]],
    
           [[1.91774220e+03, 1.02946473e+04, 1.53858482e+03, 4.45907038e+05],
            [1.10845319e+03, 6.84376782e+03, 1.87590285e+03, 5.38512882e+05],
            [8.60690018e+02, 6.61578632e+03, 1.73321964e+03, 4.86466591e+05],
            ...,
            [2.93040632e+03, 1.41601353e+04, 1.65314049e+03, 4.79553045e+05],
            [1.34657416e+03, 6.52778071e+03, 1.88199840e+03, 4.98310725e+05],
            [1.11214343e+03, 6.87056322e+03, 1.90770903e+03, 5.46673967e+05]],
    
           [[2.68303759e+03, 1.45898420e+04, 1.64969900e+03, 4.72285064e+05],
            [1.35029905e+03, 8.42027236e+03, 2.02413428e+03, 5.67966086e+05],
            [1.11529802e+03, 8.18244062e+03, 1.85267173e+03, 5.08496167e+05],
            ...,
            [4.55257784e+03, 2.23630644e+04, 1.75681551e+03, 5.03969654e+05],
            [1.59135936e+03, 8.05410003e+03, 2.03107368e+03, 5.20642801e+05],
            [1.35897925e+03, 8.48735976e+03, 2.06936660e+03, 5.78591454e+05]],
    
           [[4.62918131e+03, 2.76306569e+04, 1.80356263e+03, 5.07119540e+05],
            [1.71420463e+03, 1.13785953e+04, 2.32062172e+03, 6.21257829e+05],
            [1.46385904e+03, 1.10598853e+04, 2.07559759e+03, 5.44811169e+05],
            ...,
            [9.03590039e+03, 4.84560680e+04, 1.92541656e+03, 5.43818813e+05],
            [1.99292595e+03, 1.08056977e+04, 2.32890427e+03, 5.59164341e+05],
            [1.73216225e+03, 1.15609968e+04, 2.39562940e+03, 6.36400440e+05]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.104 -0.8605 ... -1.377 -1.117</div><input id='attrs-f9ebdcfb-bc11-4f2e-a9b5-ce04b341713c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f9ebdcfb-bc11-4f2e-a9b5-ce04b341713c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ccdd0584-24d7-4885-815d-0af5cc91ec50' class='xr-var-data-in' type='checkbox'><label for='data-ccdd0584-24d7-4885-815d-0af5cc91ec50' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.10411658e+00, -8.60518393e-01, -7.16784795e-01,
            -6.17653736e-01],
           [-9.71677438e-02, -1.11022013e+00, -1.40382354e+00,
            -5.02859512e-01],
           [-1.26379491e-01, -4.86884770e-01, -7.18522393e-01,
            -4.79452276e-01],
           [-4.52422793e-01, -1.93333229e+00, -5.00455434e-01,
            -1.68288890e+00],
           [-1.77582129e+00, -1.81675928e+00, -6.45761505e-01,
            -1.26822491e+00],
           [-4.19236257e-01, -7.36725304e-01, -5.88288846e-01,
            -5.35883968e-01],
           [-2.38022803e-01, -3.55809330e+00, -6.40125338e-01,
            -5.74950263e-01],
           [-1.73817820e-01, -7.34598141e-01, -2.61159334e+00,
            -7.64042342e-01],
           [-4.73905984e-01, -7.13539688e-01, -1.10285485e+00,
            -7.24913376e-01],
           [-1.32395779e+00, -8.11761680e-01, -8.46011808e-01,
            -5.51032758e-01],
    ...
           [-1.01592890e-01, -4.79814690e-01, -6.35199304e-01,
            -6.05242163e-01],
           [-1.22292032e+00, -4.50207145e-01, -6.76787810e-01,
            -6.37484674e-01],
           [-1.67351201e-01, -4.67010341e-01, -9.91646098e-01,
            -4.74594397e-01],
           [-1.92083052e-01, -4.58259135e-01, -5.80156839e-01,
            -8.47241055e-01],
           [-2.10829032e-01, -1.01189770e+00, -5.54408585e-01,
            -6.32997915e-01],
           [-1.16844431e+00, -6.26967330e-01, -5.80030112e-01,
            -1.53446235e+00],
           [-9.01672683e-01, -4.57860990e-01, -1.03169522e+00,
            -5.74647516e-01],
           [-1.05251060e+00, -8.92423114e-01, -8.59350508e-01,
            -5.68212417e-01],
           [-4.45119561e+00, -4.50278191e-01, -2.02793908e+00,
            -4.09393596e+00],
           [-6.52068347e-01, -1.38718584e+00, -1.37661021e+00,
            -1.11698385e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.182e+03 1.243e+04 ... 5.517e+05</div><input id='attrs-8fc67002-e7b8-44a5-80e8-4fd91a6c55c7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8fc67002-e7b8-44a5-80e8-4fd91a6c55c7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3341585c-8131-4de3-99de-ef1886439302' class='xr-var-data-in' type='checkbox'><label for='data-3341585c-8131-4de3-99de-ef1886439302' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[2.18162849e+03, 1.24257000e+04, 1.52577094e+03, 4.40945274e+05],
           [1.09958217e+03, 6.91444023e+03, 1.90194146e+03, 5.43262840e+05],
           [8.19299286e+02, 6.72249379e+03, 1.74414365e+03, 4.85728702e+05],
           [1.34442984e+03, 7.88406955e+03, 1.92699779e+03, 5.22806009e+05],
           [2.02215194e+03, 9.95208269e+03, 1.35457381e+03, 3.85513809e+05],
           [1.16456825e+03, 6.75141500e+03, 1.86768322e+03, 5.31218300e+05],
           [9.74606784e+02, 7.84678895e+03, 1.99596137e+03, 5.42076141e+05],
           [5.26238618e+02, 6.64682211e+03, 1.62602103e+03, 4.93795881e+05],
           [1.00262899e+03, 5.45229157e+03, 1.62926061e+03, 4.84117917e+05],
           [1.20762494e+03, 7.51936560e+03, 1.59651920e+03, 4.35830701e+05],
           [1.15705997e+03, 6.48214415e+03, 1.86134542e+03, 5.39411723e+05],
           [8.95195556e+02, 6.10607558e+03, 1.59554236e+03, 4.65099825e+05],
           [1.00493824e+03, 5.28690806e+03, 1.62684011e+03, 4.86913703e+05],
           [1.22760429e+03, 5.62224827e+03, 1.67800796e+03, 4.40692564e+05],
           [1.24788531e+03, 7.78521787e+03, 1.91218613e+03, 5.31345305e+05],
           [9.70813625e+02, 6.12232538e+03, 1.64902312e+03, 5.03636371e+05],
           [1.41305533e+03, 7.21176980e+03, 1.83792244e+03, 5.50049677e+05],
           [1.00550040e+03, 5.25089611e+03, 1.62591827e+03, 4.89191817e+05],
           [1.15863699e+03, 7.59313473e+03, 1.84191912e+03, 4.96249918e+05],
           [1.19396143e+03, 4.94706436e+03, 1.65938160e+03, 4.61382892e+05],
    ...
           [6.64632204e+02, 6.04513343e+03, 1.67040515e+03, 5.06455937e+05],
           [1.34218930e+03, 6.32100076e+03, 1.90358496e+03, 5.06994513e+05],
           [1.25912031e+03, 6.09581514e+03, 1.60821173e+03, 4.95012704e+05],
           [1.73635434e+03, 6.35750826e+03, 1.51798983e+03, 4.12698936e+05],
           [1.16425107e+03, 7.68738997e+03, 1.87666637e+03, 5.27812289e+05],
           [2.91780147e+03, 1.81579721e+04, 1.69238150e+03, 4.89905570e+05],
           [1.20333636e+03, 5.30495806e+03, 1.66954494e+03, 4.50283139e+05],
           [7.31721394e+02, 7.95489000e+03, 1.86854984e+03, 5.44102701e+05],
           [1.14274265e+03, 7.26979847e+03, 1.91415303e+03, 5.30655885e+05],
           [1.01102957e+03, 5.24034218e+03, 1.62027910e+03, 4.89223932e+05],
           [1.44924345e+03, 6.12019424e+03, 1.66610601e+03, 4.98103771e+05],
           [1.26875550e+03, 6.07267723e+03, 1.60784447e+03, 4.95799192e+05],
           [1.30360506e+03, 4.88163752e+03, 1.42820660e+03, 4.49638473e+05],
           [5.29127416e+02, 6.52956268e+03, 1.62947114e+03, 4.97693257e+05],
           [1.15886350e+03, 6.59539487e+03, 1.86504999e+03, 5.36828572e+05],
           [1.01888615e+03, 5.68529723e+03, 1.63477505e+03, 4.77680802e+05],
           [1.09147447e+03, 6.94646520e+03, 1.90237968e+03, 5.44983430e+05],
           [3.75430230e+03, 1.89196233e+04, 1.65042722e+03, 4.78066979e+05],
           [1.35020748e+03, 6.62298182e+03, 1.91345907e+03, 4.99406450e+05],
           [1.09915558e+03, 7.02040474e+03, 1.94804404e+03, 5.51675592e+05]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.03426 1.731 ... 0.591 0.9941</div><input id='attrs-8df38903-9dd1-4df9-95fb-acda260d8285' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8df38903-9dd1-4df9-95fb-acda260d8285' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1bf45993-ea58-4dde-81ee-b1a6d47fed1e' class='xr-var-data-in' type='checkbox'><label for='data-1bf45993-ea58-4dde-81ee-b1a6d47fed1e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[3.42592593e-02, 1.73136841e+00, 4.60967210e-01, 6.58301830e-01,
            3.15769725e-01, 6.12994158e-01, 5.00607146e-01, 4.21510416e-15,
            6.84230275e-01, 9.68022156e-01],
           [4.25925926e-02, 3.89230185e+00, 3.15209849e-01, 1.11941089e+00,
            2.00143243e-01, 9.08482999e-01, 2.76307172e-01, 3.82579165e-05,
            7.99856757e-01, 9.51151796e-01],
           [1.66666667e-02, 1.49314736e+00, 2.84366630e-01, 1.09690254e+00,
            2.89385970e-01, 8.11815414e-01, 5.13803008e-01, 5.97484703e-16,
            7.10614030e-01, 9.91162660e-01],
           [2.31481481e-02, 2.19836635e+00, 3.63966176e-01, 9.91960143e-01,
            4.09021607e-01, 7.21805097e-01, 6.38222651e-01, 4.16394768e-26,
            5.90978393e-01, 9.94067486e-01]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-bdefb7d7-4f41-444c-ac35-135a9bff0cd6' class='xr-section-summary-in' type='checkbox'  ><label for='section-bdefb7d7-4f41-444c-ac35-135a9bff0cd6' class='xr-section-summary' >Indexes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-3fd0d6a5-a71d-4fd4-8614-decaf607095f' class='xr-index-data-in' type='checkbox'/><label for='index-3fd0d6a5-a71d-4fd4-8614-decaf607095f' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([ 756,  769,  692,  616,   35,  164,  680,  331,  299,  727,
           ...
             27,  959,   29,  346,  304,  264,  798,  751,  470, 1043],
          dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;, length=216))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-e9949517-9e18-4366-8500-64d313c7f2d9' class='xr-index-data-in' type='checkbox'/><label for='index-e9949517-9e18-4366-8500-64d313c7f2d9' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-3014f413-cf49-496b-b75d-1fc29b73c561' class='xr-index-data-in' type='checkbox'/><label for='index-3014f413-cf49-496b-b75d-1fc29b73c561' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-949bd520-18c1-410e-b581-ce5da993b9c4' class='xr-index-data-in' type='checkbox'/><label for='index-949bd520-18c1-410e-b581-ce5da993b9c4' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>centile</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-3ae4347d-fca0-4502-ba31-46d82cc3d1a0' class='xr-index-data-in' type='checkbox'/><label for='index-3ae4347d-fca0-4502-ba31-46d82cc3d1a0' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0.05, 0.25, 0.5, 0.75, 0.95], dtype=&#x27;float64&#x27;, name=&#x27;centile&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-65fa905f-e5cb-4a43-aa46-169038f04dff' class='xr-index-data-in' type='checkbox'/><label for='index-65fa905f-e5cb-4a43-aa46-169038f04dff' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;,
           &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-2027ffa1-9b35-4ddf-bd75-200f54fa3f91' class='xr-section-summary-in' type='checkbox'  checked><label for='section-2027ffa1-9b35-4ddf-bd75-200f54fa3f91' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{&#x27;sex&#x27;: [&#x27;F&#x27;, &#x27;M&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;F&#x27;: 589, &#x27;M&#x27;: 489}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;F&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}, &#x27;M&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd><dt><span>covariate_ranges :</span></dt><dd>{&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd></dl></div></li></ul></div></div>



Harmonize
~~~~~~~~~

.. code:: ipython3

    # Harmonizing is also easy:
    reference_batch_effect = {
        "site": "Beijing_Zang",
        "sex": "M",
    }  # Set a pseudo-batch effect. I.e., this means 'pretend that all data was from this site and sex'
    
    model.harmonize(test, reference_batch_effect=reference_batch_effect)  # <- easy
    
    plt.style.use("seaborn-v0_8")
    df = test.to_dataframe()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    sns.scatterplot(data=df, x=("X", "age"), y=("Y", feature_to_plot), hue=("batch_effects", "site"), ax=ax[0])
    sns.scatterplot(data=df, x=("X", "age"), y=("Y_harmonized", feature_to_plot), hue=("batch_effects", "site"), ax=ax[1])
    ax[0].title.set_text("Unharmonized")
    ax[1].title.set_text("Harmonized")
    ax[0].legend([], [])
    ax[1].legend([], [])
    ax[0].set_xlabel("Age")
    ax[0].set_ylabel(feature_to_plot)
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel(feature_to_plot)
    plt.tight_layout()
    plt.show()


.. parsed-literal::

    Process: 27518 - 2025-06-24 12:16:31 - Harmonizing data on 4 response variables.
    Process: 27518 - 2025-06-24 12:16:31 - Harmonizing data for Right-Lateral-Ventricle.
    Process: 27518 - 2025-06-24 12:16:31 - Harmonizing data for CortexVol.
    Process: 27518 - 2025-06-24 12:16:31 - Harmonizing data for WM-hypointensities.
    Process: 27518 - 2025-06-24 12:16:31 - Harmonizing data for Right-Amygdala.



.. image:: 02_BLR_files/02_BLR_27_1.png


Synthesize
~~~~~~~~~~

Our models can synthesize new data that follows the learned
distribution.

Not only the distribution of the response variables given a covariate is
learned, but also the ranges of the covariates *within* each batch
effect. So if we have fitted a model on a number of sites, and subjects
from A have an age between 10 and 20, then the synthesized
pseudo-subjects from site A will also have an age between 10 and 20.

Not only that, but we also sample the batch effects in the frequency of
the batch effects in the original data. So if the train data contained
twice as many subjects from site A as site B, then the synthesized
pseudo-subjects will also have twice as many subjects from site A as
site B.

.. code:: ipython3

    # Generate 10000 synthetic datapoints from scratch
    synthetic_data = model.synthesize(covariate_range_per_batch_effect=True, n_samples=10000)  # <- also easy
    plot_centiles(
        model,
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=synthetic_data,
        show_other_data=True,
        harmonize_data=True,
        show_legend=True,
    )



.. parsed-literal::

    Process: 27518 - 2025-06-24 12:16:33 - Dataset "synthesized" created.
        - 10000 observations
        - 10000 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        
    Process: 27518 - 2025-06-24 12:16:33 - Synthesizing data for 4 response variables.
    Process: 27518 - 2025-06-24 12:16:33 - Synthesizing data for WM-hypointensities.
    Process: 27518 - 2025-06-24 12:16:34 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 27518 - 2025-06-24 12:16:34 - Synthesizing data for Right-Amygdala.
    Process: 27518 - 2025-06-24 12:16:35 - Synthesizing data for CortexVol.


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 27518 - 2025-06-24 12:16:36 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)


.. parsed-literal::

    Process: 27518 - 2025-06-24 12:16:36 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 27518 - 2025-06-24 12:16:36 - Computing centiles for 4 response variables.
    Process: 27518 - 2025-06-24 12:16:36 - Computing centiles for Right-Lateral-Ventricle.
    Process: 27518 - 2025-06-24 12:16:36 - Computing centiles for CortexVol.
    Process: 27518 - 2025-06-24 12:16:36 - Computing centiles for WM-hypointensities.
    Process: 27518 - 2025-06-24 12:16:36 - Computing centiles for Right-Amygdala.
    Process: 27518 - 2025-06-24 12:16:38 - Harmonizing data on 4 response variables.
    Process: 27518 - 2025-06-24 12:16:38 - Harmonizing data for Right-Lateral-Ventricle.
    Process: 27518 - 2025-06-24 12:16:39 - Harmonizing data for CortexVol.
    Process: 27518 - 2025-06-24 12:16:40 - Harmonizing data for WM-hypointensities.
    Process: 27518 - 2025-06-24 12:16:41 - Harmonizing data for Right-Amygdala.



.. image:: 02_BLR_files/02_BLR_29_3.png



.. image:: 02_BLR_files/02_BLR_29_4.png



.. image:: 02_BLR_files/02_BLR_29_5.png



.. image:: 02_BLR_files/02_BLR_29_6.png


.. code:: ipython3

    # Synthesize new Y data for existing X data
    new_test_data = test.copy()
    
    # Remove the Y data, this way we will synthesize new Y data for the existing X data
    if hasattr(new_test_data, "Y"):
        del new_test_data["Y"]
    
    synthetic = model.synthesize(new_test_data)  # <- will fill in the missing Y data
    plot_centiles(
        model,
        centiles=[0.05, 0.5, 0.95],  # Plot arbitrary centiles
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=train,  # Scatter the train data points
        batch_effects="all",  # You can set this to "all" to show all batch effects
        show_other_data=True,  # Show data points that do not match any batch effects
        harmonize_data=True,  # Set this to False to see the difference
        show_legend=False,  # Don't show the legend because it crowds the plot
    )


.. parsed-literal::

    Process: 27518 - 2025-06-24 12:16:42 - Synthesizing data for 4 response variables.
    Process: 27518 - 2025-06-24 12:16:42 - Synthesizing data for WM-hypointensities.
    Process: 27518 - 2025-06-24 12:16:43 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 27518 - 2025-06-24 12:16:43 - Synthesizing data for Right-Amygdala.
    Process: 27518 - 2025-06-24 12:16:43 - Synthesizing data for CortexVol.
    Process: 27518 - 2025-06-24 12:16:43 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 27518 - 2025-06-24 12:16:43 - Computing centiles for 4 response variables.
    Process: 27518 - 2025-06-24 12:16:43 - Computing centiles for Right-Lateral-Ventricle.
    Process: 27518 - 2025-06-24 12:16:43 - Computing centiles for CortexVol.
    Process: 27518 - 2025-06-24 12:16:43 - Computing centiles for WM-hypointensities.
    Process: 27518 - 2025-06-24 12:16:43 - Computing centiles for Right-Amygdala.


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 27518 - 2025-06-24 12:16:43 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)


.. parsed-literal::

    Process: 27518 - 2025-06-24 12:16:43 - Harmonizing data on 4 response variables.
    Process: 27518 - 2025-06-24 12:16:43 - Harmonizing data for Right-Lateral-Ventricle.
    Process: 27518 - 2025-06-24 12:16:43 - Harmonizing data for CortexVol.
    Process: 27518 - 2025-06-24 12:16:43 - Harmonizing data for WM-hypointensities.
    Process: 27518 - 2025-06-24 12:16:43 - Harmonizing data for Right-Amygdala.



.. image:: 02_BLR_files/02_BLR_30_3.png



.. image:: 02_BLR_files/02_BLR_30_4.png



.. image:: 02_BLR_files/02_BLR_30_5.png



.. image:: 02_BLR_files/02_BLR_30_6.png


Next steps
----------

Please see the other tutorials for more examples, and we also recommend
you to read the documentation! As this toolkit is still in development,
the documentation may not be up to date. If you find any issues, please
let us know!

Also, feel free to contact us on Github if you have any questions or
suggestions.

Have fun modeling!
