Transfering and extending normative models
==========================================

Welcome to this tutorial notebook that will go through the transfering
and extending of existing models on new data.

Transfer and Extend are both useful for when you have only a small
dataset to your disposal, but you still want to derive a well-calibrated
model from that. In both cases, a reference model is used in tandem with
the small dataset to derive a new model that is better than a model that
would be trained solely on the small dataset.

For transfer, the new model will only be able to handle data from the
batches in the small dataset; a small model is derived from a large
reference model.

For extend, the new model will be able to handle data from batches in
the reference training set, as well as the batches in the new small
dataset; a larger reference model is derived from a large reference
model.

Imports
~~~~~~~

.. code:: ipython3

    import warnings
    import logging
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    from pcntoolkit import (
        HBR,
        BsplineBasisFunction,
        NormativeModel,
        NormData,
        load_fcon1000,
        SHASHbLikelihood,
        NormalLikelihood,
        BetaLikelihood,
        make_prior,
        plot_centiles,
        plot_qq,
        plot_ridge,
    )
    
    import numpy as np
    import pcntoolkit.util.output
    import seaborn as sns
    import os
    
    sns.set_style("darkgrid")
    
    # Suppress some annoying warnings and logs
    pymc_logger = logging.getLogger("pymc")
    
    pymc_logger.setLevel(logging.WARNING)
    pymc_logger.propagate = False
    
    warnings.simplefilter(action="ignore", category=FutureWarning)
    pd.options.mode.chained_assignment = None  # default='warn'
    pcntoolkit.util.output.Output.set_show_messages(False)

Load data
---------

First we download a small example dataset from github.

.. code:: ipython3

    # Download the dataset
    norm_data: NormData = load_fcon1000()
    features_to_model = [
        "WM-hypointensities",
        # "Right-Lateral-Ventricle",
        # "Right-Amygdala",
        # "CortexVol",
    ]
    # Select only a few features
    norm_data = norm_data.sel({"response_vars": features_to_model})
    
    # Leave two sites out for doing transfer and extend later
    transfer_sites = ["Milwaukee_b", "Oulu"]
    transfer_data, fit_data = norm_data.batch_effects_split({"site": transfer_sites}, names=("transfer", "fit"))
    
    # Split into train and test sets
    train, test = fit_data.train_test_split()
    transfer_train, transfer_test = transfer_data.train_test_split()


.. code:: ipython3

    # Visualize the data
    feature_to_plot = features_to_model[0]
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



.. image:: 06_transfer_extend_files/06_transfer_extend_6_0.png


Creating a Normative model
--------------------------

We will use the same HBR model that we used in the tutorial “Normative
Modelling: Hierarchical Bayesian Regression with Normal likelihood”.
Please read that tutorial for an extensive coverage of the
configuration.

.. code:: ipython3

    mu = make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 10.0)),
        intercept=make_prior(
            random=True,
            mu=make_prior(dist_name="Normal", dist_params=(0.0, 1.0)),
            sigma=make_prior(dist_name="Normal", dist_params=(0.0, 1.0), mapping="softplus", mapping_params=(0.0, 3.0)),
        ),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    sigma = make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        mapping="softplus",
        mapping_params=(0.0, 3.0),
    )
    
    likelihood = NormalLikelihood(mu, sigma)
    
    template_hbr = HBR(
        name="template",
        cores=16,
        progressbar=True,
        draws=1500,
        tune=500,
        chains=4,
        nuts_sampler="nutpie",
        likelihood=likelihood,
    )
    model = NormativeModel(
        template_regression_model=template_hbr,
        savemodel=True,
        evaluate_model=True,
        saveresults=True,
        saveplots=False,
        save_dir="resources/hbr/save_dir",
        inscaler="standardize",
        outscaler="standardize",
    )

.. code:: ipython3

    test = model.fit_predict(train, test)



.. raw:: html

    
    <style>
        :root {
            --column-width-1: 40%; /* Progress column width */
            --column-width-2: 15%; /* Chain column width */
            --column-width-3: 15%; /* Divergences column width */
            --column-width-4: 15%; /* Step Size column width */
            --column-width-5: 15%; /* Gradients/Draw column width */
        }
    
        .nutpie {
            max-width: 800px;
            margin: 10px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            //color: #333;
            //background-color: #fff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            font-size: 14px; /* Smaller font size for a more compact look */
        }
        .nutpie table {
            width: 100%;
            border-collapse: collapse; /* Remove any extra space between borders */
        }
        .nutpie th, .nutpie td {
            padding: 8px 10px; /* Reduce padding to make table more compact */
            text-align: left;
            border-bottom: 1px solid #888;
        }
        .nutpie th {
            //background-color: #f0f0f0;
        }
    
        .nutpie th:nth-child(1) { width: var(--column-width-1); }
        .nutpie th:nth-child(2) { width: var(--column-width-2); }
        .nutpie th:nth-child(3) { width: var(--column-width-3); }
        .nutpie th:nth-child(4) { width: var(--column-width-4); }
        .nutpie th:nth-child(5) { width: var(--column-width-5); }
    
        .nutpie progress {
            width: 100%;
            height: 15px; /* Smaller progress bars */
            border-radius: 5px;
        }
        progress::-webkit-progress-bar {
            background-color: #eee;
            border-radius: 5px;
        }
        progress::-webkit-progress-value {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        progress::-moz-progress-bar {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        .nutpie .progress-cell {
            width: 100%;
        }
    
        .nutpie p strong { font-size: 16px; font-weight: bold; }
    
        @media (prefers-color-scheme: dark) {
            .nutpie {
                //color: #ddd;
                //background-color: #1e1e1e;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
            .nutpie table, .nutpie th, .nutpie td {
                border-color: #555;
                color: #ccc;
            }
            .nutpie th {
                background-color: #2a2a2a;
            }
            .nutpie progress::-webkit-progress-bar {
                background-color: #444;
            }
            .nutpie progress::-webkit-progress-value {
                background-color: #3178c6;
            }
            .nutpie progress::-moz-progress-bar {
                background-color: #3178c6;
            }
        }
    </style>




.. raw:: html

    
    <div class="nutpie">
        <p><strong>Sampler Progress</strong></p>
        <p>Total Chains: <span id="total-chains">4</span></p>
        <p>Active Chains: <span id="active-chains">0</span></p>
        <p>
            Finished Chains:
            <span id="active-chains">4</span>
        </p>
        <p>Sampling for now</p>
        <p>
            Estimated Time to Completion:
            <span id="eta">now</span>
        </p>
    
        <progress
            id="total-progress-bar"
            max="8000"
            value="8000">
        </progress>
        <table>
            <thead>
                <tr>
                    <th>Progress</th>
                    <th>Draws</th>
                    <th>Divergences</th>
                    <th>Step Size</th>
                    <th>Gradients/Draw</th>
                </tr>
            </thead>
            <tbody id="chain-details">
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>14</td>
                        <td>0.13</td>
                        <td>63</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>57</td>
                        <td>0.12</td>
                        <td>31</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>13</td>
                        <td>0.12</td>
                        <td>255</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>11</td>
                        <td>0.12</td>
                        <td>31</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>





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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 44kB
    Dimensions:            (observations: 186, response_vars: 1, covariates: 1,
                            batch_effect_dims: 2, centile: 5, statistic: 10)
    Coordinates:
      * observations       (observations) int64 1kB 515 441 1029 64 ... 640 648 635
      * response_vars      (response_vars) &lt;U18 72B &#x27;WM-hypointensities&#x27;
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) &lt;U8 320B &#x27;MACE&#x27; &#x27;MAPE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;
    Data variables:
        subjects           (observations) object 1kB &#x27;Cambridge_Buckner_sub83409&#x27;...
        Y                  (observations, response_vars) float64 1kB 1.113e+03 .....
        X                  (observations, covariates) float64 1kB 18.0 20.0 ... 20.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 25kB &#x27;F&#x27; ... &#x27;I...
        Z                  (observations, response_vars) float64 1kB -0.2964 ... ...
        centiles           (centile, observations, response_vars) float64 7kB 597...
        logp               (observations, response_vars) float64 1kB -0.2175 ... ...
        Yhat               (observations, response_vars) float64 1kB 1.226e+03 .....
        statistics         (response_vars, statistic) float64 80B 0.03011 ... 0.938
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fit_test
        unique_batch_effects:           {&#x27;sex&#x27;: [&#x27;F&#x27;, &#x27;M&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;...
        batch_effect_counts:            {&#x27;sex&#x27;: {&#x27;F&#x27;: 493, &#x27;M&#x27;: 437}, &#x27;site&#x27;: {&#x27;A...
        batch_effect_covariate_ranges:  {&#x27;sex&#x27;: {&#x27;F&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;...
        covariate_ranges:               {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-cbc6e844-2b1b-4ead-a5e1-cde43b213393' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-cbc6e844-2b1b-4ead-a5e1-cde43b213393' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 186</li><li><span class='xr-has-index'>response_vars</span>: 1</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>centile</span>: 5</li><li><span class='xr-has-index'>statistic</span>: 10</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-1b041ce6-e052-4c12-8fb3-61c886cd4bba' class='xr-section-summary-in' type='checkbox'  checked><label for='section-1b041ce6-e052-4c12-8fb3-61c886cd4bba' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>515 441 1029 64 ... 549 640 648 635</div><input id='attrs-5abfb6bc-8e5a-40ae-92f5-d6afc9284960' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5abfb6bc-8e5a-40ae-92f5-d6afc9284960' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b3ab7f88-5ba0-416f-8fb5-0d029c6e05dc' class='xr-var-data-in' type='checkbox'><label for='data-b3ab7f88-5ba0-416f-8fb5-0d029c6e05dc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 515,  441, 1029,   64,  654,   14,  562,  599, 1003,  293,  150,  876,
            219,  108,  392,  682,  142,  350,  825,  343,  416,   51,  761,  201,
            354,  302,  838,  744,  465, 1049, 1066,  777,  346,  846,  361,  621,
           1055,  405,  314,  209,  474,  446,  626,  857,  504,  462,  487,  105,
            574, 1007,  275,  449,  259,   73,  656,  989,  788,  810,  564, 1012,
            548,  396,  841,   47,  502,  676,  390,  692,  364,  779,  813,  204,
            378,  678, 1069,  318,  109,   26, 1043, 1057,  435,  178,  106,  339,
            120,   11,  748,  783,  468,   22,  198,  636,  591,  521,   52,  476,
            247,   60,   39,  593,  795,   87,  311, 1041,   88,  394, 1017,  856,
            873,  423, 1025,  660,  669,  833,   78,  457,  205,  854,  279,  269,
            124,  553,  331,  545,  133,  872,  527,  183,  587,   12,   79,  766,
              4,  518,  242, 1024,  313,  816,  212,  803,  792,  321,  225,  220,
            628,  582,  181,  264,  437,  426, 1054,   33,  679,  129,  517, 1006,
             86,  412,  274,  513,  749,   57,  147,  143,  186,  413,  308,  867,
            282,  499,  536,  557,  324,  185,  646, 1046,  375,  320,  334,  804,
            539,  611,  549,  640,  648,  635])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U18</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27;</div><input id='attrs-cba5eb9f-3092-4030-975a-9f15209efc01' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cba5eb9f-3092-4030-975a-9f15209efc01' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d2acbf80-ef6a-465c-821d-c530991e8aae' class='xr-var-data-in' type='checkbox'><label for='data-d2acbf80-ef6a-465c-821d-c530991e8aae' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;], dtype=&#x27;&lt;U18&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-4f42251d-ede8-49aa-b4bd-2a19bbb1ea4e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4f42251d-ede8-49aa-b4bd-2a19bbb1ea4e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-41fc0094-67bd-4184-ab19-08a5fba9a021' class='xr-var-data-in' type='checkbox'><label for='data-41fc0094-67bd-4184-ab19-08a5fba9a021' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-fef5184f-dce9-4ef9-829d-5f22c57b1d87' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fef5184f-dce9-4ef9-829d-5f22c57b1d87' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1334dd97-1af8-4105-9859-4c1bf7895836' class='xr-var-data-in' type='checkbox'><label for='data-1334dd97-1af8-4105-9859-4c1bf7895836' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-74af6674-e002-42fa-a555-627f129af391' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-74af6674-e002-42fa-a555-627f129af391' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1248d9f4-1ae8-4f22-8013-0d743ef8db69' class='xr-var-data-in' type='checkbox'><label for='data-1248d9f4-1ae8-4f22-8013-0d743ef8db69' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;MACE&#x27; &#x27;MAPE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-bb955b19-a8ea-41ec-8fde-c492985cd289' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bb955b19-a8ea-41ec-8fde-c492985cd289' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c1e6524a-41fb-4787-a5a2-b9c4b19c1401' class='xr-var-data-in' type='checkbox'><label for='data-c1e6524a-41fb-4787-a5a2-b9c4b19c1401' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;,
           &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-67fe2614-960d-4e68-9683-e5db6bd66dac' class='xr-section-summary-in' type='checkbox'  checked><label for='section-67fe2614-960d-4e68-9683-e5db6bd66dac' class='xr-section-summary' >Data variables: <span>(9)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subjects</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Cambridge_Buckner_sub83409&#x27; ......</div><input id='attrs-590c1105-55bd-4d90-81e4-d6588034d2b6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-590c1105-55bd-4d90-81e4-d6588034d2b6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c74df862-3252-47af-9c5b-7c2c635c0c0b' class='xr-var-data-in' type='checkbox'><label for='data-c74df862-3252-47af-9c5b-7c2c635c0c0b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Cambridge_Buckner_sub83409&#x27;, &#x27;Cambridge_Buckner_sub47278&#x27;,
           &#x27;Queensland_sub04117&#x27;, &#x27;Atlanta_sub24972&#x27;, &#x27;ICBM_sub92028&#x27;,
           &#x27;AnnArbor_a_sub47659&#x27;, &#x27;Cleveland_sub34189&#x27;, &#x27;ICBM_sub28422&#x27;,
           &#x27;Oxford_sub76621&#x27;, &#x27;Beijing_Zang_sub83728&#x27;,
           &#x27;Beijing_Zang_sub12220&#x27;, &#x27;Newark_sub59397&#x27;,
           &#x27;Beijing_Zang_sub46058&#x27;, &#x27;Bangor_sub01903&#x27;,
           &#x27;Cambridge_Buckner_sub16846&#x27;, &#x27;Leiden_2200_sub36743&#x27;,
           &#x27;Beijing_Zang_sub08816&#x27;, &#x27;Berlin_Margulies_sub97162&#x27;,
           &#x27;NewYork_a_sub84224&#x27;, &#x27;Berlin_Margulies_sub75506&#x27;,
           &#x27;Cambridge_Buckner_sub29425&#x27;, &#x27;AnnArbor_b_sub90127&#x27;,
           &#x27;NewYork_a_sub07578&#x27;, &#x27;Beijing_Zang_sub35309&#x27;,
           &#x27;Cambridge_Buckner_sub02591&#x27;, &#x27;Beijing_Zang_sub89088&#x27;,
           &#x27;NewYork_a_sub98076&#x27;, &#x27;Munchen_sub26670&#x27;,
           &#x27;Cambridge_Buckner_sub58360&#x27;, &#x27;SaintLouis_sub03345&#x27;,
           &#x27;SaintLouis_sub73471&#x27;, &#x27;NewYork_a_sub29216&#x27;,
           &#x27;Berlin_Margulies_sub86111&#x27;, &#x27;NewYork_a_ADHD_sub17109&#x27;,
           &#x27;Cambridge_Buckner_sub05453&#x27;, &#x27;ICBM_sub51677&#x27;,
           &#x27;SaintLouis_sub35127&#x27;, &#x27;Cambridge_Buckner_sub24670&#x27;,
           &#x27;Beijing_Zang_sub92799&#x27;, &#x27;Beijing_Zang_sub40037&#x27;,
           &#x27;Cambridge_Buckner_sub61209&#x27;, &#x27;Cambridge_Buckner_sub50454&#x27;,
    ...
           &#x27;Beijing_Zang_sub49782&#x27;, &#x27;Beijing_Zang_sub46259&#x27;, &#x27;ICBM_sub59589&#x27;,
           &#x27;ICBM_sub02382&#x27;, &#x27;Beijing_Zang_sub28792&#x27;, &#x27;Beijing_Zang_sub68012&#x27;,
           &#x27;Cambridge_Buckner_sub45354&#x27;, &#x27;Cambridge_Buckner_sub39142&#x27;,
           &#x27;SaintLouis_sub28304&#x27;, &#x27;AnnArbor_b_sub42616&#x27;,
           &#x27;Leiden_2200_sub13537&#x27;, &#x27;Beijing_Zang_sub01244&#x27;,
           &#x27;Cambridge_Buckner_sub84064&#x27;, &#x27;Oxford_sub82071&#x27;,
           &#x27;Baltimore_sub23750&#x27;, &#x27;Cambridge_Buckner_sub27230&#x27;,
           &#x27;Beijing_Zang_sub73421&#x27;, &#x27;Cambridge_Buckner_sub82213&#x27;,
           &#x27;Munchen_sub50162&#x27;, &#x27;Atlanta_sub00368&#x27;, &#x27;Beijing_Zang_sub10973&#x27;,
           &#x27;Beijing_Zang_sub08992&#x27;, &#x27;Beijing_Zang_sub29785&#x27;,
           &#x27;Cambridge_Buckner_sub27613&#x27;, &#x27;Beijing_Zang_sub91399&#x27;,
           &#x27;Newark_sub36023&#x27;, &#x27;Beijing_Zang_sub80569&#x27;,
           &#x27;Cambridge_Buckner_sub76631&#x27;, &#x27;Cambridge_Buckner_sub92440&#x27;,
           &#x27;Cleveland_sub20003&#x27;, &#x27;Beijing_Zang_sub98617&#x27;,
           &#x27;Beijing_Zang_sub29590&#x27;, &#x27;ICBM_sub82228&#x27;, &#x27;Queensland_sub93238&#x27;,
           &#x27;Cambridge_Buckner_sub10268&#x27;, &#x27;Beijing_Zang_sub95755&#x27;,
           &#x27;Berlin_Margulies_sub33248&#x27;, &#x27;NewYork_a_sub54887&#x27;,
           &#x27;Cambridge_Buckner_sub93609&#x27;, &#x27;ICBM_sub40217&#x27;,
           &#x27;Cleveland_sub02480&#x27;, &#x27;ICBM_sub73490&#x27;, &#x27;ICBM_sub85442&#x27;,
           &#x27;ICBM_sub66794&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.113e+03 1.175e+03 ... 460.3</div><input id='attrs-a94ff0e3-606e-4812-8d79-27b0d5b2389c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a94ff0e3-606e-4812-8d79-27b0d5b2389c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e3b14734-ed2a-4b26-8aa8-be2c09106109' class='xr-var-data-in' type='checkbox'><label for='data-e3b14734-ed2a-4b26-8aa8-be2c09106109' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1113.3],
           [1175. ],
           [1198.1],
           [1030.8],
           [ 549.8],
           [2060. ],
           [1751.2],
           [3902.4],
           [ 969.7],
           [ 797.1],
           [1932.7],
           [2075. ],
           [ 890.5],
           [1161.7],
           [1439.4],
           [ 598.6],
           [1186.6],
           [1300.8],
           [ 539.1],
           [ 823.9],
    ...
           [ 618.9],
           [1586.7],
           [ 705.6],
           [1426.2],
           [1108.4],
           [1487.7],
           [1289.9],
           [1171.5],
           [ 833.8],
           [1381. ],
           [2281.3],
           [1547.8],
           [ 503.1],
           [1006.2],
           [1220.9],
           [3017.7],
           [2829.9],
           [ 759.8],
           [1255.3],
           [ 460.3]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>18.0 20.0 26.0 ... 68.0 43.0 20.0</div><input id='attrs-e40144b1-dedb-4617-a6cb-aa4fe9630a63' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e40144b1-dedb-4617-a6cb-aa4fe9630a63' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8c9065fc-6c89-4fc3-8f4d-a941f105637f' class='xr-var-data-in' type='checkbox'><label for='data-8c9065fc-6c89-4fc3-8f4d-a941f105637f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[18.  ],
           [20.  ],
           [26.  ],
           [24.  ],
           [56.  ],
           [33.13],
           [55.  ],
           [42.  ],
           [27.  ],
           [21.  ],
           [22.  ],
           [24.  ],
           [25.  ],
           [21.  ],
           [29.  ],
           [25.  ],
           [21.  ],
           [28.  ],
           [27.34],
           [28.  ],
    ...
           [21.  ],
           [21.  ],
           [22.  ],
           [19.  ],
           [24.  ],
           [39.  ],
           [20.  ],
           [21.  ],
           [19.  ],
           [21.  ],
           [23.  ],
           [26.  ],
           [28.  ],
           [27.43],
           [20.  ],
           [54.  ],
           [56.  ],
           [68.  ],
           [43.  ],
           [20.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Cambridge_Buckner&#x27; ... &#x27;ICBM&#x27;</div><input id='attrs-cc0dc3e5-52ec-43e9-a7ca-68ef7686deb4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cc0dc3e5-52ec-43e9-a7ca-68ef7686deb4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2ceb4aa6-b01d-4eec-8788-f4ea49c0b86a' class='xr-var-data-in' type='checkbox'><label for='data-2ceb4aa6-b01d-4eec-8788-f4ea49c0b86a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;F&#x27;, &#x27;Queensland&#x27;],
           [&#x27;F&#x27;, &#x27;Atlanta&#x27;],
           [&#x27;F&#x27;, &#x27;ICBM&#x27;],
           [&#x27;M&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;M&#x27;, &#x27;Cleveland&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;M&#x27;, &#x27;Oxford&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Newark&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Bangor&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Leiden_2200&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;F&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;M&#x27;, &#x27;Berlin_Margulies&#x27;],
    ...
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Newark&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Cleveland&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;Queensland&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;F&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;Cleveland&#x27;],
           [&#x27;F&#x27;, &#x27;ICBM&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;ICBM&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.2964 -0.5291 ... -0.2018 -1.528</div><input id='attrs-0343fac2-f2fa-4a75-8029-1566a1b407bd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0343fac2-f2fa-4a75-8029-1566a1b407bd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-62602fcf-8ebd-4960-9899-9c2f1f2a626d' class='xr-var-data-in' type='checkbox'><label for='data-62602fcf-8ebd-4960-9899-9c2f1f2a626d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-2.96375881e-01],
           [-5.29087919e-01],
           [ 5.74962834e-01],
           [ 3.28984336e-01],
           [-7.62447589e-01],
           [ 8.47029060e-01],
           [ 3.55520273e-01],
           [ 4.37700052e+00],
           [-3.93310312e-01],
           [-1.02777059e+00],
           [ 2.67003766e+00],
           [ 1.50062103e+00],
           [-7.23840774e-01],
           [-8.71137399e-01],
           [ 2.00592427e-01],
           [-1.26589202e+00],
           [ 5.10611588e-01],
           [ 1.52173156e+00],
           [-1.18075155e+00],
           [ 2.14756918e-01],
    ...
           [-1.53247658e+00],
           [-3.80205460e-01],
           [-1.28214319e+00],
           [ 1.45293232e-01],
           [-6.73541799e-01],
           [ 3.51786311e-01],
           [ 7.69012009e-01],
           [ 4.67844694e-01],
           [-4.96959524e-01],
           [ 1.05355755e+00],
           [ 2.68766003e+00],
           [ 1.56996158e+00],
           [-2.43034487e-01],
           [ 1.12426059e-01],
           [ 2.62385525e-02],
           [ 1.76514607e+00],
           [ 1.58986035e+00],
           [-8.73439510e-01],
           [-2.01765013e-01],
           [-1.52833032e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>597.7 773.3 ... 2.375e+03 1.602e+03</div><input id='attrs-9a9533cf-6740-4f77-a4c6-e79e43f53a68' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9a9533cf-6740-4f77-a4c6-e79e43f53a68' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d7acc834-37d7-4ba2-ad5c-f23a5f555ad7' class='xr-var-data-in' type='checkbox'><label for='data-d7acc834-37d7-4ba2-ad5c-f23a5f555ad7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 5.97694549e+02],
            [ 7.73261949e+02],
            [ 4.11700919e+02],
            [ 3.42717562e+02],
            [-3.27184074e+02],
            [ 1.02056415e+03],
            [-1.43381393e+02],
            [ 4.04795147e+02],
            [ 5.19578824e+02],
            [ 5.78570564e+02],
            [ 4.24582717e+02],
            [ 9.78590321e+02],
            [ 5.66866628e+02],
            [ 8.87845127e+02],
            [ 7.50810013e+02],
            [ 4.64924409e+02],
            [ 4.24888596e+02],
            [ 1.44183592e+02],
            [ 3.70397246e+02],
            [ 1.44183592e+02],
    ...
            [ 1.74139133e+03],
            [ 2.30224349e+03],
            [ 1.72879454e+03],
            [ 1.97997418e+03],
            [ 1.91661309e+03],
            [ 2.15676296e+03],
            [ 1.60541209e+03],
            [ 1.58770936e+03],
            [ 1.62459256e+03],
            [ 1.59035337e+03],
            [ 1.91912865e+03],
            [ 1.57534954e+03],
            [ 1.19350955e+03],
            [ 1.56091896e+03],
            [ 1.80330560e+03],
            [ 2.92711481e+03],
            [ 2.90594706e+03],
            [ 5.04902501e+03],
            [ 2.37535652e+03],
            [ 1.60160594e+03]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.2175 -0.2537 ... -0.6624 -1.296</div><input id='attrs-903d19ee-589e-4c0a-a0b9-7f17a42d8b0c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-903d19ee-589e-4c0a-a0b9-7f17a42d8b0c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0f4af273-f416-481d-80ad-78c370939e30' class='xr-var-data-in' type='checkbox'><label for='data-0f4af273-f416-481d-80ad-78c370939e30' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -0.21752097],
           [ -0.25367138],
           [ -0.29395196],
           [ -0.15921238],
           [ -1.41915806],
           [ -0.64484384],
           [ -1.15674152],
           [-10.2337426 ],
           [ -0.21413926],
           [ -0.62442378],
           [ -3.65264038],
           [ -1.23911348],
           [ -0.35348255],
           [ -0.50304741],
           [ -0.17505828],
           [ -0.92082162],
           [ -0.22533296],
           [ -1.31529363],
           [ -0.82420224],
           [ -0.17750733],
    ...
           [ -1.27123244],
           [ -0.19811243],
           [ -0.90821067],
           [ -0.14997614],
           [ -0.31124808],
           [ -0.55485134],
           [ -0.40871674],
           [ -0.20438864],
           [ -0.2768706 ],
           [ -0.68356049],
           [ -3.6973344 ],
           [ -1.33606236],
           [ -0.18293072],
           [ -0.13322078],
           [ -0.11290264],
           [ -2.63025706],
           [ -2.41322885],
           [ -2.06438271],
           [ -0.66242818],
           [ -1.29639752]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.226e+03 1.365e+03 ... 1.01e+03</div><input id='attrs-5a94d6fd-e85b-4588-b705-1172a33fe4d6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5a94d6fd-e85b-4588-b705-1172a33fe4d6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4f3ddeac-27c9-4044-9c4f-808b846ef0d1' class='xr-var-data-in' type='checkbox'><label for='data-4f3ddeac-27c9-4044-9c4f-808b846ef0d1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1226.2942653 ],
           [1365.12475728],
           [ 994.84723146],
           [ 916.36729068],
           [1295.73672773],
           [1707.29480845],
           [1419.72501991],
           [1364.13945259],
           [1110.83808286],
           [1159.98094834],
           [ 999.84764619],
           [1552.24004957],
           [1144.08443814],
           [1469.25551167],
           [1364.65599257],
           [1042.14221901],
           [1006.29898016],
           [ 745.68755618],
           [ 964.90257327],
           [ 745.68755618],
    ...
           [1159.98094834],
           [1720.83311058],
           [1153.52961437],
           [1372.66251681],
           [1342.9633658 ],
           [1306.41632758],
           [1013.5492788 ],
           [1006.29898016],
           [1017.2808942 ],
           [1008.94298652],
           [1346.28225913],
           [ 992.2032251 ],
           [ 592.00558801],
           [ 965.51335405],
           [1211.4427891 ],
           [1420.61599353],
           [1283.02626162],
           [2237.02093696],
           [1376.45913248],
           [1009.74313466]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.03011 5.203 ... 0.5787 0.938</div><input id='attrs-64ffa4d3-9624-409b-9694-932ae3f10fc8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-64ffa4d3-9624-409b-9694-932ae3f10fc8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1b855011-276d-41e6-b602-150c5876d623' class='xr-var-data-in' type='checkbox'><label for='data-1b855011-276d-41e6-b602-150c5876d623' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[3.01075269e-02, 5.20328053e+00, 3.70812813e-01, 8.61696331e-01,
            4.21305192e-01, 6.31334522e-01, 5.68497565e-01, 2.58478842e-17,
            5.78694808e-01, 9.38021254e-01]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-3952e23c-6822-49e9-844d-92aebd3ada90' class='xr-section-summary-in' type='checkbox'  ><label for='section-3952e23c-6822-49e9-844d-92aebd3ada90' class='xr-section-summary' >Indexes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-2cee322f-b3a2-4b49-bb11-4f95ea791a6b' class='xr-index-data-in' type='checkbox'/><label for='index-2cee322f-b3a2-4b49-bb11-4f95ea791a6b' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([ 515,  441, 1029,   64,  654,   14,  562,  599, 1003,  293,
           ...
            375,  320,  334,  804,  539,  611,  549,  640,  648,  635],
          dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;, length=186))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-675e12e3-319a-48d8-b46c-12c3979366a2' class='xr-index-data-in' type='checkbox'/><label for='index-675e12e3-319a-48d8-b46c-12c3979366a2' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;], dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-23146dda-4bbd-4cfc-9127-76a659572cad' class='xr-index-data-in' type='checkbox'/><label for='index-23146dda-4bbd-4cfc-9127-76a659572cad' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-eafcf3f4-8ff4-4bd3-bc52-0e04ef1d4fac' class='xr-index-data-in' type='checkbox'/><label for='index-eafcf3f4-8ff4-4bd3-bc52-0e04ef1d4fac' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>centile</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-3fc1e2ea-55c8-438a-a382-e1e4de8400be' class='xr-index-data-in' type='checkbox'/><label for='index-3fc1e2ea-55c8-438a-a382-e1e4de8400be' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0.05, 0.25, 0.5, 0.75, 0.95], dtype=&#x27;float64&#x27;, name=&#x27;centile&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-5dd73500-f889-4250-a6c4-6d3919f2a1fd' class='xr-index-data-in' type='checkbox'/><label for='index-5dd73500-f889-4250-a6c4-6d3919f2a1fd' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;,
           &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1bb2c473-73e6-44f4-9b36-99cf2b3fff39' class='xr-section-summary-in' type='checkbox'  checked><label for='section-1bb2c473-73e6-44f4-9b36-99cf2b3fff39' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fit_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{&#x27;sex&#x27;: [&#x27;F&#x27;, &#x27;M&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;F&#x27;: 493, &#x27;M&#x27;: 437}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;F&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}, &#x27;M&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Munchen&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oxford&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd><dt><span>covariate_ranges :</span></dt><dd>{&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd></dl></div></li></ul></div></div>



.. code:: ipython3

    plot_centiles(
        model,
        centiles=[0.05, 0.5, 0.95],  # Plot these centiles, the default is [0.05, 0.25, 0.5, 0.75, 0.95]
        scatter_data=train,  # Scatter this data along with the centiles
        batch_effects={"site": ["Beijing_Zang", "AnnArbor_a"], "sex": ["M"]},  # Highlight these groups
        show_other_data=True,  # scatter data not in those groups as smaller black circles
        harmonize=True,  # harmonize the scatterdata, this means that we 'remove' the batch effects from the data, by simulating what the data would have looked like if all data was from the same batch.
    )


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 3614 - 2025-06-24 11:14:04 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)



.. image:: 06_transfer_extend_files/06_transfer_extend_10_1.png


Extending
---------

Now that we have a fitted model, we can extend it using the data that we
held out of the train set. This is from previously unseen sites. Trying
to run predict on it now, with the current model, will result in an
error:

.. code:: ipython3

    try:
        model.predict(transfer_train)
    except Exception as e:
        print(e)


.. parsed-literal::

    Data is not compatible with the model!


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 3614 - 2025-06-24 11:14:05 - The dataset transfer_train has unknown batch effects: {'sex': [], 'site': ['Milwaukee_b', 'Oulu']}
      warnings.warn(message)


And just to show why we prefer extend over just fitting a new model on
the held-out dataset, we can show how bad such a model would be:

.. code:: ipython3

    small_model = NormativeModel(
        template_regression_model=template_hbr,
        savemodel=True,
        evaluate_model=True,
        saveresults=True,
        saveplots=False,
        save_dir="resources/hbr/save_dir",
        inscaler="standardize",
        outscaler="standardize",
    )
    small_model.fit_predict(transfer_train, transfer_test)
    plot_centiles(
        small_model,
        centiles=[0.05, 0.5, 0.95],  # Plot these centiles, the default is [0.05, 0.25, 0.5, 0.75, 0.95]
        scatter_data=transfer_train,
        show_other_data=True,
        harmonize=True,
    )



.. raw:: html

    
    <style>
        :root {
            --column-width-1: 40%; /* Progress column width */
            --column-width-2: 15%; /* Chain column width */
            --column-width-3: 15%; /* Divergences column width */
            --column-width-4: 15%; /* Step Size column width */
            --column-width-5: 15%; /* Gradients/Draw column width */
        }
    
        .nutpie {
            max-width: 800px;
            margin: 10px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            //color: #333;
            //background-color: #fff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            font-size: 14px; /* Smaller font size for a more compact look */
        }
        .nutpie table {
            width: 100%;
            border-collapse: collapse; /* Remove any extra space between borders */
        }
        .nutpie th, .nutpie td {
            padding: 8px 10px; /* Reduce padding to make table more compact */
            text-align: left;
            border-bottom: 1px solid #888;
        }
        .nutpie th {
            //background-color: #f0f0f0;
        }
    
        .nutpie th:nth-child(1) { width: var(--column-width-1); }
        .nutpie th:nth-child(2) { width: var(--column-width-2); }
        .nutpie th:nth-child(3) { width: var(--column-width-3); }
        .nutpie th:nth-child(4) { width: var(--column-width-4); }
        .nutpie th:nth-child(5) { width: var(--column-width-5); }
    
        .nutpie progress {
            width: 100%;
            height: 15px; /* Smaller progress bars */
            border-radius: 5px;
        }
        progress::-webkit-progress-bar {
            background-color: #eee;
            border-radius: 5px;
        }
        progress::-webkit-progress-value {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        progress::-moz-progress-bar {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        .nutpie .progress-cell {
            width: 100%;
        }
    
        .nutpie p strong { font-size: 16px; font-weight: bold; }
    
        @media (prefers-color-scheme: dark) {
            .nutpie {
                //color: #ddd;
                //background-color: #1e1e1e;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
            .nutpie table, .nutpie th, .nutpie td {
                border-color: #555;
                color: #ccc;
            }
            .nutpie th {
                background-color: #2a2a2a;
            }
            .nutpie progress::-webkit-progress-bar {
                background-color: #444;
            }
            .nutpie progress::-webkit-progress-value {
                background-color: #3178c6;
            }
            .nutpie progress::-moz-progress-bar {
                background-color: #3178c6;
            }
        }
    </style>




.. raw:: html

    
    <div class="nutpie">
        <p><strong>Sampler Progress</strong></p>
        <p>Total Chains: <span id="total-chains">4</span></p>
        <p>Active Chains: <span id="active-chains">0</span></p>
        <p>
            Finished Chains:
            <span id="active-chains">4</span>
        </p>
        <p>Sampling for now</p>
        <p>
            Estimated Time to Completion:
            <span id="eta">now</span>
        </p>
    
        <progress
            id="total-progress-bar"
            max="8000"
            value="8000">
        </progress>
        <table>
            <thead>
                <tr>
                    <th>Progress</th>
                    <th>Draws</th>
                    <th>Divergences</th>
                    <th>Step Size</th>
                    <th>Gradients/Draw</th>
                </tr>
            </thead>
            <tbody id="chain-details">
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>3</td>
                        <td>0.13</td>
                        <td>63</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>3</td>
                        <td>0.13</td>
                        <td>127</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>2</td>
                        <td>0.15</td>
                        <td>63</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>3</td>
                        <td>0.14</td>
                        <td>31</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>



.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 3614 - 2025-06-24 11:16:56 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)



.. image:: 06_transfer_extend_files/06_transfer_extend_14_3.png


The interpolation between ages 22 and 45 is very bad, and that’s because
there was no train data there. This model will not perform well on new
data. Now instead, let’s transfer the model we fitted before to our
smaller dataset, and see how those centiles look:

.. code:: ipython3

    extended_model = model.extend_predict(transfer_train, transfer_test)
    plot_centiles(
        extended_model,
        centiles=[0.05, 0.5, 0.95],  # Plot these centiles, the default is [0.05, 0.25, 0.5, 0.75, 0.95]
        scatter_data=transfer_train,
        show_other_data=True,
        harmonize=True,
    )



.. raw:: html

    
    <style>
        :root {
            --column-width-1: 40%; /* Progress column width */
            --column-width-2: 15%; /* Chain column width */
            --column-width-3: 15%; /* Divergences column width */
            --column-width-4: 15%; /* Step Size column width */
            --column-width-5: 15%; /* Gradients/Draw column width */
        }
    
        .nutpie {
            max-width: 800px;
            margin: 10px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            //color: #333;
            //background-color: #fff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            font-size: 14px; /* Smaller font size for a more compact look */
        }
        .nutpie table {
            width: 100%;
            border-collapse: collapse; /* Remove any extra space between borders */
        }
        .nutpie th, .nutpie td {
            padding: 8px 10px; /* Reduce padding to make table more compact */
            text-align: left;
            border-bottom: 1px solid #888;
        }
        .nutpie th {
            //background-color: #f0f0f0;
        }
    
        .nutpie th:nth-child(1) { width: var(--column-width-1); }
        .nutpie th:nth-child(2) { width: var(--column-width-2); }
        .nutpie th:nth-child(3) { width: var(--column-width-3); }
        .nutpie th:nth-child(4) { width: var(--column-width-4); }
        .nutpie th:nth-child(5) { width: var(--column-width-5); }
    
        .nutpie progress {
            width: 100%;
            height: 15px; /* Smaller progress bars */
            border-radius: 5px;
        }
        progress::-webkit-progress-bar {
            background-color: #eee;
            border-radius: 5px;
        }
        progress::-webkit-progress-value {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        progress::-moz-progress-bar {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        .nutpie .progress-cell {
            width: 100%;
        }
    
        .nutpie p strong { font-size: 16px; font-weight: bold; }
    
        @media (prefers-color-scheme: dark) {
            .nutpie {
                //color: #ddd;
                //background-color: #1e1e1e;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
            .nutpie table, .nutpie th, .nutpie td {
                border-color: #555;
                color: #ccc;
            }
            .nutpie th {
                background-color: #2a2a2a;
            }
            .nutpie progress::-webkit-progress-bar {
                background-color: #444;
            }
            .nutpie progress::-webkit-progress-value {
                background-color: #3178c6;
            }
            .nutpie progress::-moz-progress-bar {
                background-color: #3178c6;
            }
        }
    </style>




.. raw:: html

    
    <div class="nutpie">
        <p><strong>Sampler Progress</strong></p>
        <p>Total Chains: <span id="total-chains">4</span></p>
        <p>Active Chains: <span id="active-chains">0</span></p>
        <p>
            Finished Chains:
            <span id="active-chains">4</span>
        </p>
        <p>Sampling for now</p>
        <p>
            Estimated Time to Completion:
            <span id="eta">now</span>
        </p>
    
        <progress
            id="total-progress-bar"
            max="8000"
            value="8000">
        </progress>
        <table>
            <thead>
                <tr>
                    <th>Progress</th>
                    <th>Draws</th>
                    <th>Divergences</th>
                    <th>Step Size</th>
                    <th>Gradients/Draw</th>
                </tr>
            </thead>
            <tbody id="chain-details">
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>932</td>
                        <td>0.12</td>
                        <td>31</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>6</td>
                        <td>0.12</td>
                        <td>31</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>2</td>
                        <td>0.11</td>
                        <td>255</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>17</td>
                        <td>0.12</td>
                        <td>63</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>



.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 3614 - 2025-06-24 11:19:13 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 3614 - 2025-06-24 11:19:17 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 3614 - 2025-06-24 11:19:18 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)



.. image:: 06_transfer_extend_files/06_transfer_extend_16_3.png


These centiles look much better. The extended model is a larger model
than the original one, it can be used on the original train data as well
as the extended data:

.. code:: ipython3

    extended_model.predict(train)


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 3614 - 2025-06-24 11:21:57 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 179kB
    Dimensions:            (observations: 744, response_vars: 1, covariates: 1,
                            batch_effect_dims: 2, centile: 5, statistic: 10)
    Coordinates:
      * observations       (observations) int64 6kB 459 995 432 ... 1023 1062 372
      * response_vars      (response_vars) &lt;U18 72B &#x27;WM-hypointensities&#x27;
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) &lt;U8 320B &#x27;MACE&#x27; &#x27;MAPE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;
    Data variables:
        subjects           (observations) object 6kB &#x27;Cambridge_Buckner_sub53615&#x27;...
        Y                  (observations, response_vars) float64 6kB 974.0 ... 1....
        X                  (observations, covariates) float64 6kB 19.0 29.0 ... 25.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 101kB &#x27;M&#x27; ... &#x27;...
        Z                  (observations, response_vars) float64 6kB -1.048 ... 2...
        centiles           (centile, observations, response_vars) float64 30kB 76...
        logp               (observations, response_vars) float64 6kB -0.7252 ... ...
        Yhat               (observations, response_vars) float64 6kB 1.409e+03 .....
        statistics         (response_vars, statistic) float64 80B 0.04194 ... 0.9058
        Y_harmonized       (observations, response_vars) float64 6kB 1.013e+03 .....
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fit_train
        unique_batch_effects:           {&#x27;sex&#x27;: [&#x27;F&#x27;, &#x27;M&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;...
        batch_effect_counts:            {&#x27;sex&#x27;: {&#x27;F&#x27;: 493, &#x27;M&#x27;: 437}, &#x27;site&#x27;: {&#x27;A...
        batch_effect_covariate_ranges:  {&#x27;sex&#x27;: {&#x27;F&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;...
        covariate_ranges:               {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-585d7b75-b9ac-40c0-96d6-a7ca4dc14cd3' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-585d7b75-b9ac-40c0-96d6-a7ca4dc14cd3' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 744</li><li><span class='xr-has-index'>response_vars</span>: 1</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>centile</span>: 5</li><li><span class='xr-has-index'>statistic</span>: 10</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-cbf3f676-e5be-4b53-abc7-62a7d618efc2' class='xr-section-summary-in' type='checkbox'  checked><label for='section-cbf3f676-e5be-4b53-abc7-62a7d618efc2' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>459 995 432 288 ... 1023 1062 372</div><input id='attrs-3a9d5a01-2022-47a3-ab1e-b0bcd68e3a74' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3a9d5a01-2022-47a3-ab1e-b0bcd68e3a74' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-28c855a5-f721-4da8-88e7-d2d61969d5e6' class='xr-var-data-in' type='checkbox'><label for='data-28c855a5-f721-4da8-88e7-d2d61969d5e6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 459,  995,  432, ..., 1023, 1062,  372])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U18</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27;</div><input id='attrs-55e37864-097c-4281-ba72-6ac3b6347a6f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-55e37864-097c-4281-ba72-6ac3b6347a6f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f0294b72-62be-4608-95ac-8334e03ad9bd' class='xr-var-data-in' type='checkbox'><label for='data-f0294b72-62be-4608-95ac-8334e03ad9bd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;], dtype=&#x27;&lt;U18&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-d8e0aef8-8dfe-4153-a545-477a164c4d1d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d8e0aef8-8dfe-4153-a545-477a164c4d1d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6b375f43-2336-49ff-bf87-cd8e23ac6f9f' class='xr-var-data-in' type='checkbox'><label for='data-6b375f43-2336-49ff-bf87-cd8e23ac6f9f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-c888d331-8a0e-4399-bcfc-a1501d32b97b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c888d331-8a0e-4399-bcfc-a1501d32b97b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-eeb2254a-f4ee-4955-878f-cfe97396873a' class='xr-var-data-in' type='checkbox'><label for='data-eeb2254a-f4ee-4955-878f-cfe97396873a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-31866eb8-c363-49ee-baea-916dfb3ff369' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-31866eb8-c363-49ee-baea-916dfb3ff369' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-38151c92-2c5a-442f-b5b4-9a8cb457b575' class='xr-var-data-in' type='checkbox'><label for='data-38151c92-2c5a-442f-b5b4-9a8cb457b575' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;MACE&#x27; &#x27;MAPE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-a716a7a4-cc04-428b-9a40-7c4905699d41' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a716a7a4-cc04-428b-9a40-7c4905699d41' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-96dd3acc-0b63-4c47-8c7d-6621d07ba1d2' class='xr-var-data-in' type='checkbox'><label for='data-96dd3acc-0b63-4c47-8c7d-6621d07ba1d2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;,
           &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-74f30d5a-02d8-4f64-a26a-260c33c4ebc3' class='xr-section-summary-in' type='checkbox'  checked><label for='section-74f30d5a-02d8-4f64-a26a-260c33c4ebc3' class='xr-section-summary' >Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subjects</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Cambridge_Buckner_sub53615&#x27; ......</div><input id='attrs-64190500-0313-4f05-8f4e-dbdc9fb69a97' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-64190500-0313-4f05-8f4e-dbdc9fb69a97' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ed194e15-c438-4f65-9186-915f23678ec7' class='xr-var-data-in' type='checkbox'><label for='data-ed194e15-c438-4f65-9186-915f23678ec7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Cambridge_Buckner_sub53615&#x27;, &#x27;Oxford_sub47141&#x27;,
           &#x27;Cambridge_Buckner_sub42146&#x27;, &#x27;Beijing_Zang_sub82714&#x27;,
           &#x27;AnnArbor_a_sub96621&#x27;, &#x27;SaintLouis_sub95721&#x27;, &#x27;Cleveland_sub99664&#x27;,
           &#x27;Cambridge_Buckner_sub83683&#x27;, &#x27;Beijing_Zang_sub59347&#x27;,
           &#x27;Munchen_sub70942&#x27;, &#x27;Cambridge_Buckner_sub13187&#x27;,
           &#x27;Cambridge_Buckner_sub13902&#x27;, &#x27;Queensland_sub39524&#x27;,
           &#x27;Atlanta_sub52783&#x27;, &#x27;Cleveland_sub18011&#x27;, &#x27;Beijing_Zang_sub61961&#x27;,
           &#x27;Baltimore_sub52358&#x27;, &#x27;Beijing_Zang_sub54890&#x27;,
           &#x27;Beijing_Zang_sub55736&#x27;, &#x27;Munchen_sub31272&#x27;,
           &#x27;Beijing_Zang_sub55856&#x27;, &#x27;Beijing_Zang_sub55541&#x27;,
           &#x27;Atlanta_sub91049&#x27;, &#x27;NewYork_a_ADHD_sub15758&#x27;,
           &#x27;Baltimore_sub19738&#x27;, &#x27;Cleveland_sub26557&#x27;,
           &#x27;Cambridge_Buckner_sub57221&#x27;, &#x27;Beijing_Zang_sub40427&#x27;,
           &#x27;ICBM_sub53801&#x27;, &#x27;Oxford_sub66945&#x27;, &#x27;Beijing_Zang_sub38602&#x27;,
           &#x27;ICBM_sub54887&#x27;, &#x27;AnnArbor_b_sub43409&#x27;,
           &#x27;Cambridge_Buckner_sub51050&#x27;, &#x27;ICBM_sub29353&#x27;, &#x27;ICBM_sub76678&#x27;,
           &#x27;Munchen_sub28902&#x27;, &#x27;ICBM_sub47753&#x27;, &#x27;SaintLouis_sub74078&#x27;,
           &#x27;Cambridge_Buckner_sub78547&#x27;, &#x27;Leiden_2180_sub56299&#x27;,
           &#x27;Beijing_Zang_sub89592&#x27;, &#x27;ICBM_sub30623&#x27;, &#x27;Atlanta_sub86323&#x27;,
           &#x27;SaintLouis_sub99965&#x27;, &#x27;Baltimore_sub86414&#x27;,
    ...
           &#x27;PaloAlto_sub58313&#x27;, &#x27;SaintLouis_sub88823&#x27;, &#x27;Baltimore_sub54329&#x27;,
           &#x27;Beijing_Zang_sub42512&#x27;, &#x27;Newark_sub13411&#x27;,
           &#x27;Cambridge_Buckner_sub99085&#x27;, &#x27;Beijing_Zang_sub51015&#x27;,
           &#x27;Berlin_Margulies_sub85681&#x27;, &#x27;Beijing_Zang_sub00440&#x27;,
           &#x27;Cambridge_Buckner_sub13093&#x27;, &#x27;Beijing_Zang_sub80927&#x27;,
           &#x27;SaintLouis_sub46405&#x27;, &#x27;Cambridge_Buckner_sub34586&#x27;,
           &#x27;Atlanta_sub58250&#x27;, &#x27;Cambridge_Buckner_sub50953&#x27;,
           &#x27;Berlin_Margulies_sub12855&#x27;, &#x27;Berlin_Margulies_sub06716&#x27;,
           &#x27;Cambridge_Buckner_sub07413&#x27;, &#x27;Beijing_Zang_sub95575&#x27;,
           &#x27;Beijing_Zang_sub92430&#x27;, &#x27;Beijing_Zang_sub30272&#x27;,
           &#x27;Cambridge_Buckner_sub45604&#x27;, &#x27;ICBM_sub98317&#x27;,
           &#x27;Baltimore_sub54257&#x27;, &#x27;Oxford_sub40451&#x27;, &#x27;Atlanta_sub00354&#x27;,
           &#x27;Beijing_Zang_sub04191&#x27;, &#x27;Leiden_2180_sub08518&#x27;,
           &#x27;Cambridge_Buckner_sub16390&#x27;, &#x27;AnnArbor_b_sub57196&#x27;,
           &#x27;Beijing_Zang_sub75878&#x27;, &#x27;ICBM_sub76325&#x27;, &#x27;Beijing_Zang_sub35776&#x27;,
           &#x27;PaloAlto_sub46856&#x27;, &#x27;ICBM_sub48210&#x27;, &#x27;AnnArbor_b_sub00306&#x27;,
           &#x27;AnnArbor_b_sub98007&#x27;, &#x27;NewYork_a_sub53710&#x27;,
           &#x27;NewYork_a_ADHD_sub20676&#x27;, &#x27;Berlin_Margulies_sub54976&#x27;,
           &#x27;NewYork_a_sub20732&#x27;, &#x27;PaloAlto_sub96705&#x27;, &#x27;SaintLouis_sub58674&#x27;,
           &#x27;Cambridge_Buckner_sub09015&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>974.0 1.114e+03 ... 485.4 1.934e+03</div><input id='attrs-0b964119-3062-404a-9b75-ad3fca60cbc7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0b964119-3062-404a-9b75-ad3fca60cbc7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9a51c85b-f444-4c02-8141-53918d554b55' class='xr-var-data-in' type='checkbox'><label for='data-9a51c85b-f444-4c02-8141-53918d554b55' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[  974. ],
           [ 1113.8],
           [  916. ],
           [  612.6],
           [ 1622.6],
           [ 1028.1],
           [  845. ],
           [  987.5],
           [ 1390.5],
           [ 4640.5],
           [  946. ],
           [  934. ],
           [ 1003.4],
           [  626.5],
           [ 1378.7],
           [ 1108.7],
           [  865.6],
           [ 1456.4],
           [ 1041.4],
           [ 1829.5],
    ...
           [  710.6],
           [  855.6],
           [  927.2],
           [ 1136.1],
           [ 1422.8],
           [ 2170.7],
           [ 1034.2],
           [  765.3],
           [  697.5],
           [  760.9],
           [ 3769.9],
           [ 1080. ],
           [ 4248.1],
           [ 1178. ],
           [ 1295.7],
           [  664.4],
           [  937.1],
           [ 2242.1],
           [  485.4],
           [ 1934.5]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>19.0 29.0 24.0 ... 29.0 28.0 25.0</div><input id='attrs-8f6d1552-01bf-4343-857d-80e81492bdc5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8f6d1552-01bf-4343-857d-80e81492bdc5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0e4af8f6-6d62-4e48-98a5-0d671c295e18' class='xr-var-data-in' type='checkbox'><label for='data-0e4af8f6-6d62-4e48-98a5-0d671c295e18' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[19.  ],
           [29.  ],
           [24.  ],
           [20.  ],
           [15.56],
           [21.  ],
           [57.  ],
           [22.  ],
           [21.  ],
           [70.  ],
           [23.  ],
           [19.  ],
           [27.  ],
           [22.  ],
           [55.  ],
           [21.  ],
           [40.  ],
           [21.  ],
           [19.  ],
           [74.  ],
    ...
           [34.  ],
           [28.  ],
           [24.  ],
           [21.  ],
           [23.  ],
           [77.  ],
           [24.  ],
           [57.  ],
           [24.  ],
           [22.  ],
           [73.  ],
           [68.  ],
           [66.  ],
           [34.72],
           [49.19],
           [37.  ],
           [11.07],
           [29.  ],
           [28.  ],
           [25.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;M&#x27; ... &#x27;Cambridge_Buckner&#x27;</div><input id='attrs-ba78a517-743a-4de8-8360-f0840ccafb6c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ba78a517-743a-4de8-8360-f0840ccafb6c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-625f0e84-f17f-4734-b1e4-ff22f6505f1d' class='xr-var-data-in' type='checkbox'><label for='data-625f0e84-f17f-4734-b1e4-ff22f6505f1d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;F&#x27;, &#x27;Oxford&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           ...,
           [&#x27;F&#x27;, &#x27;PaloAlto&#x27;],
           [&#x27;F&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.048 0.2926 ... -1.039 2.023</div><input id='attrs-cfbd0c19-89a6-42d5-b6cd-6f7f8a535d0e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cfbd0c19-89a6-42d5-b6cd-6f7f8a535d0e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f8525f11-4570-47ee-942b-5092e7680053' class='xr-var-data-in' type='checkbox'><label for='data-f8525f11-4570-47ee-942b-5092e7680053' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.04760964e+00],
           [ 2.92640399e-01],
           [-7.15423104e-01],
           [-1.00077677e+00],
           [ 2.30027855e-01],
           [ 1.80310993e-01],
           [-3.99584270e-01],
           [-5.93756655e-01],
           [ 4.98529853e-01],
           [ 8.45279153e-01],
           [-1.14747243e+00],
           [-1.14384948e+00],
           [ 1.85736900e-01],
           [-9.75176275e-01],
           [-3.26368316e-02],
           [ 2.44169368e-01],
           [ 3.08649921e-02],
           [ 1.12573847e+00],
           [-3.66358911e-01],
           [-1.21138523e+00],
    ...
           [-1.23633138e+00],
           [-2.21641089e-01],
           [-5.95377044e-01],
           [ 1.94917593e-01],
           [ 5.76143656e-01],
           [-1.31818366e+00],
           [-3.12708330e-01],
           [-5.28565947e-01],
           [-1.20219016e+00],
           [-1.31701523e+00],
           [ 1.00663156e-02],
           [-1.03724359e+00],
           [ 1.19920066e+00],
           [ 2.81158626e-01],
           [-2.69948478e-01],
           [-7.12139445e-01],
           [ 5.66705802e-01],
           [ 2.66609377e+00],
           [-1.03863781e+00],
           [ 2.02277465e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>765.4 359.6 ... 1.349e+03 1.766e+03</div><input id='attrs-66726660-ce9f-426f-9bd5-bfb81fe12743' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-66726660-ce9f-426f-9bd5-bfb81fe12743' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cd520f7c-51ac-48e1-92ee-43ec0a85c08d' class='xr-var-data-in' type='checkbox'><label for='data-cd520f7c-51ac-48e1-92ee-43ec0a85c08d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 765.35085531],
            [ 359.63002482],
            [ 615.63166899],
            ...,
            [ 821.08884893],
            [ 146.16421434],
            [ 611.07817   ]],
    
           [[1123.62790112],
            [ 721.76192195],
            [ 954.05021606],
            ...,
            [1183.22074607],
            [ 501.01507105],
            [ 951.60166882]],
    
           [[1372.66251681],
            [ 973.47600435],
            [1189.28139763],
            ...,
            [1434.93482847],
            [ 747.66817864],
            [1188.29598027]],
    
           [[1621.69713251],
            [1225.19008676],
            [1424.51257919],
            ...,
            [1686.64891087],
            [ 994.32128624],
            [1424.99029171]],
    
           [[1979.97417832],
            [1587.32198389],
            [1762.93112626],
            ...,
            [2048.780808  ],
            [1349.17214295],
            [1765.51379054]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.7252 -0.2433 ... -0.4027 -2.361</div><input id='attrs-10b17c94-2488-4194-a1dc-635a8dc20961' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-10b17c94-2488-4194-a1dc-635a8dc20961' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-12c9b275-d27b-453f-9c05-3999648efa66' class='xr-var-data-in' type='checkbox'><label for='data-12c9b275-d27b-453f-9c05-3999648efa66' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -0.72521097],
           [ -0.2432777 ],
           [ -0.39126183],
           [ -0.73603717],
           [ -0.3270347 ],
           [ -0.42436803],
           [ -1.27034144],
           [ -0.26571526],
           [ -0.30904238],
           [ -2.24814158],
           [ -0.74473899],
           [ -0.84867475],
           [ -0.14109581],
           [ -0.47334352],
           [ -1.09262669],
           [ -0.13692576],
           [ -0.52991916],
           [ -0.90819573],
           [ -0.20475006],
           [ -2.34255704],
    ...
           [ -0.95522997],
           [ -0.16985572],
           [ -0.27992248],
           [ -0.13450865],
           [ -0.29987525],
           [ -2.53752146],
           [ -0.13470752],
           [ -1.31391853],
           [ -0.9108154 ],
           [ -1.91878178],
           [ -1.98220842],
           [ -2.05400602],
           [ -2.44440238],
           [ -0.35749543],
           [ -0.88071845],
           [ -0.51319058],
           [ -0.78492232],
           [ -2.54131725],
           [ -0.40269435],
           [ -2.36137465]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.409e+03 1e+03 ... 882.4 1.172e+03</div><input id='attrs-d574fc19-b8f4-4cb8-bfe3-97eeafc21ebf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d574fc19-b8f4-4cb8-bfe3-97eeafc21ebf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ab0e8047-e0d5-41b1-9af9-efc96ae6c2d2' class='xr-var-data-in' type='checkbox'><label for='data-ab0e8047-e0d5-41b1-9af9-efc96ae6c2d2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1409.42887702],
           [1000.48982239],
           [1186.80736922],
           [1016.80979221],
           [1512.1140878 ],
           [ 956.85084273],
           [1290.96234702],
           [1217.37700718],
           [1193.89603307],
           [3194.22408922],
           [1384.22011634],
           [1409.42887702],
           [ 933.11745869],
           [1003.9467217 ],
           [1412.45581545],
           [1012.43813352],
           [ 848.80059152],
           [1012.43813352],
           [1193.73627592],
           [4090.06540017],
    ...
           [1247.11514619],
           [ 940.20168139],
           [1152.57266768],
           [1059.27217581],
           [1202.76221678],
           [4750.84373355],
           [1152.57266768],
           [1355.67618616],
           [1152.57266768],
           [1270.7384005 ],
           [3750.22442391],
           [2763.76825407],
           [2419.7791508 ],
           [1053.19081007],
           [1513.1052383 ],
           [1006.55028222],
           [ 579.89171194],
           [1210.05119667],
           [ 882.35207498],
           [1172.03347029]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.04194 2.657 ... 0.6458 0.9058</div><input id='attrs-864953c9-b6f7-4a97-971e-b41a5ac87e54' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-864953c9-b6f7-4a97-971e-b41a5ac87e54' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a774dd96-1a9e-4ba8-8cbc-f0c14357a297' class='xr-var-data-in' type='checkbox'><label for='data-a774dd96-1a9e-4ba8-8cbc-f0c14357a297' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[4.19354839e-02, 2.65669756e+00, 5.97350076e-01, 7.78461803e-01,
            3.54166619e-01, 7.69716109e-01, 5.27058901e-01, 2.02835269e-54,
            6.45833381e-01, 9.05766075e-01]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y_harmonized</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.013e+03 1.491e+03 ... 2.156e+03</div><input id='attrs-850df412-06a5-4f99-8d51-ee909c62b69a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-850df412-06a5-4f99-8d51-ee909c62b69a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b97dcafa-c9c9-482d-a3cc-f43098103138' class='xr-var-data-in' type='checkbox'><label for='data-b97dcafa-c9c9-482d-a3cc-f43098103138' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1012.93571722],
           [ 1491.29620337],
           [ 1136.94859744],
           [ 1048.9738294 ],
           [ 1441.12874256],
           [ 1520.49734358],
           [ 1367.71646615],
           [ 1208.47984582],
           [ 1646.1380888 ],
           [ 4824.34467833],
           [  985.36767907],
           [  972.86273836],
           [ 1446.52609492],
           [ 1060.68501855],
           [ 1721.77705337],
           [ 1545.71022335],
           [ 1590.78089148],
           [ 1893.77569793],
           [ 1296.59937504],
           [ 1967.11710829],
    ...
           [  905.46989109],
           [ 1289.99091987],
           [ 1182.4273858 ],
           [ 1526.26439401],
           [ 1644.14261825],
           [ 2331.04712563],
           [ 1289.51487134],
           [ 1223.1782389 ],
           [  952.53957805],
           [  928.22715019],
           [ 4047.25406915],
           [ 1267.85516538],
           [ 4469.58653488],
           [ 1582.3050056 ],
           [ 1474.921419  ],
           [ 1163.79646638],
           [ 1167.38066404],
           [ 2411.27634201],
           [  977.3471655 ],
           [ 2156.38714012]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-5784709b-7378-4ed6-9750-582d7451e351' class='xr-section-summary-in' type='checkbox'  ><label for='section-5784709b-7378-4ed6-9750-582d7451e351' class='xr-section-summary' >Indexes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-04e0f796-71d9-4cc5-8949-c3a7ff46bd85' class='xr-index-data-in' type='checkbox'/><label for='index-04e0f796-71d9-4cc5-8949-c3a7ff46bd85' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([ 459,  995,  432,  288,   23, 1073,  579,  516,  248,  752,
           ...
            618,   24,   54,  801,  847,  340,  771, 1023, 1062,  372],
          dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;, length=744))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-eee3017f-1d90-48ec-b092-00da3b7e9db5' class='xr-index-data-in' type='checkbox'/><label for='index-eee3017f-1d90-48ec-b092-00da3b7e9db5' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;], dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-09c7a3a2-16c3-46da-b5bd-eb4f6a7073ff' class='xr-index-data-in' type='checkbox'/><label for='index-09c7a3a2-16c3-46da-b5bd-eb4f6a7073ff' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-99ee1c7d-f60c-4cac-84c6-73ba39798049' class='xr-index-data-in' type='checkbox'/><label for='index-99ee1c7d-f60c-4cac-84c6-73ba39798049' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>centile</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-ac2d275c-6125-494e-86f1-004ab68b8b82' class='xr-index-data-in' type='checkbox'/><label for='index-ac2d275c-6125-494e-86f1-004ab68b8b82' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0.05, 0.25, 0.5, 0.75, 0.95], dtype=&#x27;float64&#x27;, name=&#x27;centile&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-49c2b1f5-c390-4d44-9207-34302dbc8268' class='xr-index-data-in' type='checkbox'/><label for='index-49c2b1f5-c390-4d44-9207-34302dbc8268' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;,
           &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-fb2837ab-ed58-4dad-b840-6e89d484e4cd' class='xr-section-summary-in' type='checkbox'  checked><label for='section-fb2837ab-ed58-4dad-b840-6e89d484e4cd' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fit_train</dd><dt><span>unique_batch_effects :</span></dt><dd>{&#x27;sex&#x27;: [&#x27;F&#x27;, &#x27;M&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;F&#x27;: 493, &#x27;M&#x27;: 437}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;F&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}, &#x27;M&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Munchen&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oxford&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd><dt><span>covariate_ranges :</span></dt><dd>{&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd></dl></div></li></ul></div></div>



Transfering
-----------

Transfering looks very similar to extending, but the underlying
mathematics is very different. Besides that, it leads to a smaller model
instead of a bigger one; we can *not* use a transfered model on the
original train data.

.. code:: ipython3

    transfered_model = model.transfer_predict(transfer_train, transfer_test)
    plot_centiles(
        transfered_model,
        centiles=[0.05, 0.5, 0.95],  # Plot these centiles, the default is [0.05, 0.25, 0.5, 0.75, 0.95]
        scatter_data=transfer_train,
        show_other_data=True,
        harmonize=True,
    )



.. raw:: html

    
    <style>
        :root {
            --column-width-1: 40%; /* Progress column width */
            --column-width-2: 15%; /* Chain column width */
            --column-width-3: 15%; /* Divergences column width */
            --column-width-4: 15%; /* Step Size column width */
            --column-width-5: 15%; /* Gradients/Draw column width */
        }
    
        .nutpie {
            max-width: 800px;
            margin: 10px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            //color: #333;
            //background-color: #fff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            font-size: 14px; /* Smaller font size for a more compact look */
        }
        .nutpie table {
            width: 100%;
            border-collapse: collapse; /* Remove any extra space between borders */
        }
        .nutpie th, .nutpie td {
            padding: 8px 10px; /* Reduce padding to make table more compact */
            text-align: left;
            border-bottom: 1px solid #888;
        }
        .nutpie th {
            //background-color: #f0f0f0;
        }
    
        .nutpie th:nth-child(1) { width: var(--column-width-1); }
        .nutpie th:nth-child(2) { width: var(--column-width-2); }
        .nutpie th:nth-child(3) { width: var(--column-width-3); }
        .nutpie th:nth-child(4) { width: var(--column-width-4); }
        .nutpie th:nth-child(5) { width: var(--column-width-5); }
    
        .nutpie progress {
            width: 100%;
            height: 15px; /* Smaller progress bars */
            border-radius: 5px;
        }
        progress::-webkit-progress-bar {
            background-color: #eee;
            border-radius: 5px;
        }
        progress::-webkit-progress-value {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        progress::-moz-progress-bar {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        .nutpie .progress-cell {
            width: 100%;
        }
    
        .nutpie p strong { font-size: 16px; font-weight: bold; }
    
        @media (prefers-color-scheme: dark) {
            .nutpie {
                //color: #ddd;
                //background-color: #1e1e1e;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
            .nutpie table, .nutpie th, .nutpie td {
                border-color: #555;
                color: #ccc;
            }
            .nutpie th {
                background-color: #2a2a2a;
            }
            .nutpie progress::-webkit-progress-bar {
                background-color: #444;
            }
            .nutpie progress::-webkit-progress-value {
                background-color: #3178c6;
            }
            .nutpie progress::-moz-progress-bar {
                background-color: #3178c6;
            }
        }
    </style>




.. raw:: html

    
    <div class="nutpie">
        <p><strong>Sampler Progress</strong></p>
        <p>Total Chains: <span id="total-chains">4</span></p>
        <p>Active Chains: <span id="active-chains">0</span></p>
        <p>
            Finished Chains:
            <span id="active-chains">4</span>
        </p>
        <p>Sampling for now</p>
        <p>
            Estimated Time to Completion:
            <span id="eta">now</span>
        </p>
    
        <progress
            id="total-progress-bar"
            max="8000"
            value="8000">
        </progress>
        <table>
            <thead>
                <tr>
                    <th>Progress</th>
                    <th>Draws</th>
                    <th>Divergences</th>
                    <th>Step Size</th>
                    <th>Gradients/Draw</th>
                </tr>
            </thead>
            <tbody id="chain-details">
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>5</td>
                        <td>0.22</td>
                        <td>63</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>4</td>
                        <td>0.22</td>
                        <td>63</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>5</td>
                        <td>0.23</td>
                        <td>31</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>2</td>
                        <td>0.20</td>
                        <td>63</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>



.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 3614 - 2025-06-24 11:25:01 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 3614 - 2025-06-24 11:25:05 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 3614 - 2025-06-24 11:25:07 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)



.. image:: 06_transfer_extend_files/06_transfer_extend_21_3.png


Here we see that the transfered model is also much better than the
‘small model’ that we trained directly on the small dataset.
