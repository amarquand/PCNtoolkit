.. code:: ipython3

    import logging
    import warnings
    
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymc as pm
    import seaborn as sns
    
    import pcntoolkit.util.output
    from pcntoolkit import (
        HBR,
        BsplineBasisFunction,
        NormalLikelihood,
        NormativeModel,
        NormData,
        load_fcon1000,
        make_prior,
    )
    from pcntoolkit.util.model_comparison import compare_hbr_models
    
    sns.set_style("darkgrid")
    
    # Suppress some annoying warnings and logs
    pymc_logger = logging.getLogger("pymc")
    
    pymc_logger.setLevel(logging.WARNING)
    pymc_logger.propagate = False
    
    warnings.simplefilter(action="ignore", category=FutureWarning)
    pd.options.mode.chained_assignment = None  # default='warn'
    pcntoolkit.util.output.Output.set_show_messages(True)

.. code:: ipython3

    # Download an example dataset
    norm_data: NormData = load_fcon1000()
    
    # Select only a few features
    features_to_model = [
        "WM-hypointensities",
        "Right-Lateral-Ventricle",
        # "Right-Amygdala",
        # "CortexVol",
    ]
    norm_data = norm_data.sel({"response_vars": features_to_model})
    
    # Split into train and test sets
    train, test = norm_data.train_test_split()


.. parsed-literal::

    Process: 88956 - 2025-09-04 16:31:34 - Removed 0 NANs
    Process: 88956 - 2025-09-04 16:31:34 - Dataset "fcon1000" created.
        - 1078 observations
        - 1078 unique subjects
        - 1 covariates
        - 217 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        


.. code:: ipython3

    mu1 = make_prior(
        # Mu is linear because we want to allow the mean to vary as a function of the covariates.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 10.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 5.0)),
        # The intercept is not random, because we want to compare to a model with random intercept
        intercept=make_prior(
            dist_name="Normal",
            dist_params=(0.0, 2.0),
        ),
        # We use a B-spline basis function to allow for non-linearity in the mean.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    sigma1 = make_prior(
        # Sigma is also linear, because we want to allow the standard deviation to vary as a function of the covariates: heteroskedasticity.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 2.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        # The intercept is not random, because we assume the intercept of the variance to be the same for all sites and sexes.
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
    # Set the likelihood with the priors we just created.
    likelihood1 = NormalLikelihood(mu1, sigma1)
    
    template_hbr_1 = HBR(
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
        # The sampler to use for the model.
        nuts_sampler="nutpie",
        # The likelihood function to use for the model.
        likelihood=likelihood1,
    )
    model1 = NormativeModel(
        # The regression model to use for the normative model.
        template_regression_model=template_hbr_1,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir="resources/compare_hbr/model1",
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
    )

.. code:: ipython3

    mu2 = make_prior(
        # Mu is linear because we want to allow the mean to vary as a function of the covariates.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 10.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 5.0)),
        # The intercept is random, because we expect the intercept to vary between sites and sexes.
        intercept=make_prior(
            random=True,
            # Mu is the mean of the intercept, which is normally distributed with a mean of 0 and a standard deviation of 1.
            mu=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
            # Sigma is the scale at which the intercepts vary. It is a positive parameter, so we have to map it to the positive domain.
            sigma=make_prior(dist_name="Normal", dist_params=(1.0, 0.5), mapping="softplus", mapping_params=(0.0, 2.0)),
        ),
        # We use a B-spline basis function to allow for non-linearity in the mean.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    sigma2 = make_prior(
        # Sigma is also linear, because we want to allow the standard deviation to vary as a function of the covariates: heteroskedasticity.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 2.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        # The intercept is not random, because we assume the intercept of the variance to be the same for all sites and sexes.
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
    # Set the likelihood with the priors we just created.
    likelihood2 = NormalLikelihood(mu2, sigma2)
    
    template_hbr_2 = HBR(
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
        # The sampler to use for the model.
        nuts_sampler="nutpie",
        # The likelihood function to use for the model.
        likelihood=likelihood2,
    )
    model2 = NormativeModel(
        # The regression model to use for the normative model.
        template_regression_model=template_hbr_2,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir="resources/compare_hbr/model2",
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
    )

.. code:: ipython3

    model1.fit_predict(train, test)
    model2.fit_predict(train, test)


.. parsed-literal::

    Process: 88956 - 2025-09-04 16:31:34 - Fitting models on 2 response variables.
    Process: 88956 - 2025-09-04 16:31:34 - Fitting model for WM-hypointensities.



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
                        <td>0</td>
                        <td>0.12</td>
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
                        <td>0</td>
                        <td>0.11</td>
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
                        <td>0</td>
                        <td>0.11</td>
                        <td>319</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.11</td>
                        <td>255</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>



.. parsed-literal::

    Process: 88956 - 2025-09-04 16:31:50 - Fitting model for Right-Lateral-Ventricle.



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
                        <td>0</td>
                        <td>0.14</td>
                        <td>191</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
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
                        <td>0</td>
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
                        <td>0</td>
                        <td>0.12</td>
                        <td>63</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>



.. parsed-literal::

    Process: 88956 - 2025-09-04 16:32:03 - Making predictions on 2 response variables.
    Process: 88956 - 2025-09-04 16:32:03 - Computing z-scores for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:03 - Computing z-scores for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:04 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:04 - Computing centiles for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:04 - Computing centiles for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:05 - Computing centiles for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:06 - Computing log-probabilities for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:06 - Computing log-probabilities for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:07 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:07 - Computing yhat for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:08 - Saving model to:
    	resources/compare_hbr/model1.
    Process: 88956 - 2025-09-04 16:32:08 - Making predictions on 2 response variables.
    Process: 88956 - 2025-09-04 16:32:08 - Computing z-scores for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:08 - Computing z-scores for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:08 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:08 - Computing centiles for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:08 - Computing centiles for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:09 - Computing centiles for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:10 - Computing log-probabilities for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:10 - Computing log-probabilities for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:10 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:10 - Computing yhat for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:10 - Saving model to:
    	resources/compare_hbr/model1.
    Process: 88956 - 2025-09-04 16:32:10 - Fitting models on 2 response variables.
    Process: 88956 - 2025-09-04 16:32:10 - Fitting model for WM-hypointensities.



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
        <p>Sampling for 15 seconds</p>
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
                        <td>0</td>
                        <td>0.10</td>
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
                        <td>0</td>
                        <td>0.09</td>
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
                        <td>0.09</td>
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
                        <td>5</td>
                        <td>0.10</td>
                        <td>95</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>



.. parsed-literal::

    Process: 88956 - 2025-09-04 16:32:31 - Fitting model for Right-Lateral-Ventricle.



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
                        <td>2</td>
                        <td>0.13</td>
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
                        <td>0</td>
                        <td>0.14</td>
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
                        <td>0.13</td>
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
                        <td>1</td>
                        <td>0.12</td>
                        <td>95</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>



.. parsed-literal::

    Process: 88956 - 2025-09-04 16:32:44 - Making predictions on 2 response variables.
    Process: 88956 - 2025-09-04 16:32:44 - Computing z-scores for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:44 - Computing z-scores for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:45 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:45 - Computing centiles for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:45 - Computing centiles for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:47 - Computing centiles for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:48 - Computing log-probabilities for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:48 - Computing log-probabilities for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:49 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:49 - Computing yhat for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:50 - Saving model to:
    	resources/compare_hbr/model2.
    Process: 88956 - 2025-09-04 16:32:50 - Making predictions on 2 response variables.
    Process: 88956 - 2025-09-04 16:32:50 - Computing z-scores for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:50 - Computing z-scores for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:50 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:51 - Computing centiles for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:51 - Computing centiles for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:52 - Computing centiles for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:53 - Computing log-probabilities for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:53 - Computing log-probabilities for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:53 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:54 - Computing yhat for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:54 - Saving model to:
    	resources/compare_hbr/model2.




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
      --xr-font-color0: var(
        --jp-content-font-color0,
        var(--pst-color-text-base rgba(0, 0, 0, 1))
      );
      --xr-font-color2: var(
        --jp-content-font-color2,
        var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
      );
      --xr-font-color3: var(
        --jp-content-font-color3,
        var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
      );
      --xr-border-color: var(
        --jp-border-color2,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 10))
      );
      --xr-disabled-color: var(
        --jp-layout-color3,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 40))
      );
      --xr-background-color: var(
        --jp-layout-color0,
        var(--pst-color-on-background, white)
      );
      --xr-background-color-row-even: var(
        --jp-layout-color1,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 5))
      );
      --xr-background-color-row-odd: var(
        --jp-layout-color2,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 15))
      );
    }
    
    html[theme="dark"],
    html[data-theme="dark"],
    body[data-theme="dark"],
    body.vscode-dark {
      --xr-font-color0: var(
        --jp-content-font-color0,
        var(--pst-color-text-base, rgba(255, 255, 255, 1))
      );
      --xr-font-color2: var(
        --jp-content-font-color2,
        var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
      );
      --xr-font-color3: var(
        --jp-content-font-color3,
        var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
      );
      --xr-border-color: var(
        --jp-border-color2,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 10))
      );
      --xr-disabled-color: var(
        --jp-layout-color3,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 40))
      );
      --xr-background-color: var(
        --jp-layout-color0,
        var(--pst-color-on-background, #111111)
      );
      --xr-background-color-row-even: var(
        --jp-layout-color1,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 5))
      );
      --xr-background-color-row-odd: var(
        --jp-layout-color2,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 15))
      );
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
      border: 2px solid transparent !important;
    }
    
    .xr-section-item input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }
    
    .xr-section-item input:focus + label {
      border: 2px solid var(--xr-font-color0) !important;
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
      border-color: var(--xr-background-color-row-odd);
      margin-bottom: 0;
      padding-top: 2px;
    }
    
    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }
    
    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
      border-color: var(--xr-background-color-row-even);
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
      border-top: 2px dotted var(--xr-background-color);
      padding-bottom: 20px !important;
      padding-top: 10px !important;
    }
    
    .xr-var-attrs-in + label,
    .xr-var-data-in + label,
    .xr-index-data-in + label {
      padding: 0 1px;
    }
    
    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data,
    .xr-index-data-in:checked ~ .xr-index-data {
      display: block;
    }
    
    .xr-var-data > table {
      float: right;
    }
    
    .xr-var-data > pre,
    .xr-index-data > pre,
    .xr-var-data > table > tbody > tr {
      background-color: transparent !important;
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
    
    .xr-var-attrs-in:checked + label > .xr-icon-file-text2,
    .xr-var-data-in:checked + label > .xr-icon-database,
    .xr-index-data-in:checked + label > .xr-icon-database {
      color: var(--xr-font-color0);
      filter: drop-shadow(1px 1px 5px var(--xr-font-color2));
      stroke-width: 0.8px;
    }
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 66kB
    Dimensions:            (observations: 216, response_vars: 2, covariates: 1,
                            batch_effect_dims: 2, centile: 5, statistic: 11)
    Coordinates:
      * observations       (observations) int64 2kB 756 769 692 616 ... 751 470 1043
      * response_vars      (response_vars) &lt;U23 184B &#x27;WM-hypointensities&#x27; &#x27;Right-...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) &lt;U8 352B &#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;
    Data variables:
        subjects           (observations) object 2kB &#x27;Munchen_sub96752&#x27; ... &#x27;Quee...
        Y                  (observations, response_vars) float64 3kB 2.721e+03 .....
        X                  (observations, covariates) float64 2kB 63.0 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 29kB &#x27;F&#x27; ... &#x27;Q...
        Z                  (observations, response_vars) float64 3kB 0.6006 ... 1...
        centiles           (centile, observations, response_vars) float64 17kB -5...
        logp               (observations, response_vars) float64 3kB -1.696 ... -...
        Yhat               (observations, response_vars) float64 3kB 1.84e+03 ......
        statistics         (response_vars, statistic) float64 176B 0.3673 ... 0.8771
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [np.str_(&#x27;F&#x27;), np.str_(&#x27;...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(7.88)...
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): {np.str_(...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-000a496e-3958-4c89-82e4-01c081751dd5' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-000a496e-3958-4c89-82e4-01c081751dd5' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 2</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>centile</span>: 5</li><li><span class='xr-has-index'>statistic</span>: 11</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-6cdd9f80-6ac4-43f5-a26b-aa84dba1fc92' class='xr-section-summary-in' type='checkbox'  checked><label for='section-6cdd9f80-6ac4-43f5-a26b-aa84dba1fc92' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-25a2b622-aec6-42c4-914a-c2e65e72499a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-25a2b622-aec6-42c4-914a-c2e65e72499a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6e50f696-c02a-4f8f-81f6-be3278805cdf' class='xr-var-data-in' type='checkbox'><label for='data-6e50f696-c02a-4f8f-81f6-be3278805cdf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043], shape=(216,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; &#x27;Right-Late...</div><input id='attrs-745edc4d-3d4b-4856-ae2f-ead91577d7df' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-745edc4d-3d4b-4856-ae2f-ead91577d7df' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8f93755f-0c6e-4971-850d-c0ca8766a2f3' class='xr-var-data-in' type='checkbox'><label for='data-8f93755f-0c6e-4971-850d-c0ca8766a2f3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-46bf0813-7a6e-48c8-9c24-efca2a6593d6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-46bf0813-7a6e-48c8-9c24-efca2a6593d6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-305ac3ef-4fb9-4616-9ee6-fc75f1caf534' class='xr-var-data-in' type='checkbox'><label for='data-305ac3ef-4fb9-4616-9ee6-fc75f1caf534' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-f64b66e2-7aa9-45e5-b582-d931e7f1b731' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f64b66e2-7aa9-45e5-b582-d931e7f1b731' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a72a6934-4c57-48c6-8319-b0e5b56dc680' class='xr-var-data-in' type='checkbox'><label for='data-a72a6934-4c57-48c6-8319-b0e5b56dc680' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-dafd2017-27ca-48a8-9f3b-c9feb7027876' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dafd2017-27ca-48a8-9f3b-c9feb7027876' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cfa67139-6eef-499e-aa2f-0afd980ebeaa' class='xr-var-data-in' type='checkbox'><label for='data-cfa67139-6eef-499e-aa2f-0afd980ebeaa' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-932d46f8-e71c-4c81-a795-632d94644bc8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-932d46f8-e71c-4c81-a795-632d94644bc8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1d38c359-f4f2-427d-862e-7bac41f6de67' class='xr-var-data-in' type='checkbox'><label for='data-1d38c359-f4f2-427d-862e-7bac41f6de67' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1338abff-a419-4c1c-bb7e-0fc2c402b674' class='xr-section-summary-in' type='checkbox'  checked><label for='section-1338abff-a419-4c1c-bb7e-0fc2c402b674' class='xr-section-summary' >Data variables: <span>(9)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subjects</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Munchen_sub96752&#x27; ... &#x27;Queensla...</div><input id='attrs-2784df01-ef44-4787-a631-929255be757f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2784df01-ef44-4787-a631-929255be757f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5eff7476-b210-4e94-9652-acf216fc8d46' class='xr-var-data-in' type='checkbox'><label for='data-5eff7476-b210-4e94-9652-acf216fc8d46' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Munchen_sub96752&#x27;, &#x27;NewYork_a_sub18638&#x27;, &#x27;Leiden_2200_sub87320&#x27;,
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
           &#x27;Cambridge_Buckner_sub59729&#x27;, &#x27;Queensland_sub86245&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.721e+03 1.289e+04 ... 1.07e+04</div><input id='attrs-ea0c74b1-803b-4385-b2c9-f7ec427d7c1b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ea0c74b1-803b-4385-b2c9-f7ec427d7c1b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d359afc9-f4b8-4b4b-9614-9c9a3c71594a' class='xr-var-data-in' type='checkbox'><label for='data-d359afc9-f4b8-4b4b-9614-9c9a3c71594a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 2721.4, 12891.6],
           [ 1143.1,  9919.1],
           [  955.8,  7477.3],
           [ 1473.9, 14302.1],
           [  757.8,  4119.3],
           [  871.1,  5030.9],
           [ 1207.3, 17866.4],
           [  595. ,  5007.9],
           [  682.4,  7286.6],
           [  445.1,  5742.9],
           [ 1620. ,  3713.7],
           [  602.8,  5301.2],
           [ 1432.5,  4429.7],
           [ 1908.2,  3578.1],
           [ 1834. ,  3271.9],
           [  459.6,  3985.8],
           [ 1210. ,  8721.3],
           [  845.9,  6593.1],
           [  995.2,  7040.2],
           [ 1734.7,  4014.8],
    ...
           [  785.8,  5709. ],
           [ 2240.1,  4366.6],
           [  758.1,  6529.8],
           [ 1440.5,  6705.3],
           [  818.6,  9383.3],
           [ 3769.9, 15864.4],
           [  880.2,  4370.2],
           [  823.9,  6379. ],
           [ 2113.9, 10722.5],
           [  741.9,  8801.7],
           [ 1333.9,  6980. ],
           [  707.3,  5680.7],
           [ 1134.1,  5592.2],
           [  438.6,  6330. ],
           [  966.3,  9215.5],
           [  424.3,  4511.1],
           [  604.7,  7590.8],
           [ 2343.2, 17192.3],
           [ 2721.7,  6086. ],
           [  703.5, 10700.3]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 23.27 22.0 ... 72.0 23.0 23.0</div><input id='attrs-dfd45032-9ac2-431b-b398-4685fb0b3f45' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dfd45032-9ac2-431b-b398-4685fb0b3f45' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-900d72fb-9c13-4810-86bf-7df84ef1a904' class='xr-var-data-in' type='checkbox'><label for='data-900d72fb-9c13-4810-86bf-7df84ef1a904' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[63.  ],
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
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-56e905de-e5e2-4aa6-8abf-edff9e91dde6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-56e905de-e5e2-4aa6-8abf-edff9e91dde6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0b2cd5b6-aef0-4f9b-a2d5-d31e9590f4ae' class='xr-var-data-in' type='checkbox'><label for='data-0b2cd5b6-aef0-4f9b-a2d5-d31e9590f4ae' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
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
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.6006 0.2981 ... -1.115 1.153</div><input id='attrs-0f4939bf-8323-4126-9abb-c26d517e4c31' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0f4939bf-8323-4126-9abb-c26d517e4c31' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9394a109-8c8f-4d14-87d3-e0d4850eb33e' class='xr-var-data-in' type='checkbox'><label for='data-9394a109-8c8f-4d14-87d3-e0d4850eb33e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 6.00581077e-01,  2.98053432e-01],
           [ 1.41406449e-02,  8.91969606e-01],
           [ 2.43845549e-01,  5.18227738e-01],
           [ 6.20683256e-02,  1.51608902e+00],
           [-8.45779389e-01, -1.29356601e+00],
           [-7.98414078e-01, -7.24869454e-01],
           [ 4.56342381e-01,  3.83843981e+00],
           [-4.13452256e-02, -4.20846117e-01],
           [-8.51505984e-01,  5.97349095e-01],
           [-9.67533456e-01, -3.90235406e-01],
           [ 1.11243276e+00, -1.12192623e+00],
           [-8.57542805e-01, -2.48511422e-01],
           [ 1.05597231e+00, -3.82298134e-01],
           [ 1.86263880e+00, -7.86966070e-01],
           [ 1.07626923e+00, -1.18870207e+00],
           [-1.46565705e+00, -6.44716006e-01],
           [-6.27913124e-01,  4.96451294e-01],
           [-4.36774846e-01,  4.85454488e-01],
           [-4.50446153e-01, -2.03681923e-01],
           [ 1.25307473e+00, -4.12662811e-01],
    ...
           [ 1.34668233e-01, -2.40613768e-02],
           [ 2.22293936e+00, -8.59350052e-01],
           [-1.37934261e+00,  1.84073678e-01],
           [-4.87022812e-01,  1.54894279e-01],
           [-9.24964561e-01,  5.81362328e-01],
           [ 1.85472495e-01,  1.09893040e-01],
           [-8.15264554e-01, -4.61039891e-01],
           [ 7.93911764e-02, -4.10590349e-01],
           [ 2.34973190e+00,  8.90929727e-01],
           [-6.94105712e-01,  1.32023165e+00],
           [-2.63005685e-01,  6.66721255e-02],
           [-1.51252358e+00, -1.11370587e-01],
           [-3.57399822e-01,  9.32161640e-02],
           [-4.34143934e-01,  4.74700597e-02],
           [-5.44698340e-01,  8.24752534e-01],
           [-1.54999028e+00, -4.86552599e-01],
           [-1.40722958e+00,  1.31315600e-01],
           [-3.98528063e-01,  3.32590719e-01],
           [ 3.61164294e+00, -3.47581715e-01],
           [-1.11470098e+00,  1.15295269e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-580.7 2.498e+03 ... 1.149e+04</div><input id='attrs-249890fd-bcb9-4469-b1ca-24b7956aadd8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-249890fd-bcb9-4469-b1ca-24b7956aadd8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6cda5a0b-acdc-466d-b1d0-9e55073e4c6c' class='xr-var-data-in' type='checkbox'><label for='data-6cda5a0b-acdc-466d-b1d0-9e55073e4c6c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ -580.67936177,  2498.45556286],
            [  421.39508988,  1512.78895223],
            [  450.61802768,  1535.16698691],
            ...,
            [ -501.31357762,  2622.24148278],
            [  427.75965688,  1519.56736103],
            [  427.75965688,  1519.56736103]],
    
           [[  848.55163654,  7907.70400788],
            [  844.63472375,  4478.47369657],
            [  875.02544203,  4386.74906794],
            ...,
            [ 1677.88478271,  9474.32554995],
            [  850.96168253,  4460.66561917],
            [  850.96168253,  4460.66561917]],
    
           [[ 1841.99510874, 11667.61585109],
            [ 1138.82415286,  6539.89004799],
            [ 1170.02658309,  6368.8538552 ],
            ...,
            [ 3192.6227    , 14237.13729665],
            [ 1145.12497056,  6504.99216197],
            [ 1145.12497056,  6504.99216197]],
    
           [[ 2835.43858094, 15427.52769429],
            [ 1433.01358196,  8601.30639941],
            [ 1465.02772414,  8350.95864245],
            ...,
            [ 4707.36061729, 18999.94904335],
            [ 1439.28825858,  8549.31870477],
            [ 1439.28825858,  8549.31870477]],
    
           [[ 4264.66957925, 20836.77613932],
            [ 1856.25321584, 11566.99114374],
            [ 1889.43513849, 11202.54072349],
            ...,
            [ 6886.55897762, 25852.03311052],
            [ 1862.49028424, 11490.4169629 ],
            [ 1862.49028424, 11490.4169629 ]]], shape=(5, 216, 2))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.696 -1.32 ... -0.8063 -1.641</div><input id='attrs-c58162c6-5cfd-4b82-b937-da60a5b8e6a7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c58162c6-5cfd-4b82-b937-da60a5b8e6a7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-842aa43f-3fd6-4f72-844d-2c8292195257' class='xr-var-data-in' type='checkbox'><label for='data-842aa43f-3fd6-4f72-844d-2c8292195257' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -1.69607946,  -1.3197607 ],
           [ -0.29235793,  -1.30146718],
           [ -0.4151767 ,  -0.72055835],
           [ -0.80483313,  -2.31018476],
           [ -1.79226416,  -2.23571906],
           [ -0.49009894,  -0.79885594],
           [ -0.30355467,  -8.98333154],
           [ -0.93773872,  -0.92240653],
           [ -0.98484479,  -0.68849584],
           [ -1.56720694,  -1.25040579],
           [ -0.71166795,  -0.95353907],
           [ -1.06744748,  -0.759356  ],
           [ -0.42722099,  -0.76599492],
           [ -2.01237245,  -1.25567488],
           [ -1.32135666,  -1.4477959 ],
           [ -1.70351798,  -0.92771896],
           [ -0.29907815,  -0.97042493],
           [ -0.73230235,  -0.5940885 ],
           [ -0.52020565,  -0.94074975],
           [ -0.84901381,  -0.75875242],
    ...
           [ -0.73827274,  -0.63508544],
           [ -2.90762819,  -0.77982453],
           [ -0.73960802,  -0.6507542 ],
           [ -0.59388127,  -0.74099683],
           [ -0.51318643,  -1.07902382],
           [ -2.00583281,  -1.5908576 ],
           [ -0.51504713,  -0.88107789],
           [ -0.49585029,  -0.8355864 ],
           [ -2.84185634,  -1.37966658],
           [ -0.96054037,  -1.13996752],
           [ -0.34834684,  -0.64358438],
           [ -0.85605582,  -0.67673257],
           [ -0.3716804 ,  -0.56937224],
           [ -1.52449026,  -0.71569619],
           [ -0.43996052,  -1.17115186],
           [ -1.57643575,  -0.94161991],
           [ -1.07628083,  -0.74227709],
           [ -2.01736186,  -1.62926077],
           [ -6.84411001,  -0.68985355],
           [ -0.80626532,  -1.64062338]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.84e+03 1.128e+04 ... 7.291e+03</div><input id='attrs-c099f752-9943-4fcc-89b1-f172c9b0b977' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c099f752-9943-4fcc-89b1-f172c9b0b977' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-caecd222-6c70-4e4d-8fb4-090fe069b527' class='xr-var-data-in' type='checkbox'><label for='data-caecd222-6c70-4e4d-8fb4-090fe069b527' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1839.96675798, 11275.29195664],
           [ 1137.67491803,  7261.04175869],
           [  862.84285654,  5987.56710723],
           [ 1433.08279207,  7958.62718965],
           [ 1998.48789445, 11157.7470248 ],
           [ 1173.13677632,  7174.51240197],
           [ 1030.98756063,  7143.50326947],
           [  610.9115235 ,  6357.17686976],
           [ 1011.66207648,  5617.89102922],
           [ 1286.06553902,  7495.94174483],
           [ 1181.53567451,  6762.22180079],
           [  927.21371096,  6035.89754834],
           [ 1016.23275353,  5468.37671202],
           [ 1199.57776917,  6102.376873  ],
           [ 1316.10736967,  7821.78979017],
           [ 1026.49896659,  5786.69282656],
           [ 1449.49788269,  7294.35665023],
           [ 1022.40587343,  5308.6931826 ],
           [ 1203.56151319,  7807.93658636],
           [ 1212.38630961,  5079.74253019],
    ...
           [  733.71499712,  5776.12561849],
           [ 1363.86699906,  6701.68771157],
           [ 1284.19496171,  6000.51156146],
           [ 1624.37438011,  6221.11413985],
           [ 1168.04189796,  7565.87741857],
           [ 3342.22930148, 15082.49790537],
           [ 1191.15161793,  5695.26586654],
           [  792.97025168,  7759.05982548],
           [ 1175.52966998,  7656.22414715],
           [ 1022.40587343,  5308.6931826 ],
           [ 1437.46676642,  6799.21338412],
           [ 1284.19496171,  6000.51156146],
           [ 1278.33696533,  5345.68476593],
           [  602.49429233,  6185.32283059],
           [ 1176.96499747,  6911.736118  ],
           [ 1008.82921346,  5991.05692305],
           [ 1137.65374544,  7204.68059995],
           [ 3232.33585956, 14908.02390699],
           [ 1355.46810087,  7113.97831274],
           [ 1125.28165256,  7290.69697593]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.3673 0.04444 ... 0.7675 0.8771</div><input id='attrs-4787d8c0-a170-406b-a52e-f55bc8b74531' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4787d8c0-a170-406b-a52e-f55bc8b74531' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-deaf878d-d7f3-46ba-afc8-cc8805ba6ac7' class='xr-var-data-in' type='checkbox'><label for='data-deaf878d-d7f3-46ba-afc8-cc8805ba6ac7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[3.67312216e-01, 4.44444444e-02, 1.62233340e+00, 2.34693785e-01,
            8.84575255e-01, 3.64235179e-01, 5.90885645e-01, 4.91234727e-01,
            1.60553357e-14, 6.35764821e-01, 9.63971537e-01],
           [2.33405043e-01, 5.37037037e-02, 3.21501054e+00, 1.03237708e-01,
            1.33138303e+00, 2.32486430e-01, 8.89925654e-01, 2.79495956e-01,
            3.08292901e-05, 7.67513570e-01, 8.77088446e-01]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-41926210-00be-4e95-837c-846297704611' class='xr-section-summary-in' type='checkbox'  ><label for='section-41926210-00be-4e95-837c-846297704611' class='xr-section-summary' >Indexes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-8718becd-14a3-401e-bb83-94fd0d0b3b48' class='xr-index-data-in' type='checkbox'/><label for='index-8718becd-14a3-401e-bb83-94fd0d0b3b48' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([ 756,  769,  692,  616,   35,  164,  680,  331,  299,  727,
           ...
             27,  959,   29,  346,  304,  264,  798,  751,  470, 1043],
          dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;, length=216))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-24ca17f3-c792-4dd0-96cd-5a1b38893cd8' class='xr-index-data-in' type='checkbox'/><label for='index-24ca17f3-c792-4dd0-96cd-5a1b38893cd8' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;], dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-e3a393a1-6d3b-4b43-9d99-c97800b2aa6b' class='xr-index-data-in' type='checkbox'/><label for='index-e3a393a1-6d3b-4b43-9d99-c97800b2aa6b' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-fa4d045c-b221-41eb-8b58-b62f4881d12c' class='xr-index-data-in' type='checkbox'/><label for='index-fa4d045c-b221-41eb-8b58-b62f4881d12c' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>centile</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-34155a3b-0d28-4df7-affa-d7dfa313326f' class='xr-index-data-in' type='checkbox'/><label for='index-34155a3b-0d28-4df7-affa-d7dfa313326f' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0.05, 0.25, 0.5, 0.75, 0.95], dtype=&#x27;float64&#x27;, name=&#x27;centile&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-2c913752-ed20-41c2-8ac5-1e23bc1eb797' class='xr-index-data-in' type='checkbox'/><label for='index-2c913752-ed20-41c2-8ac5-1e23bc1eb797' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d1acfa7f-57fb-4399-88f4-0a40abf00748' class='xr-section-summary-in' type='checkbox'  checked><label for='section-d1acfa7f-57fb-4399-88f4-0a40abf00748' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [np.str_(&#x27;F&#x27;), np.str_(&#x27;M&#x27;)], np.str_(&#x27;site&#x27;): [np.str_(&#x27;AnnArbor_a&#x27;), np.str_(&#x27;AnnArbor_b&#x27;), np.str_(&#x27;Atlanta&#x27;), np.str_(&#x27;Baltimore&#x27;), np.str_(&#x27;Bangor&#x27;), np.str_(&#x27;Beijing_Zang&#x27;), np.str_(&#x27;Berlin_Margulies&#x27;), np.str_(&#x27;Cambridge_Buckner&#x27;), np.str_(&#x27;Cleveland&#x27;), np.str_(&#x27;ICBM&#x27;), np.str_(&#x27;Leiden_2180&#x27;), np.str_(&#x27;Leiden_2200&#x27;), np.str_(&#x27;Milwaukee_b&#x27;), np.str_(&#x27;Munchen&#x27;), np.str_(&#x27;NewYork_a&#x27;), np.str_(&#x27;NewYork_a_ADHD&#x27;), np.str_(&#x27;Newark&#x27;), np.str_(&#x27;Oulu&#x27;), np.str_(&#x27;Oxford&#x27;), np.str_(&#x27;PaloAlto&#x27;), np.str_(&#x27;Pittsburgh&#x27;), np.str_(&#x27;Queensland&#x27;), np.str_(&#x27;SaintLouis&#x27;)]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x154adf600&gt;, {np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): 589, np.str_(&#x27;M&#x27;): 489}, np.str_(&#x27;site&#x27;): {np.str_(&#x27;AnnArbor_a&#x27;): 24, np.str_(&#x27;AnnArbor_b&#x27;): 32, np.str_(&#x27;Atlanta&#x27;): 28, np.str_(&#x27;Baltimore&#x27;): 23, np.str_(&#x27;Bangor&#x27;): 20, np.str_(&#x27;Beijing_Zang&#x27;): 198, np.str_(&#x27;Berlin_Margulies&#x27;): 26, np.str_(&#x27;Cambridge_Buckner&#x27;): 198, np.str_(&#x27;Cleveland&#x27;): 31, np.str_(&#x27;ICBM&#x27;): 85, np.str_(&#x27;Leiden_2180&#x27;): 12, np.str_(&#x27;Leiden_2200&#x27;): 19, np.str_(&#x27;Milwaukee_b&#x27;): 46, np.str_(&#x27;Munchen&#x27;): 15, np.str_(&#x27;NewYork_a&#x27;): 83, np.str_(&#x27;NewYork_a_ADHD&#x27;): 25, np.str_(&#x27;Newark&#x27;): 19, np.str_(&#x27;Oulu&#x27;): 102, np.str_(&#x27;Oxford&#x27;): 22, np.str_(&#x27;PaloAlto&#x27;): 17, np.str_(&#x27;Pittsburgh&#x27;): 3, np.str_(&#x27;Queensland&#x27;): 19, np.str_(&#x27;SaintLouis&#x27;): 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(85.0)}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(85.0)}}, np.str_(&#x27;M&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(9.21), &#x27;max&#x27;: np.float64(78.0)}}}, np.str_(&#x27;site&#x27;): {np.str_(&#x27;AnnArbor_a&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(13.41), &#x27;max&#x27;: np.float64(40.98)}}, np.str_(&#x27;AnnArbor_b&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(79.0)}}, np.str_(&#x27;Atlanta&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(22.0), &#x27;max&#x27;: np.float64(57.0)}}, np.str_(&#x27;Baltimore&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(40.0)}}, np.str_(&#x27;Bangor&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(38.0)}}, np.str_(&#x27;Beijing_Zang&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(26.0)}}, np.str_(&#x27;Berlin_Margulies&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(23.0), &#x27;max&#x27;: np.float64(44.0)}}, np.str_(&#x27;Cambridge_Buckner&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(30.0)}}, np.str_(&#x27;Cleveland&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(24.0), &#x27;max&#x27;: np.float64(60.0)}}, np.str_(&#x27;ICBM&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(85.0)}}, np.str_(&#x27;Leiden_2180&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(27.0)}}, np.str_(&#x27;Leiden_2200&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(28.0)}}, np.str_(&#x27;Milwaukee_b&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(44.0), &#x27;max&#x27;: np.float64(65.0)}}, np.str_(&#x27;Munchen&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(63.0), &#x27;max&#x27;: np.float64(74.0)}}, np.str_(&#x27;NewYork_a&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(49.16)}}, np.str_(&#x27;NewYork_a_ADHD&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(20.69), &#x27;max&#x27;: np.float64(50.9)}}, np.str_(&#x27;Newark&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(21.0), &#x27;max&#x27;: np.float64(39.0)}}, np.str_(&#x27;Oulu&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(23.0)}}, np.str_(&#x27;Oxford&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(35.0)}}, np.str_(&#x27;PaloAlto&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(22.0), &#x27;max&#x27;: np.float64(46.0)}}, np.str_(&#x27;Pittsburgh&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(25.0), &#x27;max&#x27;: np.float64(47.0)}}, np.str_(&#x27;Queensland&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(34.0)}}, np.str_(&#x27;SaintLouis&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: np.float64(21.0), &#x27;max&#x27;: np.float64(29.0)}}}}</dd></dl></div></li></ul></div></div>



.. code:: ipython3

    # Delete references to model objects to ensure what follows will work for models saved to disk too
    del model1
    del model2

.. code:: ipython3

    dct = {"model1": "resources/compare_hbr/model1", "model2": "resources/compare_hbr/model2"}
    comparison = compare_hbr_models(dct)


.. parsed-literal::

    Process: 88956 - 2025-09-04 16:32:54 - Dataset "synthesized" created.
        - 1078 observations
        - 1078 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        
    Process: 88956 - 2025-09-04 16:32:54 - Synthesizing data for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:54 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:55 - Synthesizing data for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:55 - Making predictions on 2 response variables.
    Process: 88956 - 2025-09-04 16:32:55 - Computing z-scores for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:55 - Computing z-scores for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:55 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:56 - Computing centiles for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:56 - Computing centiles for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:57 - Computing centiles for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:59 - Computing log-probabilities for 2 response variables.
    Process: 88956 - 2025-09-04 16:32:59 - Computing log-probabilities for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:32:59 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:32:59 - Computing yhat for 2 response variables.
    Process: 88956 - 2025-09-04 16:33:00 - Dataset "synthesized" created.
        - 1078 observations
        - 1078 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        
    Process: 88956 - 2025-09-04 16:33:00 - Synthesizing data for 2 response variables.
    Process: 88956 - 2025-09-04 16:33:00 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:33:00 - Synthesizing data for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:33:01 - Making predictions on 2 response variables.
    Process: 88956 - 2025-09-04 16:33:01 - Computing z-scores for 2 response variables.
    Process: 88956 - 2025-09-04 16:33:01 - Computing z-scores for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:33:01 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:33:02 - Computing centiles for 2 response variables.
    Process: 88956 - 2025-09-04 16:33:02 - Computing centiles for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:33:04 - Computing centiles for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:33:06 - Computing log-probabilities for 2 response variables.
    Process: 88956 - 2025-09-04 16:33:06 - Computing log-probabilities for WM-hypointensities.
    Process: 88956 - 2025-09-04 16:33:07 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 88956 - 2025-09-04 16:33:07 - Computing yhat for 2 response variables.



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/arviz/stats/stats.py:1057: RuntimeWarning: overflow encountered in exp
      weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/arviz/stats/stats.py:797: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/arviz/stats/stats.py:797: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/arviz/stats/stats.py:1057: RuntimeWarning: overflow encountered in exp
      weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/arviz/stats/stats.py:797: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/arviz/stats/stats.py:1057: RuntimeWarning: overflow encountered in exp
      weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/arviz/stats/stats.py:797: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(


.. code:: ipython3

    for k, v in comparison.items():
        print(k)
        display(v)


.. parsed-literal::

    Right-Lateral-Ventricle



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
          <th></th>
          <th>rank</th>
          <th>elpd_loo</th>
          <th>p_loo</th>
          <th>elpd_diff</th>
          <th>weight</th>
          <th>se</th>
          <th>dse</th>
          <th>warning</th>
          <th>scale</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>model2</th>
          <td>0</td>
          <td>-1788.316443</td>
          <td>117.191582</td>
          <td>0.000000</td>
          <td>0.53159</td>
          <td>38.451843</td>
          <td>0.000000</td>
          <td>True</td>
          <td>log</td>
        </tr>
        <tr>
          <th>model1</th>
          <td>1</td>
          <td>-1866.148969</td>
          <td>159.648880</td>
          <td>77.832526</td>
          <td>0.46841</td>
          <td>47.306735</td>
          <td>61.681774</td>
          <td>True</td>
          <td>log</td>
        </tr>
      </tbody>
    </table>
    </div>


.. parsed-literal::

    WM-hypointensities



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
          <th></th>
          <th>rank</th>
          <th>elpd_loo</th>
          <th>p_loo</th>
          <th>elpd_diff</th>
          <th>weight</th>
          <th>se</th>
          <th>dse</th>
          <th>warning</th>
          <th>scale</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>model2</th>
          <td>0</td>
          <td>-1746.725077</td>
          <td>120.637510</td>
          <td>0.000000</td>
          <td>0.52086</td>
          <td>41.409869</td>
          <td>0.00000</td>
          <td>True</td>
          <td>log</td>
        </tr>
        <tr>
          <th>model1</th>
          <td>1</td>
          <td>-1803.947305</td>
          <td>120.159585</td>
          <td>57.222227</td>
          <td>0.47914</td>
          <td>43.802446</td>
          <td>59.87482</td>
          <td>True</td>
          <td>log</td>
        </tr>
      </tbody>
    </table>
    </div>

