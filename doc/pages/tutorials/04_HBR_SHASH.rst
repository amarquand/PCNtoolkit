Normative Modelling: Hierarchical Bayesian Regression with SHASH likelihood
===========================================================================

Welcome to this tutorial notebook that will go through the fitting and
evaluation of Normative models with a Hierarchical Bayesian Regression
model using a SHASH likelihood.

Let’s jump right in.

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
        make_prior,
        plot_centiles,
        plot_qq,
        plot_ridge,
    )
    
    import numpy as np
    import pcntoolkit.util.output
    import seaborn as sns
    
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



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_6_0.png


Creating a Normative model
--------------------------

A normative model has a regression model for each response variable. We
provide a template regression model which is copied for each response
variable.

A template regression model can be anything that extends the
``RegressionModel``. We provide a number of built-in regression models,
but you can also create your own.

Here we use the ``HBR`` class, which implements a Hierarchical Bayesian
Regression model.

Likelihoods
~~~~~~~~~~~

``HBR`` models are composed of a likelihood and a number of priors on
the parameters of the likelihood. The PCNtoolkit offers a number of
likelihood functions: 1. NormallLikelihood: Good for modeling data that
is (approximately) normally distributed. 2. SHASHbLikelihood: Good for
modeling data that is heavily skewed, or tailed. 3. BetaLikelihood: Good
for modeling data that is bounded, e.g. between 0 and 1.

Likelihood parameters
~~~~~~~~~~~~~~~~~~~~~

Each of these likelihoods takes their own set of parameters, and for
each, we have to set a prior: 1. NormalLikelihood: - ``mu``: The mean of
the normal distribution. - ``sigma``: The standard deviation of the
normal distribution. 2. SHASHbLikelihood: - ``mu``: The mean of the
skew-normal distribution. - ``sigma``: The standard deviation of the
skew-normal distribution. - ``epsilon``: The skewness parameter of the
skew-normal distribution. - ``delta``: The tail thickness (or kurtosis)
of the skew-normal distribution. 3. BetaLikelihood: - ``alpha``: The
shape parameter of the beta distribution. - ``beta``: The scale
parameter of the beta distribution.

Configuring likelihood parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each likelihood parameter needs to be configured. The defaults should
work reasonably well for most cases, at least where the data is
standardized. Here’s a quick guide to configuring the likelihood
parameters yourself, using the ``make_prior`` function.

1. Is your parameter a function of the covariates? If so, you set the
   ``linear`` parameter to ``True``.

   1. If so, you can choose the basis expansion to use for the
      parameter: BSplineBasisFunction, LinearBasisFunction, or
      PolynomialBasisFunction.
   2. Also, determine whether the slope and intercept of the prior have
      a random effect or not. Here’s an example of a linear prior with a
      bspline basis expansion and a random effect in the intercept.

.. code:: python

   mu = make_prior('mu', linear=True, basis_function=BSplineBasisFunction(degree=3, nknots=5), intercept = make_prior('intercept_mu', random=True))

2. If your parameter is not a function of the covariates, you have to
   decide whether the parameter itself has a random effect or not.
   Here’s an example of a prior with a random effect.

.. code:: python

   epsilon = make_prior('epsilon', random=True)

3. Some parameters (such as sigma) need to be strictly positive, which
   we can enforce with a mapping. Here’s an example of a prior with a
   mapping to the positive real line.

.. code:: python

   # The mapping_params are (horizontal shift, scaling, vertical shift)
   sigma = make_prior('sigma', mapping='softplus', mapping_params=(0, 5, 0))

.. code:: ipython3

    # Mini demo of the mapping params
    xsp = np.linspace(-7, 7, 100)
    softplus = lambda x: np.log(1 + np.exp(x))
    paramaterized_softplus = lambda x, a, b, c: softplus((x - a) / b) * b + c
    plt.plot(xsp, paramaterized_softplus(xsp, 0, 1, 0), label="no mapping")
    plt.plot(xsp, paramaterized_softplus(xsp, 1.5, 1, 0), label="horizontal shift of 1.5")
    plt.plot(xsp, paramaterized_softplus(xsp, 0, 1, 1), label="vertical shift of 1")
    plt.plot(xsp, paramaterized_softplus(xsp, 0, 2, 0), label="scale with a factor of 2")
    plt.legend()
    plt.show()



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_8_0.png


4. Any non-linear parameters can be further configured with
   ``dist_name`` and ``dist_params``. Here’s an example of a prior with
   a gamma distribution.

.. code:: python

   alpha = make_prior('alpha', dist_name='gamma', dist_params=(1, 1))

We currently support the following distributions: - Normal - HalfNormal
- LogNormal - Uniform - Gamma

The order of the parameters is important, and follows the order of the
parameters in the corresponding distributions in PyMC.

Creating a HBR model
~~~~~~~~~~~~~~~~~~~~

Here’s a thoroughly commented example of a HBR model with a SHASH
Likelihood, which we will use to model our response variable.

.. code:: ipython3

    # The SHASHb likelihood is a bit more flexible than the Normal likelihood, and takes four parameters, mu, sigma, epsilon, and delta.
    # Mu and sigma fulfill the same role as in the Normal likelihood, namely the mean and standard deviation of the distribution.
    # Epsilon and delta are parameters that control the skewness and kurtosis of the distribution.
    
    # SHASHb model with fixed values for epsilon and delta
    
    mu = make_prior(
        # Mu is linear because we want to allow the mean to vary as a function of the covariates.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 10.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 10.0)),
        # The intercept is random, because we expect the intercept to vary between sites and sexes.
        intercept=make_prior(
            random=True,
            # Mu is the mean of the intercept, which is normally distributed with a mean of 0 and a standard deviation of 1.
            mu=make_prior(dist_name="Normal", dist_params=(0.0, 1.0)),
            # Sigma is the scale at which the intercepts vary. It is a positive parameter, so we have to map it to the positive domain.
            sigma=make_prior(dist_name="Normal", dist_params=(0.0, 1.0), mapping="softplus", mapping_params=(0.0, 3.0)),
        ),
        # We use a B-spline basis function to allow for non-linearity in the mean.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    sigma = make_prior(
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
    
    epsilon = make_prior(
        # Epsilon is assumed to follow a normal distribution, with a mean of 0 and a standard deviation of 1.
        dist_name="Normal",
        dist_params=(0.0, 1.0),
    )
    
    delta = make_prior(
        # Delta is sampled from a normal distribution, with a mean of 1 and a standard deviation of 1, and then mapped to the positive real line using a softplus function.
        dist_name="Normal",
        dist_params=(1.0, 1.0),
        mapping="softplus",
        # We apply a softplus mapping to the delta parameter, to ensure that it is strictly positive.
        mapping_params=(
            0.0,  # Horizontal shift
            3.0,  # Scale for smoothness
            0.6,  # We need to provide a vertical shift as well, because the SHASH mapping goes a bit wild with low values for delta
        ),
    )
    
    shashb1_regression_model = HBR(
        name="template",
        cores=16,
        progressbar=True,
        draws=1500,
        tune=500,
        chains=4,
        nuts_sampler="nutpie",
        likelihood=SHASHbLikelihood(mu, sigma, epsilon, delta),
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
        template_regression_model=shashb1_regression_model,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir="resources/hbr_SHASH/save_dir",
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
        <p>Sampling for 2 minutes</p>
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
                        <td>0.01</td>
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
                        <td>0</td>
                        <td>0.01</td>
                        <td>1023</td>
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
                        <td>0.01</td>
                        <td>1023</td>
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
                        <td>0.01</td>
                        <td>255</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>




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
        <p>Sampling for 2 minutes</p>
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
                        <td>0.01</td>
                        <td>1023</td>
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
                        <td>0.01</td>
                        <td>1023</td>
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
                        <td>0.01</td>
                        <td>1023</td>
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
                        <td>0.01</td>
                        <td>1023</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>




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
        <p>Sampling for 3 minutes</p>
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
                        <td>0.01</td>
                        <td>1023</td>
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
                        <td>0.01</td>
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
                        <td>0</td>
                        <td>0.01</td>
                        <td>511</td>
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
                        <td>0.01</td>
                        <td>255</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>




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
        <p>Sampling for 2 minutes</p>
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
                        <td>0.02</td>
                        <td>1023</td>
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
                        <td>0.01</td>
                        <td>767</td>
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
                        <td>0.02</td>
                        <td>767</td>
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
                        <td>0.02</td>
                        <td>1023</td>
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
        subject_ids        (observations) object 2kB &#x27;Munchen_sub96752&#x27; ... &#x27;Quee...
        Y                  (observations, response_vars) float64 7kB 2.721e+03 .....
        X                  (observations, covariates) float64 2kB 63.0 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 29kB &#x27;F&#x27; ... &#x27;Q...
        Z                  (observations, response_vars) float64 7kB 0.5176 ... -...
        centiles           (centile, observations, response_vars) float64 35kB 1....
        logp               (observations, response_vars) float64 7kB -1.619 ... -...
        Yhat               (observations, response_vars) float64 7kB 2.368e+03 .....
        statistics         (response_vars, statistic) float64 352B 0.3923 ... 0.995
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [np.str_(&#x27;F&#x27;), np.str_(&#x27;...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.2...
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): {np.str_(...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-e22dce66-eaab-4895-a189-3d8b5212a10e' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-e22dce66-eaab-4895-a189-3d8b5212a10e' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 4</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>centile</span>: 5</li><li><span class='xr-has-index'>statistic</span>: 11</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-5529905b-5a56-4074-8c5a-baaa2bc9900a' class='xr-section-summary-in' type='checkbox'  checked><label for='section-5529905b-5a56-4074-8c5a-baaa2bc9900a' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-cb2a81d1-b723-4c78-970f-2795d5fa2b58' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cb2a81d1-b723-4c78-970f-2795d5fa2b58' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-704ae540-000f-445a-8f7a-b1397f4128a8' class='xr-var-data-in' type='checkbox'><label for='data-704ae540-000f-445a-8f7a-b1397f4128a8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043], shape=(216,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Cortex...</div><input id='attrs-8389d1f4-d37c-4efd-90c0-a023260458d8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8389d1f4-d37c-4efd-90c0-a023260458d8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6f2baed6-0c13-44a2-a90e-2686142194d0' class='xr-var-data-in' type='checkbox'><label for='data-6f2baed6-0c13-44a2-a90e-2686142194d0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-e663c63a-62be-4242-ade8-ce9055df2805' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e663c63a-62be-4242-ade8-ce9055df2805' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bfc873a9-9fbb-46dd-a09c-36f29d4bdbfb' class='xr-var-data-in' type='checkbox'><label for='data-bfc873a9-9fbb-46dd-a09c-36f29d4bdbfb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-eb4dcbc0-70e1-4339-982c-08495f84428d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-eb4dcbc0-70e1-4339-982c-08495f84428d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-80554912-7173-4579-ab14-b00de6c64ae5' class='xr-var-data-in' type='checkbox'><label for='data-80554912-7173-4579-ab14-b00de6c64ae5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-252a7c73-d3b4-4928-af49-c955d1074cd9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-252a7c73-d3b4-4928-af49-c955d1074cd9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-12be1e0f-a00e-422e-8d00-239a50b93423' class='xr-var-data-in' type='checkbox'><label for='data-12be1e0f-a00e-422e-8d00-239a50b93423' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-c2db2890-423b-4a9a-9622-d18d7a792084' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c2db2890-423b-4a9a-9622-d18d7a792084' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7c98eb20-0d24-434a-9c36-dbf65eaa0075' class='xr-var-data-in' type='checkbox'><label for='data-7c98eb20-0d24-434a-9c36-dbf65eaa0075' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-c6d204dc-a89f-4b35-aa6f-9869494ecd98' class='xr-section-summary-in' type='checkbox'  checked><label for='section-c6d204dc-a89f-4b35-aa6f-9869494ecd98' class='xr-section-summary' >Data variables: <span>(9)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Munchen_sub96752&#x27; ... &#x27;Queensla...</div><input id='attrs-e3138e86-a010-4dc5-ae00-a7716caaa5b2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e3138e86-a010-4dc5-ae00-a7716caaa5b2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5209e2ff-8483-43ac-a487-f48822dcc249' class='xr-var-data-in' type='checkbox'><label for='data-5209e2ff-8483-43ac-a487-f48822dcc249' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Munchen_sub96752&#x27;, &#x27;NewYork_a_sub18638&#x27;, &#x27;Leiden_2200_sub87320&#x27;,
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
           &#x27;Cambridge_Buckner_sub59729&#x27;, &#x27;Queensland_sub86245&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.721e+03 1.289e+04 ... 5.035e+05</div><input id='attrs-1d40a290-bef6-4091-8906-d903a346ac59' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1d40a290-bef6-4091-8906-d903a346ac59' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-09b71366-5132-4b31-a7c5-85d3ad997432' class='xr-var-data-in' type='checkbox'><label for='data-09b71366-5132-4b31-a7c5-85d3ad997432' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[2.72140000e+03, 1.28916000e+04, 1.43940000e+03, 4.57858328e+05],
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
           [7.03500000e+02, 1.07003000e+04, 1.67620000e+03, 5.03535771e+05]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 23.27 22.0 ... 72.0 23.0 23.0</div><input id='attrs-d3c24659-268d-481e-8918-fd9490e544a1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d3c24659-268d-481e-8918-fd9490e544a1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-be5ca533-84a6-4b3d-a023-55f73566457d' class='xr-var-data-in' type='checkbox'><label for='data-be5ca533-84a6-4b3d-a023-55f73566457d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[63.  ],
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
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-2c493298-f427-4b73-b7b6-a3f1bd87420f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2c493298-f427-4b73-b7b6-a3f1bd87420f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-eb74001b-466f-4c6a-a3e3-fb5631d2b6d9' class='xr-var-data-in' type='checkbox'><label for='data-eb74001b-466f-4c6a-a3e3-fb5631d2b6d9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
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
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.5176 0.3966 ... -1.322 -1.132</div><input id='attrs-4185deb8-3fd4-40e8-b08a-2f8264171256' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4185deb8-3fd4-40e8-b08a-2f8264171256' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e76e8e1b-0ca3-4f69-a6f6-ccf02b9cfffb' class='xr-var-data-in' type='checkbox'><label for='data-e76e8e1b-0ca3-4f69-a6f6-ccf02b9cfffb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 5.17634102e-01,  3.96587902e-01, -2.74938367e-01,
             4.04224021e-01],
           [ 4.22983875e-01,  9.67919541e-01, -1.23234405e+00,
            -2.77131001e-01],
           [ 3.17634867e-01,  4.13453691e-01,  5.43831580e-01,
             2.83939528e-01],
           [ 5.69764720e-01,  1.36559455e+00, -2.58224243e-01,
             1.48353325e+00],
           [-2.02129515e+00, -1.70631918e+00, -2.81720327e-01,
            -1.89865119e+00],
           [-5.72819665e-01, -5.15533160e-01,  3.04304821e-01,
            -5.21722247e-01],
           [ 6.75130765e-01,  2.70201933e+00,  4.07280405e-01,
             3.37268109e-01],
           [-2.87499845e-02, -7.22990177e-01,  1.68718882e+00,
            -6.80139620e-01],
           [-1.09762722e+00,  6.34309335e-01, -9.32255550e-01,
            -6.58324834e-01],
           [-1.65770102e+00, -3.08356617e-01, -6.86101135e-01,
             2.04647968e-01],
    ...
           [ 3.32298675e-02,  4.58606618e-01,  3.92723301e-01,
            -4.70311460e-01],
           [-2.27429224e+00,  1.27684623e-01, -3.73443938e-01,
            -5.86993154e-01],
           [-3.67510898e-01,  2.78751625e-01,  8.27569363e-01,
             1.43202422e-01],
           [-5.48769765e-01, -4.00243565e-02, -1.88311453e-01,
            -7.81537654e-01],
           [-2.01876019e-01,  9.58187257e-01, -2.97913389e-01,
            -6.95484404e-01],
           [-2.75826300e+00, -4.66935532e-01,  3.45641866e-01,
             1.29630759e+00],
           [-1.65570328e+00,  4.00137072e-01, -9.25680950e-01,
            -4.78465796e-01],
           [-3.05791773e-01,  5.46107441e-01,  9.85380732e-01,
             2.86246030e-01],
           [ 2.45820979e+00,  3.35304544e-03,  1.98769592e+00,
             3.01652827e+00],
           [-9.60598976e-01,  1.16834804e+00, -1.32241264e+00,
            -1.13164727e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.01e+03 4.953e+03 ... 6.056e+05</div><input id='attrs-17571e21-46cb-4a70-9aca-4765fee3673d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-17571e21-46cb-4a70-9aca-4765fee3673d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3637f49c-7dcc-4653-97ca-19e00ab330b3' class='xr-var-data-in' type='checkbox'><label for='data-3637f49c-7dcc-4653-97ca-19e00ab330b3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[1.00990116e+03, 4.95261546e+03, 1.23966108e+03, 3.86732134e+05],
            [6.08012367e+02, 3.80876868e+03, 1.58705750e+03, 4.79670834e+05],
            [4.61196194e+02, 3.70985506e+03, 1.44074817e+03, 4.27885648e+05],
            ...,
            [1.09531427e+03, 5.75299910e+03, 1.29907742e+03, 4.16415097e+05],
            [9.34686309e+02, 3.51536166e+03, 1.60892383e+03, 4.41885188e+05],
            [5.78885309e+02, 3.75354553e+03, 1.62634202e+03, 4.86392330e+05]],
    
           [[1.53467407e+03, 7.37884986e+03, 1.37288154e+03, 4.18892379e+05],
            [7.85491258e+02, 4.97781713e+03, 1.74087762e+03, 5.12445871e+05],
            [6.40548301e+02, 4.83759931e+03, 1.59357876e+03, 4.60568070e+05],
            ...,
            [1.91385020e+03, 8.83549635e+03, 1.45303171e+03, 4.50273424e+05],
            [1.11246563e+03, 4.67558192e+03, 1.76250027e+03, 4.74630203e+05],
            [7.56664633e+02, 4.91376579e+03, 1.77991846e+03, 5.19137344e+05]],
    
           [[2.08162657e+03, 1.03044095e+04, 1.48434170e+03, 4.42959028e+05],
            [9.70551734e+02, 6.38751268e+03, 1.86959901e+03, 5.36982063e+05],
            [8.27567125e+02, 6.19740746e+03, 1.72147221e+03, 4.85034019e+05],
            ...,
            [2.76692895e+03, 1.25522745e+04, 1.58180991e+03, 4.75617565e+05],
            [1.29784046e+03, 6.07461523e+03, 1.89101778e+03, 4.99143722e+05],
            [9.42039462e+02, 6.31279910e+03, 1.90843597e+03, 5.43650864e+05]],
    
           [[2.95399830e+03, 1.50944831e+04, 1.61138235e+03, 4.68071391e+05],
            [1.26591640e+03, 8.69521396e+03, 2.01627348e+03, 5.62588738e+05],
            [1.12605770e+03, 8.42343693e+03, 1.86720237e+03, 5.10566766e+05],
            ...,
            [4.12754831e+03, 1.86373107e+04, 1.72854720e+03, 5.02067343e+05],
            [1.59370703e+03, 8.36486125e+03, 2.03745975e+03, 5.24726609e+05],
            [1.23790603e+03, 8.60304513e+03, 2.05487793e+03, 5.69233751e+05]],
    
           [[4.65616830e+03, 2.41035672e+04, 1.79735487e+03, 5.03817165e+05],
            [1.84256259e+03, 1.30344587e+04, 2.23086783e+03, 5.99033916e+05],
            [1.70879944e+03, 1.26092098e+04, 2.08041314e+03, 5.46906720e+05],
            ...,
            [6.78243642e+03, 3.00809143e+04, 1.94330550e+03, 5.39716496e+05],
            [2.17133183e+03, 1.26713049e+04, 2.25171355e+03, 5.61137946e+05],
            [1.81553083e+03, 1.29094888e+04, 2.26913174e+03, 6.05645088e+05]]],
          shape=(5, 216, 4))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.619 -1.649 ... -1.388 -1.171</div><input id='attrs-325e8d73-e87f-438d-a265-d4e3505950d6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-325e8d73-e87f-438d-a265-d4e3505950d6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-75fd7bd5-3705-4330-a547-3dab69fcff68' class='xr-var-data-in' type='checkbox'><label for='data-75fd7bd5-3705-4330-a547-3dab69fcff68' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.61900441e+00, -1.64872515e+00, -6.30593473e-01,
            -7.04591433e-01],
           [-4.39648922e-01, -1.51000173e+00, -1.24647849e+00,
            -6.17473598e-01],
           [-3.54191945e-01, -9.01238608e-01, -9.81020925e-01,
            -6.67393336e-01],
           [-9.91306776e-01, -2.35556208e+00, -7.69674346e-01,
            -1.77665602e+00],
           [-2.46215381e+00, -1.79897865e+00, -6.16653822e-01,
            -2.27941462e+00],
           [ 1.69124902e-01, -2.74686908e-01, -8.49530042e-01,
            -6.81679379e-01],
           [-7.20401429e-01, -4.79102243e+00, -9.04112279e-01,
            -6.86561630e-01],
           [-5.91698377e-02, -3.85568704e-01, -2.26021585e+00,
            -8.00991213e-01],
           [-5.63044665e-02, -1.08192395e+00, -9.51801357e-01,
            -7.45329757e-01],
           [-1.35906966e+00, -7.90249723e-01, -8.06928148e-01,
            -6.92904662e-01],
    ...
           [-1.46252766e-01, -8.86535169e-01, -8.92440670e-01,
            -6.86144371e-01],
           [-1.89431853e+00, -6.47296691e-01, -7.27680557e-01,
            -7.10758489e-01],
           [ 4.51881704e-02, -6.96746888e-01, -1.17981874e+00,
            -6.41081842e-01],
           [ 1.37880490e-01, -5.76469383e-01, -7.50651853e-01,
            -8.57372643e-01],
           [ 3.74132085e-02, -1.43319364e+00, -7.19239456e-01,
            -7.68537686e-01],
           [-3.09213725e+00, -3.15429520e-01, -8.73552498e-01,
            -1.43545490e+00],
           [-7.29581572e-01, -9.09495414e-01, -9.70284964e-01,
            -6.73514804e-01],
           [-1.44072104e+00, -2.02548948e+00, -1.34028475e+00,
            -7.05321045e-01],
           [-3.80088630e+00, -5.73967255e-01, -2.76215100e+00,
            -5.09948581e+00],
           [-3.49057002e-02, -1.75748713e+00, -1.38757411e+00,
            -1.17055775e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.368e+03 1.192e+04 ... 5.446e+05</div><input id='attrs-8f03ca0a-4a0b-4c6e-91c4-5f300c45a57d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8f03ca0a-4a0b-4c6e-91c4-5f300c45a57d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e9295c30-5a08-4b87-a10a-f454814d7b5e' class='xr-var-data-in' type='checkbox'><label for='data-e9295c30-5a08-4b87-a10a-f454814d7b5e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[  2368.16852993,  11920.42199819,   1497.49199916,
            443848.66189766],
           [  1067.69017098,   7165.78526919,   1884.75013303,
            537892.7199698 ],
           [   925.73331757,   6948.13977498,   1736.52510559,
            485941.54478701],
           [  1258.20991853,   8147.94193486,   1936.59550757,
            525906.9808095 ],
           [  2281.60723106,  11161.01381759,   1384.68615636,
            397564.45515247],
           [  1130.17290848,   6934.55795343,   1857.70406267,
            530776.41100181],
           [  1003.26575659,   7481.32940001,   1950.07293649,
            538695.27604126],
           [   698.82484341,   7385.26656618,   1650.05138951,
            493520.03846742],
           [  1056.31968375,   6006.30209218,   1641.87584199,
            484921.31591297],
           [  1312.36974838,   8024.64554622,   1615.80832554,
            437326.05707219],
    ...
           [  1418.81036493,   6332.66194636,   1679.94431259,
            497090.09062899],
           [  1273.21391297,   6075.59709641,   1622.88316635,
            496320.26171228],
           [  1350.8073276 ,   5516.19942497,   1461.81101712,
            449612.92447981],
           [   687.57508819,   7197.17166813,   1649.15884169,
            497195.56341665],
           [  1131.81350381,   6674.32518314,   1856.41139575,
            535306.83061507],
           [  1057.32079516,   6380.74588573,   1643.20549463,
            478392.00505681],
           [  1066.91382073,   7108.30803412,   1884.70311302,
            538885.82077162],
           [  3213.83820622,  14604.85115656,   1596.96757522,
            476558.15227448],
           [  1395.14397879,   6847.00183532,   1906.14475143,
            500053.4310279 ],
           [  1039.34297845,   7085.18570843,   1923.56293795,
            544560.57271972]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.3923 0.01852 ... 0.5869 0.995</div><input id='attrs-cee99222-852d-409f-a740-f4944d54aa34' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cee99222-852d-409f-a740-f4944d54aa34' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7362255d-9185-469f-b048-cc4119a33036' class='xr-var-data-in' type='checkbox'><label for='data-7362255d-9185-469f-b048-cc4119a33036' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 3.92305238e-01,  1.85185185e-02,  3.47942203e-01,
            -7.07599869e+00,  7.48551454e-01,  3.83409084e-01,
             4.75248044e+02,  5.12245665e-01,  7.55753167e-16,
             6.16590916e-01,  9.80829221e-01],
           [ 2.26392610e-01,  1.51851852e-02,  4.31014923e-01,
            -8.52987029e+00,  1.16063545e+00,  2.26294263e-01,
             3.44020165e+03,  2.62176804e-01,  9.64849176e-05,
             7.73705737e-01,  9.85110903e-01],
           [ 3.02254222e-01,  1.70370370e-02,  9.47297281e-02,
            -5.66336076e+00,  1.21916528e+00,  3.00410824e-01,
             1.97345598e+02,  5.25818563e-01,  9.35737195e-17,
             6.99589176e-01,  9.91809737e-01],
           [ 4.22027877e-01,  2.77777778e-02,  6.34029215e-02,
            -1.11124552e+01,  1.10510494e+00,  4.13149804e-01,
             3.75012201e+04,  6.40949145e-01,  2.20511329e-26,
             5.86850196e-01,  9.95016851e-01]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-170bfc56-80b3-448e-86aa-e2b861d14714' class='xr-section-summary-in' type='checkbox'  ><label for='section-170bfc56-80b3-448e-86aa-e2b861d14714' class='xr-section-summary' >Indexes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-dbcb50a6-3db6-4265-ab7b-19e5a04c7adc' class='xr-index-data-in' type='checkbox'/><label for='index-dbcb50a6-3db6-4265-ab7b-19e5a04c7adc' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([ 756,  769,  692,  616,   35,  164,  680,  331,  299,  727,
           ...
             27,  959,   29,  346,  304,  264,  798,  751,  470, 1043],
          dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;, length=216))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-dec77a40-9163-40bc-800c-d1cf8c0d5c30' class='xr-index-data-in' type='checkbox'/><label for='index-dec77a40-9163-40bc-800c-d1cf8c0d5c30' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-4c1b2d43-da24-4609-b420-aab3057ca650' class='xr-index-data-in' type='checkbox'/><label for='index-4c1b2d43-da24-4609-b420-aab3057ca650' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-d8e127ba-1503-441c-bed9-b0d15a8f85a3' class='xr-index-data-in' type='checkbox'/><label for='index-d8e127ba-1503-441c-bed9-b0d15a8f85a3' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>centile</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-5a8e5545-2ae0-4cab-8174-7cdd9466f589' class='xr-index-data-in' type='checkbox'/><label for='index-5a8e5545-2ae0-4cab-8174-7cdd9466f589' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0.05, 0.25, 0.5, 0.75, 0.95], dtype=&#x27;float64&#x27;, name=&#x27;centile&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-eb96c359-efb4-481d-869d-850138b377f0' class='xr-index-data-in' type='checkbox'/><label for='index-eb96c359-efb4-481d-869d-850138b377f0' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ea5daabb-4f09-4968-9bef-eaf7cf81b9c5' class='xr-section-summary-in' type='checkbox'  checked><label for='section-ea5daabb-4f09-4968-9bef-eaf7cf81b9c5' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [np.str_(&#x27;F&#x27;), np.str_(&#x27;M&#x27;)], np.str_(&#x27;site&#x27;): [np.str_(&#x27;AnnArbor_a&#x27;), np.str_(&#x27;AnnArbor_b&#x27;), np.str_(&#x27;Atlanta&#x27;), np.str_(&#x27;Baltimore&#x27;), np.str_(&#x27;Bangor&#x27;), np.str_(&#x27;Beijing_Zang&#x27;), np.str_(&#x27;Berlin_Margulies&#x27;), np.str_(&#x27;Cambridge_Buckner&#x27;), np.str_(&#x27;Cleveland&#x27;), np.str_(&#x27;ICBM&#x27;), np.str_(&#x27;Leiden_2180&#x27;), np.str_(&#x27;Leiden_2200&#x27;), np.str_(&#x27;Milwaukee_b&#x27;), np.str_(&#x27;Munchen&#x27;), np.str_(&#x27;NewYork_a&#x27;), np.str_(&#x27;NewYork_a_ADHD&#x27;), np.str_(&#x27;Newark&#x27;), np.str_(&#x27;Oulu&#x27;), np.str_(&#x27;Oxford&#x27;), np.str_(&#x27;PaloAlto&#x27;), np.str_(&#x27;Pittsburgh&#x27;), np.str_(&#x27;Queensland&#x27;), np.str_(&#x27;SaintLouis&#x27;)]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x168093a60&gt;, {np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): 589, np.str_(&#x27;M&#x27;): 489}, np.str_(&#x27;site&#x27;): {np.str_(&#x27;AnnArbor_a&#x27;): 24, np.str_(&#x27;AnnArbor_b&#x27;): 32, np.str_(&#x27;Atlanta&#x27;): 28, np.str_(&#x27;Baltimore&#x27;): 23, np.str_(&#x27;Bangor&#x27;): 20, np.str_(&#x27;Beijing_Zang&#x27;): 198, np.str_(&#x27;Berlin_Margulies&#x27;): 26, np.str_(&#x27;Cambridge_Buckner&#x27;): 198, np.str_(&#x27;Cleveland&#x27;): 31, np.str_(&#x27;ICBM&#x27;): 85, np.str_(&#x27;Leiden_2180&#x27;): 12, np.str_(&#x27;Leiden_2200&#x27;): 19, np.str_(&#x27;Milwaukee_b&#x27;): 46, np.str_(&#x27;Munchen&#x27;): 15, np.str_(&#x27;NewYork_a&#x27;): 83, np.str_(&#x27;NewYork_a_ADHD&#x27;): 25, np.str_(&#x27;Newark&#x27;): 19, np.str_(&#x27;Oulu&#x27;): 102, np.str_(&#x27;Oxford&#x27;): 22, np.str_(&#x27;PaloAlto&#x27;): 17, np.str_(&#x27;Pittsburgh&#x27;): 3, np.str_(&#x27;Queensland&#x27;): 19, np.str_(&#x27;SaintLouis&#x27;): 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.251224489795916), &#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(85.0)}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.05332767402377), &#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(85.0)}}, np.str_(&#x27;M&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.48959100204499), &#x27;min&#x27;: np.float64(9.21), &#x27;max&#x27;: np.float64(78.0)}}}, np.str_(&#x27;site&#x27;): {np.str_(&#x27;AnnArbor_a&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.28333333333333), &#x27;min&#x27;: np.float64(13.41), &#x27;max&#x27;: np.float64(40.98)}}, np.str_(&#x27;AnnArbor_b&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(44.40625), &#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(79.0)}}, np.str_(&#x27;Atlanta&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(30.892857142857142), &#x27;min&#x27;: np.float64(22.0), &#x27;max&#x27;: np.float64(57.0)}}, np.str_(&#x27;Baltimore&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(29.26086956521739), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(40.0)}}, np.str_(&#x27;Bangor&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(23.4), &#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(38.0)}}, np.str_(&#x27;Beijing_Zang&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.161616161616163), &#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(26.0)}}, np.str_(&#x27;Berlin_Margulies&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(29.76923076923077), &#x27;min&#x27;: np.float64(23.0), &#x27;max&#x27;: np.float64(44.0)}}, np.str_(&#x27;Cambridge_Buckner&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.03030303030303), &#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(30.0)}}, np.str_(&#x27;Cleveland&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(43.54838709677419), &#x27;min&#x27;: np.float64(24.0), &#x27;max&#x27;: np.float64(60.0)}}, np.str_(&#x27;ICBM&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(44.04705882352941), &#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(85.0)}}, np.str_(&#x27;Leiden_2180&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(23.0), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(27.0)}}, np.str_(&#x27;Leiden_2200&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.68421052631579), &#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(28.0)}}, np.str_(&#x27;Milwaukee_b&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(53.58695652173913), &#x27;min&#x27;: np.float64(44.0), &#x27;max&#x27;: np.float64(65.0)}}, np.str_(&#x27;Munchen&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(68.13333333333334), &#x27;min&#x27;: np.float64(63.0), &#x27;max&#x27;: np.float64(74.0)}}, np.str_(&#x27;NewYork_a&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(24.507710843373495), &#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(49.16)}}, np.str_(&#x27;NewYork_a_ADHD&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(34.9952), &#x27;min&#x27;: np.float64(20.69), &#x27;max&#x27;: np.float64(50.9)}}, np.str_(&#x27;Newark&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(24.105263157894736), &#x27;min&#x27;: np.float64(21.0), &#x27;max&#x27;: np.float64(39.0)}}, np.str_(&#x27;Oulu&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.519607843137255), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(23.0)}}, np.str_(&#x27;Oxford&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(29.0), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(35.0)}}, np.str_(&#x27;PaloAlto&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(32.470588235294116), &#x27;min&#x27;: np.float64(22.0), &#x27;max&#x27;: np.float64(46.0)}}, np.str_(&#x27;Pittsburgh&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(32.333333333333336), &#x27;min&#x27;: np.float64(25.0), &#x27;max&#x27;: np.float64(47.0)}}, np.str_(&#x27;Queensland&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(25.94736842105263), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(34.0)}}, np.str_(&#x27;SaintLouis&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(25.096774193548388), &#x27;min&#x27;: np.float64(21.0), &#x27;max&#x27;: np.float64(29.0)}}}}</dd></dl></div></li></ul></div></div>



Plot the results
----------------

The PCNtoolkit offers are a number of different plotting functions: 1.
plot_centiles: Plot the predicted centiles for a model on top of
harmonized scatter data. 2. plot_centiles_advanced: a more advanced
version of plot_centiles, with more configuration options, coloring, and
conditionals 3. plot_qq: Plot the QQ-plot of the predicted Z-scores 4.
plot_ridge: Plot density plots of the predicted Z-scores

Let’s start with the centiles.

.. code:: ipython3

    from pcntoolkit.util.plotter import plot_centiles_advanced
    plot_centiles_advanced(
        model,
        centiles=[0.05, 0.5, 0.95],  # Plot these centiles, the default is [0.05, 0.25, 0.5, 0.75, 0.95]
        scatter_data=train,  # Scatter this data along with the centiles
        batch_effects={"site": ["Beijing_Zang", "AnnArbor_a"], "sex": ["M"]},  # Highlight these groups
        show_other_data=True,  # scatter data not in those groups as smaller black circles
        harmonize=True,  # harmonize the scatterdata, this means that we 'remove' the batch effects from the data, by simulating what the data would have looked like if all data was from the same batch.
        conditionals=[30]
    )



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_16_0.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_16_1.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_16_2.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_16_3.png


Now let’s see the qq plots

.. code:: ipython3

    plot_qq(test, plot_id_line=True)



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_18_0.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_18_1.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_18_2.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_18_3.png


We can also split the QQ plots by batch effects:

.. code:: ipython3

    plot_qq(test, plot_id_line=True, hue_data="sex", split_data="sex")
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": (0, 0, 0, 0)})



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_20_0.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_20_1.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_20_2.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_20_3.png


And finally the ridge plot:

.. code:: ipython3

    plot_ridge(
        train, "Z", split_by="sex"
    )  # We can also show the 'Y' variable, and that will show the marginal distribution of the response variable, per batch effect.


.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:816: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      plt.tight_layout()



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_22_1.png


.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:816: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      plt.tight_layout()



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_22_3.png


.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:816: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      plt.tight_layout()



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_22_5.png


.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:816: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      plt.tight_layout()



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_22_7.png


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
          <th>EXPV</th>
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
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>CortexVol</th>
          <td>0.519177</td>
          <td>0.010719</td>
          <td>0.057411</td>
          <td>-11.208105</td>
          <td>1.072468</td>
          <td>0.519161</td>
          <td>36153.260848</td>
          <td>0.710251</td>
          <td>2.937339e-133</td>
          <td>0.480839</td>
          <td>0.995200</td>
        </tr>
        <tr>
          <th>Right-Amygdala</th>
          <td>0.383262</td>
          <td>0.013921</td>
          <td>0.088636</td>
          <td>-5.738210</td>
          <td>1.181985</td>
          <td>0.383123</td>
          <td>192.426517</td>
          <td>0.599847</td>
          <td>2.324945e-85</td>
          <td>0.616877</td>
          <td>0.995671</td>
        </tr>
        <tr>
          <th>Right-Lateral-Ventricle</th>
          <td>0.214934</td>
          <td>0.013968</td>
          <td>0.399611</td>
          <td>-8.643894</td>
          <td>1.030930</td>
          <td>0.213145</td>
          <td>3415.329234</td>
          <td>0.371130</td>
          <td>1.538075e-29</td>
          <td>0.786855</td>
          <td>0.987616</td>
        </tr>
        <tr>
          <th>WM-hypointensities</th>
          <td>0.350893</td>
          <td>0.014664</td>
          <td>0.321002</td>
          <td>-7.437521</td>
          <td>0.686698</td>
          <td>0.350110</td>
          <td>658.394871</td>
          <td>0.487865</td>
          <td>9.602440e-53</td>
          <td>0.649890</td>
          <td>0.986295</td>
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
          <th>EXPV</th>
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
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>CortexVol</th>
          <td>0.422028</td>
          <td>0.027778</td>
          <td>0.063403</td>
          <td>-11.112455</td>
          <td>1.105105</td>
          <td>0.413150</td>
          <td>37501.220099</td>
          <td>0.640949</td>
          <td>2.205113e-26</td>
          <td>0.586850</td>
          <td>0.995017</td>
        </tr>
        <tr>
          <th>Right-Amygdala</th>
          <td>0.302254</td>
          <td>0.017037</td>
          <td>0.094730</td>
          <td>-5.663361</td>
          <td>1.219165</td>
          <td>0.300411</td>
          <td>197.345598</td>
          <td>0.525819</td>
          <td>9.357372e-17</td>
          <td>0.699589</td>
          <td>0.991810</td>
        </tr>
        <tr>
          <th>Right-Lateral-Ventricle</th>
          <td>0.226393</td>
          <td>0.015185</td>
          <td>0.431015</td>
          <td>-8.529870</td>
          <td>1.160635</td>
          <td>0.226294</td>
          <td>3440.201648</td>
          <td>0.262177</td>
          <td>9.648492e-05</td>
          <td>0.773706</td>
          <td>0.985111</td>
        </tr>
        <tr>
          <th>WM-hypointensities</th>
          <td>0.392305</td>
          <td>0.018519</td>
          <td>0.347942</td>
          <td>-7.075999</td>
          <td>0.748551</td>
          <td>0.383409</td>
          <td>475.248044</td>
          <td>0.512246</td>
          <td>7.557532e-16</td>
          <td>0.616591</td>
          <td>0.980829</td>
        </tr>
      </tbody>
    </table>
    </div>


What’s next?
------------

Now we have a normative hierarchical Bayesian regression model, we can
use it to:

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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 98kB
    Dimensions:            (observations: 216, response_vars: 4, covariates: 1,
                            batch_effect_dims: 2, statistic: 11, centile: 5)
    Coordinates:
      * observations       (observations) int64 2kB 756 769 692 616 ... 751 470 1043
      * response_vars      (response_vars) &lt;U23 368B &#x27;WM-hypointensities&#x27; ... &#x27;Co...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * statistic          (statistic) &lt;U8 352B &#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
    Data variables:
        subject_ids        (observations) object 2kB &#x27;Munchen_sub96752&#x27; ... &#x27;Quee...
        Y                  (observations, response_vars) float64 7kB 2.721e+03 .....
        X                  (observations, covariates) float64 2kB 63.0 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 29kB &#x27;F&#x27; ... &#x27;Q...
        Z                  (observations, response_vars) float64 7kB 0.5176 ... -...
        logp               (observations, response_vars) float64 7kB -1.619 ... -...
        Yhat               (observations, response_vars) float64 7kB 2.368e+03 .....
        statistics         (response_vars, statistic) float64 352B 0.3923 ... 0.995
        centiles           (centile, observations, response_vars) float64 35kB 1....
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [np.str_(&#x27;F&#x27;), np.str_(&#x27;...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.2...
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): {np.str_(...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-dd03de56-8e83-44e7-b463-bf5129c376d5' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-dd03de56-8e83-44e7-b463-bf5129c376d5' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 4</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>statistic</span>: 11</li><li><span class='xr-has-index'>centile</span>: 5</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-7d6295b8-4380-4fa4-a1a8-fe07fa39df7f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-7d6295b8-4380-4fa4-a1a8-fe07fa39df7f' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-13d20434-4678-40bf-9852-aa4a3448fc75' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-13d20434-4678-40bf-9852-aa4a3448fc75' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-14be1cfd-2519-4748-af67-08f07c8c7599' class='xr-var-data-in' type='checkbox'><label for='data-14be1cfd-2519-4748-af67-08f07c8c7599' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043], shape=(216,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Cortex...</div><input id='attrs-f58592e2-6a0c-4a1a-9413-68af4bdcf3a2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f58592e2-6a0c-4a1a-9413-68af4bdcf3a2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5d323f6b-82c7-442b-9491-031ac4f8e66b' class='xr-var-data-in' type='checkbox'><label for='data-5d323f6b-82c7-442b-9491-031ac4f8e66b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-e8459a73-ed09-4e00-bb10-a0a7d06426d4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e8459a73-ed09-4e00-bb10-a0a7d06426d4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-568a3f26-d98a-4d24-b936-1a735aa94640' class='xr-var-data-in' type='checkbox'><label for='data-568a3f26-d98a-4d24-b936-1a735aa94640' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-4c724f74-557f-44b2-a997-3413c5abeb41' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4c724f74-557f-44b2-a997-3413c5abeb41' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c0fc4419-3b25-4b39-9004-0b49ee8d0ad0' class='xr-var-data-in' type='checkbox'><label for='data-c0fc4419-3b25-4b39-9004-0b49ee8d0ad0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-df18b531-562d-4182-b7ba-e269eaa83130' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-df18b531-562d-4182-b7ba-e269eaa83130' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9d8a45ba-dc0b-49dd-abf4-3f7fd3f23a67' class='xr-var-data-in' type='checkbox'><label for='data-9d8a45ba-dc0b-49dd-abf4-3f7fd3f23a67' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-2de11b4a-fba9-4a4b-b0f2-42f71ae7984e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2de11b4a-fba9-4a4b-b0f2-42f71ae7984e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-101c8eca-8ac4-4a17-a0d9-8814357e0edd' class='xr-var-data-in' type='checkbox'><label for='data-101c8eca-8ac4-4a17-a0d9-8814357e0edd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d527cb2d-61c4-46a9-a47c-a918d3c933df' class='xr-section-summary-in' type='checkbox'  checked><label for='section-d527cb2d-61c4-46a9-a47c-a918d3c933df' class='xr-section-summary' >Data variables: <span>(9)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Munchen_sub96752&#x27; ... &#x27;Queensla...</div><input id='attrs-c00dd269-38a7-4935-9552-e878bddb6917' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c00dd269-38a7-4935-9552-e878bddb6917' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8aa316c5-86b2-4892-845f-c7a7ede9db0c' class='xr-var-data-in' type='checkbox'><label for='data-8aa316c5-86b2-4892-845f-c7a7ede9db0c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Munchen_sub96752&#x27;, &#x27;NewYork_a_sub18638&#x27;, &#x27;Leiden_2200_sub87320&#x27;,
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
           &#x27;Cambridge_Buckner_sub59729&#x27;, &#x27;Queensland_sub86245&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.721e+03 1.289e+04 ... 5.035e+05</div><input id='attrs-d0ce57a7-d00a-4477-8b09-2dd302dff120' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d0ce57a7-d00a-4477-8b09-2dd302dff120' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5fda0ea7-c8d1-4e0a-9f9f-4fc969f8d4f8' class='xr-var-data-in' type='checkbox'><label for='data-5fda0ea7-c8d1-4e0a-9f9f-4fc969f8d4f8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[2.72140000e+03, 1.28916000e+04, 1.43940000e+03, 4.57858328e+05],
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
           [7.03500000e+02, 1.07003000e+04, 1.67620000e+03, 5.03535771e+05]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 23.27 22.0 ... 72.0 23.0 23.0</div><input id='attrs-ff8050f9-e804-4159-bf51-640f260a7a11' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ff8050f9-e804-4159-bf51-640f260a7a11' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-91dc4b67-126a-4499-868c-c1bf6e9f51ab' class='xr-var-data-in' type='checkbox'><label for='data-91dc4b67-126a-4499-868c-c1bf6e9f51ab' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[63.  ],
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
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-1ce587ba-1125-40c2-963c-83666f967696' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1ce587ba-1125-40c2-963c-83666f967696' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-50835b02-e859-4a69-b846-7087ab057268' class='xr-var-data-in' type='checkbox'><label for='data-50835b02-e859-4a69-b846-7087ab057268' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
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
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.5176 0.3966 ... -1.322 -1.132</div><input id='attrs-8b820cf1-0bfc-4be8-b393-4149212a7e29' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8b820cf1-0bfc-4be8-b393-4149212a7e29' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e121578b-a450-4d31-927d-f4debbc1aa45' class='xr-var-data-in' type='checkbox'><label for='data-e121578b-a450-4d31-927d-f4debbc1aa45' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 5.17634102e-01,  3.96587902e-01, -2.74938367e-01,
             4.04224021e-01],
           [ 4.22983875e-01,  9.67919541e-01, -1.23234405e+00,
            -2.77131001e-01],
           [ 3.17634867e-01,  4.13453691e-01,  5.43831580e-01,
             2.83939528e-01],
           [ 5.69764720e-01,  1.36559455e+00, -2.58224243e-01,
             1.48353325e+00],
           [-2.02129515e+00, -1.70631918e+00, -2.81720327e-01,
            -1.89865119e+00],
           [-5.72819665e-01, -5.15533160e-01,  3.04304821e-01,
            -5.21722247e-01],
           [ 6.75130765e-01,  2.70201933e+00,  4.07280405e-01,
             3.37268109e-01],
           [-2.87499845e-02, -7.22990177e-01,  1.68718882e+00,
            -6.80139620e-01],
           [-1.09762722e+00,  6.34309335e-01, -9.32255550e-01,
            -6.58324834e-01],
           [-1.65770102e+00, -3.08356617e-01, -6.86101135e-01,
             2.04647968e-01],
    ...
           [ 3.32298675e-02,  4.58606618e-01,  3.92723301e-01,
            -4.70311460e-01],
           [-2.27429224e+00,  1.27684623e-01, -3.73443938e-01,
            -5.86993154e-01],
           [-3.67510898e-01,  2.78751625e-01,  8.27569363e-01,
             1.43202422e-01],
           [-5.48769765e-01, -4.00243565e-02, -1.88311453e-01,
            -7.81537654e-01],
           [-2.01876019e-01,  9.58187257e-01, -2.97913389e-01,
            -6.95484404e-01],
           [-2.75826300e+00, -4.66935532e-01,  3.45641866e-01,
             1.29630759e+00],
           [-1.65570328e+00,  4.00137072e-01, -9.25680950e-01,
            -4.78465796e-01],
           [-3.05791773e-01,  5.46107441e-01,  9.85380732e-01,
             2.86246030e-01],
           [ 2.45820979e+00,  3.35304544e-03,  1.98769592e+00,
             3.01652827e+00],
           [-9.60598976e-01,  1.16834804e+00, -1.32241264e+00,
            -1.13164727e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.619 -1.649 ... -1.388 -1.171</div><input id='attrs-36b8b9cf-0616-4f93-82b0-146da9f863b8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-36b8b9cf-0616-4f93-82b0-146da9f863b8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-de7f330a-6a79-46cf-bbf6-8bc6e5047894' class='xr-var-data-in' type='checkbox'><label for='data-de7f330a-6a79-46cf-bbf6-8bc6e5047894' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.61900441e+00, -1.64872515e+00, -6.30593473e-01,
            -7.04591433e-01],
           [-4.39648922e-01, -1.51000173e+00, -1.24647849e+00,
            -6.17473598e-01],
           [-3.54191945e-01, -9.01238608e-01, -9.81020925e-01,
            -6.67393336e-01],
           [-9.91306776e-01, -2.35556208e+00, -7.69674346e-01,
            -1.77665602e+00],
           [-2.46215381e+00, -1.79897865e+00, -6.16653822e-01,
            -2.27941462e+00],
           [ 1.69124902e-01, -2.74686908e-01, -8.49530042e-01,
            -6.81679379e-01],
           [-7.20401429e-01, -4.79102243e+00, -9.04112279e-01,
            -6.86561630e-01],
           [-5.91698377e-02, -3.85568704e-01, -2.26021585e+00,
            -8.00991213e-01],
           [-5.63044665e-02, -1.08192395e+00, -9.51801357e-01,
            -7.45329757e-01],
           [-1.35906966e+00, -7.90249723e-01, -8.06928148e-01,
            -6.92904662e-01],
    ...
           [-1.46252766e-01, -8.86535169e-01, -8.92440670e-01,
            -6.86144371e-01],
           [-1.89431853e+00, -6.47296691e-01, -7.27680557e-01,
            -7.10758489e-01],
           [ 4.51881704e-02, -6.96746888e-01, -1.17981874e+00,
            -6.41081842e-01],
           [ 1.37880490e-01, -5.76469383e-01, -7.50651853e-01,
            -8.57372643e-01],
           [ 3.74132085e-02, -1.43319364e+00, -7.19239456e-01,
            -7.68537686e-01],
           [-3.09213725e+00, -3.15429520e-01, -8.73552498e-01,
            -1.43545490e+00],
           [-7.29581572e-01, -9.09495414e-01, -9.70284964e-01,
            -6.73514804e-01],
           [-1.44072104e+00, -2.02548948e+00, -1.34028475e+00,
            -7.05321045e-01],
           [-3.80088630e+00, -5.73967255e-01, -2.76215100e+00,
            -5.09948581e+00],
           [-3.49057002e-02, -1.75748713e+00, -1.38757411e+00,
            -1.17055775e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.368e+03 1.192e+04 ... 5.446e+05</div><input id='attrs-83e30776-c86e-461b-985a-ead036f9d282' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-83e30776-c86e-461b-985a-ead036f9d282' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-65668900-3a8e-4cce-8a5d-2f59b13b7d2d' class='xr-var-data-in' type='checkbox'><label for='data-65668900-3a8e-4cce-8a5d-2f59b13b7d2d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[  2368.16852993,  11920.42199819,   1497.49199916,
            443848.66189766],
           [  1067.69017098,   7165.78526919,   1884.75013303,
            537892.7199698 ],
           [   925.73331757,   6948.13977498,   1736.52510559,
            485941.54478701],
           [  1258.20991853,   8147.94193486,   1936.59550757,
            525906.9808095 ],
           [  2281.60723106,  11161.01381759,   1384.68615636,
            397564.45515247],
           [  1130.17290848,   6934.55795343,   1857.70406267,
            530776.41100181],
           [  1003.26575659,   7481.32940001,   1950.07293649,
            538695.27604126],
           [   698.82484341,   7385.26656618,   1650.05138951,
            493520.03846742],
           [  1056.31968375,   6006.30209218,   1641.87584199,
            484921.31591297],
           [  1312.36974838,   8024.64554622,   1615.80832554,
            437326.05707219],
    ...
           [  1418.81036493,   6332.66194636,   1679.94431259,
            497090.09062899],
           [  1273.21391297,   6075.59709641,   1622.88316635,
            496320.26171228],
           [  1350.8073276 ,   5516.19942497,   1461.81101712,
            449612.92447981],
           [   687.57508819,   7197.17166813,   1649.15884169,
            497195.56341665],
           [  1131.81350381,   6674.32518314,   1856.41139575,
            535306.83061507],
           [  1057.32079516,   6380.74588573,   1643.20549463,
            478392.00505681],
           [  1066.91382073,   7108.30803412,   1884.70311302,
            538885.82077162],
           [  3213.83820622,  14604.85115656,   1596.96757522,
            476558.15227448],
           [  1395.14397879,   6847.00183532,   1906.14475143,
            500053.4310279 ],
           [  1039.34297845,   7085.18570843,   1923.56293795,
            544560.57271972]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.3923 0.01852 ... 0.5869 0.995</div><input id='attrs-7ad2c0f2-e3f6-4317-a9f4-7467fe48d558' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7ad2c0f2-e3f6-4317-a9f4-7467fe48d558' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-37fb7207-c3a4-4c26-8adc-6ef7705dab24' class='xr-var-data-in' type='checkbox'><label for='data-37fb7207-c3a4-4c26-8adc-6ef7705dab24' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 3.92305238e-01,  1.85185185e-02,  3.47942203e-01,
            -7.07599869e+00,  7.48551454e-01,  3.83409084e-01,
             4.75248044e+02,  5.12245665e-01,  7.55753167e-16,
             6.16590916e-01,  9.80829221e-01],
           [ 2.26392610e-01,  1.51851852e-02,  4.31014923e-01,
            -8.52987029e+00,  1.16063545e+00,  2.26294263e-01,
             3.44020165e+03,  2.62176804e-01,  9.64849176e-05,
             7.73705737e-01,  9.85110903e-01],
           [ 3.02254222e-01,  1.70370370e-02,  9.47297281e-02,
            -5.66336076e+00,  1.21916528e+00,  3.00410824e-01,
             1.97345598e+02,  5.25818563e-01,  9.35737195e-17,
             6.99589176e-01,  9.91809737e-01],
           [ 4.22027877e-01,  2.77777778e-02,  6.34029215e-02,
            -1.11124552e+01,  1.10510494e+00,  4.13149804e-01,
             3.75012201e+04,  6.40949145e-01,  2.20511329e-26,
             5.86850196e-01,  9.95016851e-01]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.01e+03 4.953e+03 ... 6.056e+05</div><input id='attrs-af91a9f9-8eee-4dbb-991c-8942c687e487' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-af91a9f9-8eee-4dbb-991c-8942c687e487' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-34841941-3681-4c02-a34c-cd345c1913c3' class='xr-var-data-in' type='checkbox'><label for='data-34841941-3681-4c02-a34c-cd345c1913c3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[1.00990116e+03, 4.95261546e+03, 1.23966108e+03, 3.86732134e+05],
            [6.08012367e+02, 3.80876868e+03, 1.58705750e+03, 4.79670834e+05],
            [4.61196194e+02, 3.70985506e+03, 1.44074817e+03, 4.27885648e+05],
            ...,
            [1.09531427e+03, 5.75299910e+03, 1.29907742e+03, 4.16415097e+05],
            [9.34686309e+02, 3.51536166e+03, 1.60892383e+03, 4.41885188e+05],
            [5.78885309e+02, 3.75354553e+03, 1.62634202e+03, 4.86392330e+05]],
    
           [[1.53467407e+03, 7.37884986e+03, 1.37288154e+03, 4.18892379e+05],
            [7.85491258e+02, 4.97781713e+03, 1.74087762e+03, 5.12445871e+05],
            [6.40548301e+02, 4.83759931e+03, 1.59357876e+03, 4.60568070e+05],
            ...,
            [1.91385020e+03, 8.83549635e+03, 1.45303171e+03, 4.50273424e+05],
            [1.11246563e+03, 4.67558192e+03, 1.76250027e+03, 4.74630203e+05],
            [7.56664633e+02, 4.91376579e+03, 1.77991846e+03, 5.19137344e+05]],
    
           [[2.08162657e+03, 1.03044095e+04, 1.48434170e+03, 4.42959028e+05],
            [9.70551734e+02, 6.38751268e+03, 1.86959901e+03, 5.36982063e+05],
            [8.27567125e+02, 6.19740746e+03, 1.72147221e+03, 4.85034019e+05],
            ...,
            [2.76692895e+03, 1.25522745e+04, 1.58180991e+03, 4.75617565e+05],
            [1.29784046e+03, 6.07461523e+03, 1.89101778e+03, 4.99143722e+05],
            [9.42039462e+02, 6.31279910e+03, 1.90843597e+03, 5.43650864e+05]],
    
           [[2.95399830e+03, 1.50944831e+04, 1.61138235e+03, 4.68071391e+05],
            [1.26591640e+03, 8.69521396e+03, 2.01627348e+03, 5.62588738e+05],
            [1.12605770e+03, 8.42343693e+03, 1.86720237e+03, 5.10566766e+05],
            ...,
            [4.12754831e+03, 1.86373107e+04, 1.72854720e+03, 5.02067343e+05],
            [1.59370703e+03, 8.36486125e+03, 2.03745975e+03, 5.24726609e+05],
            [1.23790603e+03, 8.60304513e+03, 2.05487793e+03, 5.69233751e+05]],
    
           [[4.65616830e+03, 2.41035672e+04, 1.79735487e+03, 5.03817165e+05],
            [1.84256259e+03, 1.30344587e+04, 2.23086783e+03, 5.99033916e+05],
            [1.70879944e+03, 1.26092098e+04, 2.08041314e+03, 5.46906720e+05],
            ...,
            [6.78243642e+03, 3.00809143e+04, 1.94330550e+03, 5.39716496e+05],
            [2.17133183e+03, 1.26713049e+04, 2.25171355e+03, 5.61137946e+05],
            [1.81553083e+03, 1.29094888e+04, 2.26913174e+03, 6.05645088e+05]]],
          shape=(5, 216, 4))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-bf594480-bcc2-4b33-9e6e-1183cf2c4ff4' class='xr-section-summary-in' type='checkbox'  ><label for='section-bf594480-bcc2-4b33-9e6e-1183cf2c4ff4' class='xr-section-summary' >Indexes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-d8e676f1-fd59-4fd7-aeef-7326e704f860' class='xr-index-data-in' type='checkbox'/><label for='index-d8e676f1-fd59-4fd7-aeef-7326e704f860' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([ 756,  769,  692,  616,   35,  164,  680,  331,  299,  727,
           ...
             27,  959,   29,  346,  304,  264,  798,  751,  470, 1043],
          dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;, length=216))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-0382fb24-2ab6-4dec-a91b-5b01f4b86bbd' class='xr-index-data-in' type='checkbox'/><label for='index-0382fb24-2ab6-4dec-a91b-5b01f4b86bbd' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-33c11bf1-141b-4100-8c8a-c33019a3b664' class='xr-index-data-in' type='checkbox'/><label for='index-33c11bf1-141b-4100-8c8a-c33019a3b664' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-e733eabd-576c-4fd5-a84a-1c8b83fcf94e' class='xr-index-data-in' type='checkbox'/><label for='index-e733eabd-576c-4fd5-a84a-1c8b83fcf94e' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-1c562747-a270-4e45-844e-f0d70768ba8a' class='xr-index-data-in' type='checkbox'/><label for='index-1c562747-a270-4e45-844e-f0d70768ba8a' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>centile</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-26a61cf7-e3c1-4c75-892d-988de217ed0b' class='xr-index-data-in' type='checkbox'/><label for='index-26a61cf7-e3c1-4c75-892d-988de217ed0b' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0.05, 0.25, 0.5, 0.75, 0.95], dtype=&#x27;float64&#x27;, name=&#x27;centile&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-81bdf164-4de7-44d5-8a22-c2a72e10ec84' class='xr-section-summary-in' type='checkbox'  checked><label for='section-81bdf164-4de7-44d5-8a22-c2a72e10ec84' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [np.str_(&#x27;F&#x27;), np.str_(&#x27;M&#x27;)], np.str_(&#x27;site&#x27;): [np.str_(&#x27;AnnArbor_a&#x27;), np.str_(&#x27;AnnArbor_b&#x27;), np.str_(&#x27;Atlanta&#x27;), np.str_(&#x27;Baltimore&#x27;), np.str_(&#x27;Bangor&#x27;), np.str_(&#x27;Beijing_Zang&#x27;), np.str_(&#x27;Berlin_Margulies&#x27;), np.str_(&#x27;Cambridge_Buckner&#x27;), np.str_(&#x27;Cleveland&#x27;), np.str_(&#x27;ICBM&#x27;), np.str_(&#x27;Leiden_2180&#x27;), np.str_(&#x27;Leiden_2200&#x27;), np.str_(&#x27;Milwaukee_b&#x27;), np.str_(&#x27;Munchen&#x27;), np.str_(&#x27;NewYork_a&#x27;), np.str_(&#x27;NewYork_a_ADHD&#x27;), np.str_(&#x27;Newark&#x27;), np.str_(&#x27;Oulu&#x27;), np.str_(&#x27;Oxford&#x27;), np.str_(&#x27;PaloAlto&#x27;), np.str_(&#x27;Pittsburgh&#x27;), np.str_(&#x27;Queensland&#x27;), np.str_(&#x27;SaintLouis&#x27;)]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x168093a60&gt;, {np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): 589, np.str_(&#x27;M&#x27;): 489}, np.str_(&#x27;site&#x27;): {np.str_(&#x27;AnnArbor_a&#x27;): 24, np.str_(&#x27;AnnArbor_b&#x27;): 32, np.str_(&#x27;Atlanta&#x27;): 28, np.str_(&#x27;Baltimore&#x27;): 23, np.str_(&#x27;Bangor&#x27;): 20, np.str_(&#x27;Beijing_Zang&#x27;): 198, np.str_(&#x27;Berlin_Margulies&#x27;): 26, np.str_(&#x27;Cambridge_Buckner&#x27;): 198, np.str_(&#x27;Cleveland&#x27;): 31, np.str_(&#x27;ICBM&#x27;): 85, np.str_(&#x27;Leiden_2180&#x27;): 12, np.str_(&#x27;Leiden_2200&#x27;): 19, np.str_(&#x27;Milwaukee_b&#x27;): 46, np.str_(&#x27;Munchen&#x27;): 15, np.str_(&#x27;NewYork_a&#x27;): 83, np.str_(&#x27;NewYork_a_ADHD&#x27;): 25, np.str_(&#x27;Newark&#x27;): 19, np.str_(&#x27;Oulu&#x27;): 102, np.str_(&#x27;Oxford&#x27;): 22, np.str_(&#x27;PaloAlto&#x27;): 17, np.str_(&#x27;Pittsburgh&#x27;): 3, np.str_(&#x27;Queensland&#x27;): 19, np.str_(&#x27;SaintLouis&#x27;): 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.251224489795916), &#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(85.0)}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.05332767402377), &#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(85.0)}}, np.str_(&#x27;M&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.48959100204499), &#x27;min&#x27;: np.float64(9.21), &#x27;max&#x27;: np.float64(78.0)}}}, np.str_(&#x27;site&#x27;): {np.str_(&#x27;AnnArbor_a&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.28333333333333), &#x27;min&#x27;: np.float64(13.41), &#x27;max&#x27;: np.float64(40.98)}}, np.str_(&#x27;AnnArbor_b&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(44.40625), &#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(79.0)}}, np.str_(&#x27;Atlanta&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(30.892857142857142), &#x27;min&#x27;: np.float64(22.0), &#x27;max&#x27;: np.float64(57.0)}}, np.str_(&#x27;Baltimore&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(29.26086956521739), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(40.0)}}, np.str_(&#x27;Bangor&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(23.4), &#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(38.0)}}, np.str_(&#x27;Beijing_Zang&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.161616161616163), &#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(26.0)}}, np.str_(&#x27;Berlin_Margulies&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(29.76923076923077), &#x27;min&#x27;: np.float64(23.0), &#x27;max&#x27;: np.float64(44.0)}}, np.str_(&#x27;Cambridge_Buckner&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.03030303030303), &#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(30.0)}}, np.str_(&#x27;Cleveland&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(43.54838709677419), &#x27;min&#x27;: np.float64(24.0), &#x27;max&#x27;: np.float64(60.0)}}, np.str_(&#x27;ICBM&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(44.04705882352941), &#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(85.0)}}, np.str_(&#x27;Leiden_2180&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(23.0), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(27.0)}}, np.str_(&#x27;Leiden_2200&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.68421052631579), &#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(28.0)}}, np.str_(&#x27;Milwaukee_b&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(53.58695652173913), &#x27;min&#x27;: np.float64(44.0), &#x27;max&#x27;: np.float64(65.0)}}, np.str_(&#x27;Munchen&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(68.13333333333334), &#x27;min&#x27;: np.float64(63.0), &#x27;max&#x27;: np.float64(74.0)}}, np.str_(&#x27;NewYork_a&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(24.507710843373495), &#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(49.16)}}, np.str_(&#x27;NewYork_a_ADHD&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(34.9952), &#x27;min&#x27;: np.float64(20.69), &#x27;max&#x27;: np.float64(50.9)}}, np.str_(&#x27;Newark&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(24.105263157894736), &#x27;min&#x27;: np.float64(21.0), &#x27;max&#x27;: np.float64(39.0)}}, np.str_(&#x27;Oulu&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.519607843137255), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(23.0)}}, np.str_(&#x27;Oxford&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(29.0), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(35.0)}}, np.str_(&#x27;PaloAlto&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(32.470588235294116), &#x27;min&#x27;: np.float64(22.0), &#x27;max&#x27;: np.float64(46.0)}}, np.str_(&#x27;Pittsburgh&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(32.333333333333336), &#x27;min&#x27;: np.float64(25.0), &#x27;max&#x27;: np.float64(47.0)}}, np.str_(&#x27;Queensland&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(25.94736842105263), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(34.0)}}, np.str_(&#x27;SaintLouis&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(25.096774193548388), &#x27;min&#x27;: np.float64(21.0), &#x27;max&#x27;: np.float64(29.0)}}}}</dd></dl></div></li></ul></div></div>



Harmonize
~~~~~~~~~

.. code:: ipython3

    # Harmonizing is also easy:
    reference_batch_effect = {
        "site": "Beijing_Zang",
        "sex": "M",
    }  # Set a pseudo-batch effect. I.e., this means 'pretend that all data was from this site and sex'
    
    # model.harmonize(train, reference_batch_effect=reference_batch_effect)  # <- easy
    
    plt.style.use("seaborn-v0_8")
    df = train.to_dataframe()
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



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_29_0.png


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
    synthetic_data = model.synthesize(covariate_range_per_batch_effect=True, n_samples=1000)  # <- also easy
    # Show the synthetic data along with the centiles
    plot_centiles(
        model,
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=synthetic_data,
    )



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_31_0.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_31_1.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_31_2.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_31_3.png


.. code:: ipython3

    # Synthesize new Y data for existing X data
    new_test_data = test.copy()
    
    # Remove the Y data, this way we will synthesize new Y data for the existing X data
    if hasattr(new_test_data, "Y"):
        del new_test_data["Y"]
    
    synthetic = model.synthesize(new_test_data)  # <- will fill in the missing Y data

.. code:: ipython3

    plot_centiles_advanced(
        model,
        centiles=[0.05, 0.5, 0.95],  # Plot arbitrary centiles
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=synthetic,  # Scatter the train data points
        batch_effects="all",  # You can set this to "all" to show all batch effects
        show_other_data=False,  # Show data points that do not match any batch effects
        show_centile_labels=True,
        harmonize_data=True,  # Set this to False to see the difference
        show_legend=False,  # Don't show the legend because it crowds the plot
    )



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_33_0.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_33_1.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_33_2.png



.. image:: 04_HBR_SHASH_files/04_HBR_SHASH_33_3.png


Next steps
----------

Please see the other tutorials for more examples, and we also recommend
you to read the documentation! As this toolkit is still in development,
the documentation may not be up to date. If you find any issues, please
let us know!

Also, feel free to contact us on Github if you have any questions or
suggestions.

Have fun modeling!

Bonus content
~~~~~~~~~~~~~

Here is another model configuration using a SHASH likelihood, but this
one also has a linear regression in epsilon and delta. If you have a
feature that is heavily skewed and for which the skewness also changes
with the covariates, this is the model for you:

.. code:: ipython3

    # Here's a model with a SHASHb likelihood, with a linear regression in all four parameters, so including epsilon and delta.
    # This is a very flexible model, but it will also take a lot longer to run.
    mu = make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 10.0)),
        intercept=make_prior(
            random=True,
            mu=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
            sigma=make_prior(dist_name="Gamma", dist_params=(3.0, 1.0)),
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
    
    epsilon = make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 1.0)),
        intercept=make_prior(dist_name="Normal", dist_params=(0.0, 1.0)),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    
    delta = make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 1.0)),
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        mapping="softplus",
        mapping_params=(
            0.0,
            3.0,  # Scale for smoothness
            0.6,  # We need to provide a vertical shift as well, because the SHASH mapping goes a bit wild with low values for delta
        ),
    )
    
    shashb2_regression_model = HBR(
        name="template",
        cores=16,
        progressbar=True,
        draws=1500,
        tune=500,
        chains=4,
        nuts_sampler="nutpie",
        likelihood=SHASHbLikelihood(mu, sigma, epsilon, delta),
    )
