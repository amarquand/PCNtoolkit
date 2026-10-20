Exploratory BIC approximation for HBR
=====================================

``HBR.bic_approx(X, be, Y)`` provides an opt-in, likelihood-aware score
for exploratory comparisons. It uses the model's likelihood, including
SHASH when configured, instead of a Gaussian residual-sum-of-squares formula.
The calculation is implemented entirely in HBR; it does not change the
evaluator, model fitting, or the default evaluation metrics.

.. warning::

   **Report this score as BIC_approx, not BIC.** It is not a validated
   hierarchical BIC or an estimate of model evidence. Using the model's
   likelihood improves its distributional fidelity, but does not guarantee
   a more accurate BIC or better model ranking.

Calculation
-----------

For each stored draw of the model parameters, the method sums the
conditional log likelihoods of all training observations. It selects the
largest of these totals and applies a provisional structural-count penalty:

.. math::

   \ell_{\mathrm{best}} = \max_s \sum_{i=1}^{n} \log p(y_i \mid \theta_s),
   \qquad
   \mathrm{BIC}_{\mathrm{approx}} = -2\ell_{\mathrm{best}} + k\log n.

Here, ``s`` ranges over all chains and draws, ``n`` is the number of supplied
observations, and ``k`` is the existing ``HBR.n_parameters()`` count.
Group effects are conditioned on, not integrated out. Prior log densities
are not included in the likelihood.

The sum is taken **before** selecting a draw. For example, two draws with
observation log likelihoods ``[-10, -30]`` and ``[-25, -12]`` have totals
``-40`` and ``-37``. The selected total is ``-37``, not ``-22`` obtained by
choosing a different draw for each observation.

This calculation does not use MLL: MLL averages likelihoods over draws
before taking their logarithm, whereas this score selects one joint draw.

Using a fitted normative model
------------------------------

Supply the **original training data**, not held-out data, and apply exactly
the preprocessing used for fitting. The method cannot verify training-set
identity. The following example assumes ``model`` is a fitted
``NormativeModel`` containing HBR regressions and ``training_data`` is the
original, unprocessed ``NormData``. Use a copy to preserve the supplied data:

.. code-block:: python

   model.check_is_fitted()
   prepared = training_data.copy(deep=True)
   model.preprocess(prepared)

   scores = {}
   for response in prepared.response_var_list:
       response_data = prepared.sel(response_vars=response)
       X, be, _, Y, _ = model.extract_data(response_data)
       regression = model.regression_models[response]
       scores[response] = regression.bic_approx(X, be, Y)

   print(scores)  # Label these values BIC_approx in reports.

Each successful call emits a warning about the approximation. Unfitted
models, empty likelihood arrays, nonfinite likelihoods, and unavailable
parameter counts raise errors rather than producing a misleading score.
The score is returned separately; it is not added to default statistics.

Limits on interpretation
------------------------

* The best stored draw need not maximize the likelihood. The score depends
  on prior choices, the inference method, and the number and quality of
  draws. Adding draws can lower the score even without changing the model.
  Similar sampling budgets do not remove this limitation.
* ``k`` counts structural free dimensions, including hierarchical
  parameters. It is not an effective-complexity estimate or a justified
  hierarchical BIC penalty. Likewise, ``log(n)`` does not account for the
  different information supplied by participants and groups.
* Restrict exploratory comparisons to the same response, training
  observations, hierarchical structure, and fitted response scale. Do not
  use this approximation to select different random-effect structures.
* Scores are calculated on the regression scale. No extra Jacobian is
  added for ``NormativeModel`` response transforms or scaling. Compared
  models must use identical fitted transforms and scalers; scores across
  different response scales are not comparable.
* For this initial use, use independent participants at existing sites.
  Repeated measurements and transferred models need a separate assessment
  of the data contribution and penalty.
* Posterior approximations such as ADVI, Pathfinder, or Laplace introduce
  additional error; this method does not correct that error or assess
  convergence. Lower scores are only a provisional preference under the
  same comparison conditions, not formal evidence or Bayes factors.

A finalized hierarchical BIC requires a justified likelihood estimate and
matching parameter and sample-size penalties. Those remain future work.
