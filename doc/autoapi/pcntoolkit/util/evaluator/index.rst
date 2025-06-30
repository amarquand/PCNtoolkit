pcntoolkit.util.evaluator
=========================

.. py:module:: pcntoolkit.util.evaluator


Classes
-------

.. autoapisummary::

   pcntoolkit.util.evaluator.Evaluator


Module Contents
---------------

.. py:class:: Evaluator

   A class for evaluating normative model predictions.

   This class implements various statistics to assess the quality of
   normative model predictions, including correlation coefficients, error metrics,
   and normality tests.

   .. attribute:: response_vars

      List of response variables to evaluate

      :type: :py:class:`List[str]`

   .. attribute:: Initialize the Evaluator.

      


   .. py:method:: create_statistics_group(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Create a statistics group in the data container.

      :param data: Data container to add statistics group to
      :type data: :py:class:`NormData`



   .. py:method:: empty_statistic() -> xarray.DataArray


   .. py:method:: evaluate(data: pcntoolkit.dataio.norm_data.NormData, statistics: List[str] = []) -> pcntoolkit.dataio.norm_data.NormData

      Evaluate model predictions using multiple statistics.

      :param data: Data container with predictions and actual values, and yhat
      :type data: :py:class:`NormData`

      :returns: Data container updated with evaluation statistics
      :rtype: :py:class:`NormData`



   .. py:method:: evaluate_R2(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Evaluate R2 for model predictions.



   .. py:method:: evaluate_bic(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Evaluate Bayesian Information Criterion (BIC) for model predictions.

      BIC is a criterion for model selection that statistics the trade-off between
      model fit and complexity.

      :param data: Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
      :type data: :py:class:`NormData`



   .. py:method:: evaluate_expv(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Evaluate Explained Variance score for model predictions.

      The explained variance score statistics the proportion of variance in the target variable
      that is predictable from the input features.

      :param data: Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
      :type data: :py:class:`NormData`



   .. py:method:: evaluate_mace(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Evaluate Mean Absolute Centile Error.



   .. py:method:: evaluate_mape(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Evaluate Mean Absolute Percentage Error.



   .. py:method:: evaluate_msll(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Evaluate Mean Standardized Log Loss (MSLL) for model predictions.

      MSLL compares the log loss of the model to that of a simple baseline predictor
      that always predicts the mean of the training data.

      :param data: Data container with predictions and actual values. Must contain 'y', 'Yhat',
                   and standard deviation predictions.
      :type data: :py:class:`NormData`



   .. py:method:: evaluate_nll(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Evaluate Negative Log Likelihood (NLL) for model predictions.

      NLL statistics the probabilistic accuracy of the model's predictions, assuming
      binary classification targets.

      :param data: Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
                   'y' should contain binary values (0 or 1).
      :type data: :py:class:`NormData`



   .. py:method:: evaluate_rho(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Evaluate Spearman's rank correlation coefficient.

      :param data: Data container with predictions and actual values
      :type data: :py:class:`NormData`



   .. py:method:: evaluate_rmse(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Evaluate Root Mean Square Error (RMSE) for model predictions.

      :param data: Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
      :type data: :py:class:`NormData`



   .. py:method:: evaluate_shapiro_w(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Evaluate Shapiro-Wilk test statistic for normality of residuals.

      The Shapiro-Wilk test assesses whether the z-scores follow a normal distribution.
      A higher W statistic (closer to 1) indicates stronger normality.

      :param data: Data container with predictions and actual values. Must contain 'zscores' variable.
      :type data: :py:class:`NormData`



   .. py:method:: evaluate_smse(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Evaluate Standardized Mean Square Error (SMSE) for model predictions.

      SMSE normalizes the mean squared error by the variance of the target variable,
      making it scale-independent.

      :param data: Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
      :type data: :py:class:`NormData`



   .. py:method:: n_params() -> int

      Return the number of parameters in the model.



   .. py:method:: prepare(responsevar: str) -> None

      Prepare the evaluator for a specific response variable.



   .. py:method:: reset() -> None

      Reset the evaluator state.



   .. py:attribute:: response_vars
      :type:  List[str]
      :value: []



