pcntoolkit.random
=================

.. py:module:: pcntoolkit.random

.. autoapi-nested-parse::

   Created on Fri Sep 27 13:34:31 2024

   @author: johbay



Attributes
----------

.. autoapisummary::

   pcntoolkit.random.OAS2_0157
   pcntoolkit.random.Z_measures
   pcntoolkit.random.atlas
   pcntoolkit.random.base_dir
   pcntoolkit.random.batch_dir
   pcntoolkit.random.data
   pcntoolkit.random.data_all
   pcntoolkit.random.data_demented
   pcntoolkit.random.data_dir
   pcntoolkit.random.filtered_subjects
   pcntoolkit.random.idp_nr
   pcntoolkit.random.idps
   pcntoolkit.random.measure
   pcntoolkit.random.measures
   pcntoolkit.random.model_type
   pcntoolkit.random.plot_data
   pcntoolkit.random.test_raw
   pcntoolkit.random.test_raw_demented


Functions
---------

.. autoapisummary::

   pcntoolkit.random.Fisher_transform
   pcntoolkit.random.create_regression_matrix
   pcntoolkit.random.fisher_transform
   pcntoolkit.random.flatten
   pcntoolkit.random.generate_thrive_lines
   pcntoolkit.random.get_subdiagonal
   pcntoolkit.random.get_thrive_lines
   pcntoolkit.random.make_correlation_matrix
   pcntoolkit.random.make_velocity_plots
   pcntoolkit.random.predicted_r
   pcntoolkit.random.recursive_multiply
   pcntoolkit.random.recursive_r_transform
   pcntoolkit.random.reverse_z_transform


Module Contents
---------------

.. py:function:: Fisher_transform(r)

   Takes a correlation r and returns the Fisher-transformed value Phi.


.. py:function:: create_regression_matrix(Phi: numpy.ndarray, gap: int) -> pandas.DataFrame

   Create a regression matrix for a given diagonal gap.


.. py:function:: fisher_transform(cor: float) -> float

   Apply Fisher transformation to correlation values.


.. py:function:: flatten(xss)

.. py:function:: generate_thrive_lines(pred_r: list, start_year: int, end_year: int, velocity_length: int, start_z: int, end_z: int, space_between_anchors: int, z_thrive: float = None) -> tuple[numpy.ndarray, numpy.ndarray]

   Generate thrive lines for downward or upward trends.

   Parameters:
   - pred_r (list): List of predicted correlation values.
   - start_year (int): Starting year.
   - end_year (int): Ending year.
   - velocity_length (int): Length of years thirve lines should predict in the futurw
   - start_z (int): Lowest (starting) z value, for example -3
   - end_z (int): Higest (ending)  z value, for example 4
   - space_between_anchors (int): Spacing between anchors, for example thrive lines start every (1), every second (2) year etc.
   - z_thrive (float, optional): Additional z thrive factor for upward thrive. default is -1.96

   Returns:
   - velocity_list_age (np.ndarray): Flattened array of years for velocity.
   - velocity (np.ndarray): Flattened array of thrive line results.


.. py:function:: get_subdiagonal(A, gap=1)

   :param A:
   :type A: :py:class:`correlation matrix.`
   :param gap:
   :type gap: :py:class:`offset`

   :returns: **sub_diagonal**
   :rtype: :py:class:`diagonal below the sub-diagnonal`


.. py:function:: get_thrive_lines(lst, step_results=None, current_result=1, z_thrive=-1.96)

.. py:function:: make_correlation_matrix(data: pandas.DataFrame, measure: str, re_estimate: bool = True) -> tuple[numpy.ndarray, numpy.ndarray]

   Creates a correlation matrix for age-based data.

   For each pair i, j, of ages:
       - Find all subjects that have data for both ages
       - Compute the correlation between the two measures for those subjects
       - Store the correlation in the matrix at position (i, j)

   Parameters:
   - data (pd.DataFrame): Data containing age, subject ID, visits, and measure columns.
   - measure (str): The column name of the measure to compute correlations.
   - re_estimate (bool): Whether to compute the correlation matrix.

   Returns:
   - A (np.ndarray): The correlation matrix.
   - Phi (np.ndarray): The Fisher-transformed correlation matrix.


.. py:function:: make_velocity_plots(idp_nr, measure, data=data, seletced_sex='male', model_type='SHAHSb_1', re_estimate=True)

.. py:function:: predicted_r(Phi)

   :param Phi: Fisher transformed correlation valuee.
   :type Phi: :py:class:`member` of :py:class:`type list`

   :returns: **r_pred** -- correlation value in r space.
   :rtype: :py:class:`single value`


.. py:function:: recursive_multiply(lst, step_results=None, current_product=1)

.. py:function:: recursive_r_transform(predicted_values: numpy.ndarray) -> list[float]

   Apply inverse Fisher transformation and return correlation coefficients.


.. py:function:: reverse_z_transform(z, mu, sigma)

.. py:data:: OAS2_0157

.. py:data:: Z_measures

.. py:data:: atlas
   :value: 'SC'


.. py:data:: base_dir
   :value: '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/'


.. py:data:: batch_dir
   :value: '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_...


.. py:data:: data

.. py:data:: data_all
   :value: None


.. py:data:: data_demented

.. py:data:: data_dir

.. py:data:: filtered_subjects

.. py:data:: idp_nr
   :value: 0


.. py:data:: idps

.. py:data:: measure

.. py:data:: measures

.. py:data:: model_type
   :value: 'SHASHb_1'


.. py:data:: plot_data

.. py:data:: test_raw

.. py:data:: test_raw_demented

