pcntoolkit.util.plotter
=======================

.. py:module:: pcntoolkit.util.plotter

.. autoapi-nested-parse::

   A module for plotting functions.



Functions
---------

.. autoapisummary::

   pcntoolkit.util.plotter.plot_centiles
   pcntoolkit.util.plotter.plot_qq
   pcntoolkit.util.plotter.plot_ridge


Module Contents
---------------

.. py:function:: plot_centiles(model: pcntoolkit.normative_model.NormativeModel, centiles: List[float] | numpy.ndarray | None = None, covariate: str | None = None, covariate_range: tuple[float, float] = (None, None), batch_effects: Dict[str, List[str]] | None | Literal['all'] = None, scatter_data: pcntoolkit.dataio.norm_data.NormData | None = None, harmonize_data: bool = True, hue_data: str = 'site', markers_data: str = 'sex', show_other_data: bool = False, show_thrivelines: bool = False, z_thrive: float = 0.0, save_dir: str | None = None, show_centile_labels: bool = True, show_legend: bool = True, show_yhat: bool = False, plt_kwargs: dict | None = None, **kwargs: Any) -> None

   Generate centile plots for response variables with optional data overlay.

   This function creates visualization of centile curves for all response variables
   in the dataset. It can optionally show the actual data points overlaid on the
   centile curves, with customizable styling based on categorical variables.

   :param model: The model to plot the centiles for.
   :type model: :py:class:`NormativeModel`
   :param centiles: The centiles to plot. If None, the default centiles will be used.
   :type centiles: :py:class:`List[float] | np.ndarray | None`, *optional*
   :param covariate: The covariate to plot on the x-axis. If None, the first covariate in the model will be used.
   :type covariate: :py:class:`str | None`, *optional*
   :param covariate_range: The range of the covariate to plot on the x-axis. If None, the range of the covariate that was in the train data will be used.
   :type covariate_range: :py:class:`tuple[float`, :py:class:`float]`, *optional*
   :param batch_effects: The batch effects to plot the centiles for. If None, the batch effect that appears first in alphabetical order will be used.
   :type batch_effects: :py:class:`Dict[str`, :py:class:`List[str]] | None | Literal[```"all"``:py:class:`]`, *optional*
   :param scatter_data: Data to scatter on top of the centiles.
   :type scatter_data: :py:class:`NormData | None`, *optional*
   :param harmonize_data: Whether to harmonize the scatter data before plotting. Data will be harmonized to the batch effect for which the centiles were computed.
   :type harmonize_data: :py:class:`bool`, *optional*
   :param hue_data: The column to use for color coding the data. If None, the data will not be color coded.
   :type hue_data: :py:class:`str`, *optional*
   :param markers_data: The column to use for marker styling the data. If None, the data will not be marker styled.
   :type markers_data: :py:class:`str`, *optional*
   :param show_other_data: Whether to scatter data belonging to groups not in batch_effects.
   :type show_other_data: :py:class:`bool`, *optional*
   :param save_dir: The directory to save the plot to. If None, the plot will not be saved.
   :type save_dir: :py:class:`str | None`, *optional*
   :param show_centile_labels: Whether to show the centile labels on the plot.
   :type show_centile_labels: :py:class:`bool`, *optional*
   :param show_legend: Whether to show the legend on the plot.
   :type show_legend: :py:class:`bool`, *optional*
   :param plt_kwargs: Additional keyword arguments for the plot.
   :type plt_kwargs: :py:class:`dict`, *optional*
   :param \*\*kwargs: Additional keyword arguments for the model.compute_centiles method.
   :type \*\*kwargs: :py:class:`Any`, *optional*

   :returns: Displays the plot using matplotlib.
   :rtype: :py:obj:`None`


.. py:function:: plot_qq(data: pcntoolkit.dataio.norm_data.NormData, plt_kwargs: dict | None = None, bound: int | float = 0, plot_id_line: bool = False, hue_data: str | None = None, markers_data: str | None = None, split_data: str | None = None, seed: int = 42, save_dir: str | None = None) -> None

   Plot QQ plots for each response variable in the data.

   :param data: Data containing the response variables.
   :type data: :py:class:`NormData`
   :param plt_kwargs: Additional keyword arguments for the plot. Defaults to None.
   :type plt_kwargs: :py:class:`dict` or :py:obj:`None`, *optional*
   :param bound: Axis limits for the plot. Defaults to 0.
   :type bound: :py:class:`int` or :py:class:`float`, *optional*
   :param plot_id_line: Whether to plot the identity line. Defaults to False.
   :type plot_id_line: :py:class:`bool`, *optional*
   :param hue_data: Column to use for coloring. Defaults to None.
   :type hue_data: :py:class:`str` or :py:obj:`None`, *optional*
   :param markers_data: Column to use for marker styling. Defaults to None.
   :type markers_data: :py:class:`str` or :py:obj:`None`, *optional*
   :param split_data: Column to use for splitting data. Defaults to None.
   :type split_data: :py:class:`str` or :py:obj:`None`, *optional*
   :param seed: Random seed for reproducibility. Defaults to 42.
   :type seed: :py:class:`int`, *optional*

   :rtype: :py:obj:`None`

   .. rubric:: Examples

   >>> plot_qq(data, plt_kwargs={"figsize": (10, 6)}, bound=3)


.. py:function:: plot_ridge(data: pcntoolkit.dataio.norm_data.NormData, variable: Literal['Z', 'Y'], split_by: str, save_dir: str | None = None, **kwargs: Any) -> None

   Plot a ridge plot for each response variable in the data.

   Creates a density plot for the variable split by the split_by variable.

   Each density plot will be on a different row.

   The hue of the density plot will be the split_by variable.

   :param data: Data containing the response variable.
   :type data: :py:class:`NormData`
   :param variable: The variable to plot on the x-axis. (Z or Y)
   :type variable: :py:class:`Literal[```"Z"``, ``"Y"``:py:class:`]`
   :param split_by: The variable to split the data by.
   :type split_by: :py:class:`str`
   :param save_dir: The directory to save the plot to. Defaults to None.
   :type save_dir: :py:class:`str | None`, *optional*
   :param \*\*kwargs: Additional keyword arguments for the plot.
   :type \*\*kwargs: :py:class:`Any`, *optional*

   :rtype: :py:obj:`None`


