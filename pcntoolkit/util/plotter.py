"""A module for plotting functions."""

from re import S
from typing import TYPE_CHECKING, Any, Dict, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib.font_manager import FontProperties

from pcntoolkit.dataio.norm_data import NormData

if TYPE_CHECKING:
    from pcntoolkit.normative_model import NormativeModel
import os

from pcntoolkit.util.output import Errors, Output


def plot_centiles(
    model: "NormativeModel",
    data: NormData,
    centiles: List[float] | np.ndarray | None = None,
    covariate: str | None = None,
    batch_effects: Dict[str, List[str]] | None | Literal["all"] = None,
    show_data: bool = False,
    harmonize_data: bool = True,
    plt_kwargs: dict | None = None,
    hue_data: str = "site",
    markers_data: str = "sex",
    show_other_data: bool = False,
    save_dir: str | None = None,
    show_centile_labels: bool = True,
    show_legend: bool = True,
    **kwargs: Any,
) -> None:
    """Generate centile plots for response variables with optional data overlay.

    This function creates visualization of centile curves for all response variables
    in the dataset. It can optionally show the actual data points overlaid on the
    centile curves, with customizable styling based on categorical variables.

    Parameters
    ----------
    model : NormBase
        The fitted normative model used to generate centile predictions.
    data : NormData
        Dataset containing covariates and response variables to be plotted.
    centiles : list | None, optional
        List of centile values to plot. If None, uses default values, by default None.
    covariate : str | None, optional
        Name of the covariate to plot on x-axis. If None, uses the first
        covariate in the dataset, by default None.
    batch_effects : Dict[str, List[str]] | None, optional
        Specification of batch effects for plotting. Format:
        {'batch_effect_name': ['value1', 'value2', ...]}
        For models with random effects, specifies which batch effect values
        to use for centile computation (first value in each list).
        For data visualization, specifies which batch effects to highlight.
        By default None.
    show_data : bool, optional
        If True, overlays actual data points on the centile curves.
        Points matching batch_effects are highlighted, others are shown
        in light gray, by default False.
    plt_kwargs : dict | None, optional
        Additional keyword arguments passed to plt.figure(),
        by default None.
    hue_data : str, optional
        Column name in data used for color-coding points when show_data=True,
        by default "site".
    markers_data : str, optional
        Column name in data used for marker styles when show_data=True,
        by default "sex".
    show_other_data : bool, optional
        Whether to show data points that do not match any batch effects,
        by default False.
    save_dir : str | None, optional
        Directory to save the plot, by default None.
    show_centile_labels : bool, optional
        Whether to show the centile labels, by default True.
    show_legend : bool, optional
        Whether to show the legend, by default True.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Raises
    ------
    ValueError
        If batch_effects dictionary contains invalid value types.

    Notes
    -----
    - Centile lines are styled differently based on their distance from the median:
      - Solid lines for centiles close to median (|cdf - 0.5| < 0.25)
      - Dashed lines for intermediate centiles (0.25 ≤ |cdf - 0.5| < 0.475)
      - Dotted lines for extreme centiles (|cdf - 0.5| ≥ 0.475)
    - CDF values are displayed at both ends of each centile line
    - When showing data with batch effects, matching points are highlighted
      while others are shown in gray with reduced opacity

    Examples
    --------
    >>> # Basic centile plot
    >>> plot_centiles(model, data, covariate="age")

    >>> # With data overlay and batch effects
    >>> plot_centiles(
    ...     model,
    ...     data,
    ...     centiles=[0.1587, 0.8413],
    ...     covariate="age",
    ...     batch_effects={"site": ["site1", "site2"]},
    ...     show_data=True,
    ...     hue_data="site",
    ...     markers_data="sex",
    ... )
    """
    if covariate is None:
        covariate = data.covariates[0].to_numpy().item()
        assert isinstance(covariate, str)

    if batch_effects == "all":
        batch_effects = model.unique_batch_effects
    elif batch_effects is None:
        if model.has_batch_effect:
            batch_effects = data.get_single_batch_effect()
        else:
            batch_effects = {}

    if harmonize_data:
        data = model.harmonize(data, {k: v[0] for k, v in batch_effects.items()})

    # Ensure that the batch effects are in the correct format
    if batch_effects:
        for k, v in batch_effects.items():
            if isinstance(v, str):
                batch_effects[k] = [v]
            elif not isinstance(v, list):
                raise Output.error(Errors.ERROR_BATCH_EFFECTS_NOT_LIST, batch_effect_type=type(v))

    if plt_kwargs is None:
        plt_kwargs = {}
    palette = plt_kwargs.pop("cmap", "viridis")
    synth_data = data.create_synthetic_data(
        n_datapoints=150,
        range_dim=covariate,
        batch_effects_to_sample={k: [v[0]] for k, v in batch_effects.items()} if batch_effects else None,
    )
    model.compute_centiles(synth_data, centiles=centiles, **kwargs)
    for response_var in data.coords["response_vars"].to_numpy():
        _plot_centiles(
            data=data,
            synth_data=synth_data,
            response_var=response_var,
            covariate=covariate,
            batch_effects=batch_effects,
            show_data=show_data,
            harmonize_data=harmonize_data,
            plt_kwargs=plt_kwargs,
            hue_data=hue_data,
            markers_data=markers_data,
            palette=palette,
            save_dir=save_dir,
            show_other_data=show_other_data,
            show_centile_labels=show_centile_labels,
            show_legend=show_legend,
        )


def _plot_centiles(
    data: NormData,
    synth_data: NormData,
    response_var: str,
    batch_effects: Dict[str, List[str]],
    covariate: str = None,  # type: ignore
    show_data: bool = False,
    harmonize_data: bool = True,
    plt_kwargs: dict = None,  # type: ignore
    hue_data: str = "site",
    markers_data: str = "sex",
    palette: str = "viridis",
    save_dir: str | None = None,
    show_other_data: bool = False,
    show_centile_labels: bool = True,
    show_legend: bool = True,
) -> None:
    """Plot centile curves for a single response variable.

    Parameters
    ----------
    data : NormData
        Original dataset containing response variables and covariates.
    synth_data : NormData
        Synthetic data containing computed centiles.
    response_var : str
        Name of the response variable to plot.
    batch_effects : Dict[str, List[str]]
        Dictionary specifying batch effects to highlight in the plot.
    covariate : str, optional
        Name of the covariate for x-axis.
    show_data : bool, optional
        If True, overlay data points on centile curves.
    plt_kwargs : dict, optional
        Additional keyword arguments for plt.figure().
    hue_data : str, optional
        Column name for color-coding points, by default "site".
    markers_data : str, optional
        Column name for marker styles, by default "sex".
    palette : str, optional
        Color palette name for centile curves, by default "viridis".

    Notes
    -----
    - Centile line styles vary based on distance from median
    - CDF values are displayed at both ends of centile lines
    - When showing data, batch effect points are highlighted
    """

    sns.set_style("whitegrid")
    plt.figure(**plt_kwargs)
    cmap = plt.get_cmap(palette)

    filter_dict = {
        "covariates": covariate,
        "response_vars": response_var,
    }

    filtered = synth_data.sel(filter_dict)

    for centile in synth_data.coords["centile"][::-1]:
        d_mean = abs(centile - 0.5)
        if d_mean < 0.25:
            style = "-"
        elif d_mean < 0.475:
            style = "--"
        else:
            style = ":"

        sns.lineplot(
            x=filtered.X,
            y=filtered.centiles.sel(centile=centile),
            color=cmap(centile),
            linestyle=style,
            linewidth=1,
            zorder=2,
            legend="brief",
        )
        color = cmap(centile)
        font = FontProperties()
        font.set_weight("bold")
        if show_centile_labels:
            plt.text(
                s=centile.item(),
                x=filtered.X[0] - 1,
                y=filtered.centiles.sel(centile=centile)[0],
                color=color,
                horizontalalignment="right",
                verticalalignment="center",
                fontproperties=font,
            )
            plt.text(
                s=centile.item(),
                x=filtered.X[-1] + 1,
                y=filtered.centiles.sel(centile=centile)[-1],
                color=color,
                horizontalalignment="left",
                verticalalignment="center",
                fontproperties=font,
            )

    minx, maxx = plt.xlim()
    plt.xlim(minx - 0.1 * (maxx - minx), maxx + 0.1 * (maxx - minx))

    if show_data:
        df = data.sel(filter_dict).to_dataframe()
        scatter_data = "Y_harmonized" if harmonize_data else "Y"

        columns = [("X", covariate), (scatter_data, response_var)]
        columns.extend([("batch_effects", be.item()) for be in data.batch_effect_dims])
        df = df[columns]
        df.columns = [c[1] for c in df.columns]

        if batch_effects == {}:
            sns.scatterplot(
                df,
                x=covariate,
                y=response_var,
                label=data.name,
                color="black",
                s=20,
                alpha=0.6,
                zorder=1,
                linewidth=0,
            )
        else:
            idx = np.full(len(df), True)
            for j in batch_effects:
                idx = np.logical_and(
                    idx,
                    df[j].isin(batch_effects[j]),
                )
            be_df = df[idx]
            scatter = sns.scatterplot(
                data=be_df,
                x=covariate,
                y=response_var,
                hue=hue_data if hue_data in df else None,
                style=markers_data if markers_data in df else None,
                s=50,
                alpha=0.8,
                zorder=1,
                linewidth=0,
            )
            if show_other_data:
                non_be_df = df[~idx]
                non_be_df["marker"] = ["Other data"] * len(non_be_df)
                sns.scatterplot(
                    data=non_be_df,
                    x=covariate,
                    y=response_var,
                    color="black",
                    style="marker",
                    linewidth=0,
                    s=20,
                    alpha=0.4,
                    zorder=0,
                )
            if show_legend:
                legend = scatter.get_legend()
                if legend:
                    handles = legend.legend_handles
                    labels = [t.get_text() for t in legend.get_texts()]
                    plt.legend(
                        handles,
                        labels,
                        title_fontsize=10,
                    )
            else:
                plt.legend().remove()
    title = f"Centiles of {response_var}"
    if show_data:
        if harmonize_data:
            title = f"{title}\n With harmonized {data.name} data"
        else:
            title = f"{title}\n With raw {data.name} data"
    plt.title(title)
    plt.xlabel(covariate)
    plt.ylabel(response_var)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"centiles_{response_var}_{data.name}.png"))
    else:
        plt.show(block=False)
    plt.close()


def plot_qq(
    data: NormData,
    plt_kwargs: dict | None = None,
    bound: int | float = 0,
    plot_id_line: bool = False,
    hue_data: str | None = None,
    markers_data: str | None = None,
    split_data: str | None = None,
    seed: int = 42,
    save_dir: str | None = None,
) -> None:
    """
    Plot QQ plots for each response variable in the data.

    Parameters
    ----------
    data : NormData
        Data containing the response variables.
    plt_kwargs : dict or None, optional
        Additional keyword arguments for the plot. Defaults to None.
    bound : int or float, optional
        Axis limits for the plot. Defaults to 0.
    plot_id_line : bool, optional
        Whether to plot the identity line. Defaults to False.
    hue_data : str or None, optional
        Column to use for coloring. Defaults to None.
    markers_data : str or None, optional
        Column to use for marker styling. Defaults to None.
    split_data : str or None, optional
        Column to use for splitting data. Defaults to None.
    seed : int, optional
        Random seed for reproducibility. Defaults to 42.

    Returns
    -------
    None

    Examples
    --------
    >>> plot_qq(data, plt_kwargs={"figsize": (10, 6)}, bound=3)
    """
    plt_kwargs = plt_kwargs or {}
    for response_var in data.coords["response_vars"].to_numpy():
        _plot_qq(
            data,
            response_var,
            plt_kwargs,
            bound,
            plot_id_line,
            hue_data,
            markers_data,
            split_data,
            seed,
            save_dir,
        )


def _plot_qq(
    data: NormData,
    response_var: str,
    plt_kwargs: dict,
    bound: float = 0,
    plot_id_line: bool = False,
    hue_data: str | None = None,
    markers_data: str | None = None,
    split_data: str | None = None,
    seed: int = 42,
    save_dir: str | None = None,
) -> None:
    """
    Plot a QQ plot for a single response variable.

    Parameters
    ----------
    data : NormData
        Data containing the response variable.
    response_var : str
        The response variable to plot.
    plt_kwargs : dict
        Additional keyword arguments for the plot.
    bound : float, optional
        Axis limits for the plot. Not used if 0. Defaults to 0.
    plot_id_line : bool, optional
        Whether to plot the identity line. Defaults to False.
    hue_data : str or None, optional
        Column to use for coloring. Defaults to None.
    markers_data : str or None, optional
        Column to use for marker styling. Defaults to None.
    split_data : str or None, optional
        Column to use for splitting data. Defaults to None. All split data will be offset by 1.
    seed : int, optional
        Random seed for reproducibility. Defaults to 42.
    save_dir: str | None = None,

    Returns
    -------
    None

    Examples
    --------
    >>> _plot_qq(data, "response_var", plt_kwargs={"figsize": (10, 6)}, bound=3)
    """
    np.random.seed(seed)
    sns.set_style("whitegrid")
    filter_dict = {
        "response_vars": response_var,
    }
    filt = data.sel(filter_dict)

    df: pd.DataFrame = filt.to_dataframe()

    # Create labels for the axes
    tq = "theoretical quantiles"
    rq = f"{response_var} quantiles"

    # Filter columns needed for plotting
    columns = [("Z", response_var)]
    columns.extend([("batch_effects", be.item()) for be in data.batch_effect_dims])
    df = df[columns]
    df.columns = [rq] + [be.item() for be in data.batch_effect_dims]

    # Sort the dataframe by the response variable
    df.sort_values(by=rq, inplace=True)

    # Create a column for the theoretical quantiles
    rand = np.random.randn(df.shape[0])
    rand.sort()
    df[tq] = rand

    if split_data:
        for i, g in enumerate(df.groupby(split_data, sort=False)):
            my_offset = i * 1.0
            my_id = g[1].index
            df.loc[my_id, rq] += i * 1.0
            rand = np.random.randn(g[1].shape[0])
            rand.sort()
            df.loc[my_id, tq] = rand

    # Plot the QQ-plot
    sns.scatterplot(
        data=df,
        x="theoretical quantiles",
        y=rq,
        hue=hue_data if hue_data in df else None,
        style=markers_data if markers_data in df else None,
        **plt_kwargs,
        linewidth=0,
    )
    if plot_id_line:
        if split_data:
            for i, g in enumerate(df.groupby(split_data, sort=False)):
                my_offset = i * 1.0
                my_id = g[1].index
                my_df = df.loc[my_id, rq]
                max_abs_val = max(abs(my_df.min()), abs(my_df.max())) + 0.5
                plt.plot(
                    [-max_abs_val, max_abs_val],
                    [-max_abs_val + my_offset, max_abs_val + my_offset],
                    color="black",
                    linestyle="--",
                )
        else:
            max_abs_val = max(abs(df[rq].min()), abs(df[rq].max())) + 0.5
            plt.plot(
                [-max_abs_val, max_abs_val],
                [-max_abs_val, max_abs_val],
                color="black",
                linestyle="--",
            )

    if bound != 0:
        plt.axis((-bound, bound, -bound, bound))
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"qq_{response_var}_{data.name}.png"))
    else:
        plt.show(block=False)
    plt.close()
