"""A module for plotting functions."""

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

sns.set_theme(style="darkgrid")


def plot_centiles(
    model: "NormativeModel",
    centiles: List[float] | np.ndarray | None = None,
    covariate: str | None = None,
    covariate_range: tuple[float, float] = (None, None),  # type: ignore
    batch_effects: Dict[str, List[str]] | None | Literal["all"] = None,
    scatter_data: NormData | None = None,
    harmonize_data: bool = True,
    hue_data: str = "site",
    markers_data: str = "sex",
    show_other_data: bool = False,
    show_thrivelines: bool = False,
    z_thrive: float = 0.0,
    save_dir: str | None = None,
    show_centile_labels: bool = True,
    show_legend: bool = True,
    show_yhat: bool = False,
    plt_kwargs: dict | None = None,
    **kwargs: Any,
) -> None:
    """Generate centile plots for response variables with optional data overlay.

    This function creates visualization of centile curves for all response variables
    in the dataset. It can optionally show the actual data points overlaid on the
    centile curves, with customizable styling based on categorical variables.

    Parameters
    ----------
    model: NormativeModel
        The model to plot the centiles for.
    centiles: List[float] | np.ndarray | None, optional
        The centiles to plot. If None, the default centiles will be used.
    covariate: str | None, optional
        The covariate to plot on the x-axis. If None, the first covariate in the model will be used.
    covariate_range: tuple[float, float], optional
        The range of the covariate to plot on the x-axis. If None, the range of the covariate that was in the train data will be used.
    batch_effects: Dict[str, List[str]] | None | Literal["all"], optional
        The batch effects to plot the centiles for. If None, the batch effect that appears first in alphabetical order will be used.
    scatter_data: NormData | None, optional
        Data to scatter on top of the centiles.
    harmonize_data: bool, optional
        Whether to harmonize the scatter data before plotting. Data will be harmonized to the batch effect for which the centiles were computed.
    hue_data: str, optional
        The column to use for color coding the data. If None, the data will not be color coded.
    markers_data: str, optional
        The column to use for marker styling the data. If None, the data will not be marker styled.
    show_other_data: bool, optional
        Whether to scatter data belonging to groups not in batch_effects.
    save_dir: str | None, optional
        The directory to save the plot to. If None, the plot will not be saved.
    show_centile_labels: bool, optional
        Whether to show the centile labels on the plot.
    show_legend: bool, optional
        Whether to show the legend on the plot.
    plt_kwargs: dict, optional
        Additional keyword arguments for the plot.
    **kwargs: Any, optional
        Additional keyword arguments for the model.compute_centiles method.

    Returns
    -------
    None
        Displays the plot using matplotlib.
    """
    if covariate is None:
        covariate = model.covariates[0]
        assert isinstance(covariate, str)

    cov_min = covariate_range[0] or model.covariate_ranges[covariate]["min"]
    cov_max = covariate_range[1] or model.covariate_ranges[covariate]["max"]
    covariate_range = (cov_min, cov_max)

    if batch_effects == "all":
        batch_effects = model.unique_batch_effects
    elif batch_effects is None:
        batch_effects = {k: [v[0]] for k, v in model.unique_batch_effects.items()}

    if plt_kwargs is None:
        plt_kwargs = {}

    # Create some synthetic data with a single batch effect
    # The plotted covariate is just a linspace
    centile_covariates = np.linspace(covariate_range[0], covariate_range[1], 150)
    centile_df = pd.DataFrame({covariate: centile_covariates})

    # TODO: use the mean here
    # Any other covariates are taken to be the midpoint between the observed min and max
    for cov in model.covariates:
        if cov != covariate:
            minc = model.covariate_ranges[cov]["min"]
            maxc = model.covariate_ranges[cov]["max"]
            centile_df[cov] = (minc + maxc) / 2

    # Batch effects are the first ones in the highlighted batch effects
    for be, v in batch_effects.items():
        centile_df[be] = v[0]
    # Response vars are all 0, we don't need them
    for rv in model.response_vars:
        centile_df[rv] = 0
    centile_data = NormData.from_dataframe(
        "centile",
        dataframe=centile_df,
        covariates=model.covariates,
        response_vars=model.response_vars,
        batch_effects=list(batch_effects.keys()),
    )  # type:ignore

    if not hasattr(centile_data, "centiles"):
        model.compute_centiles(centile_data, centiles=centiles, **kwargs)
    if scatter_data and show_thrivelines:
        model.compute_thrivelines(scatter_data, z_thrive=z_thrive)
    if show_yhat and not hasattr(centile_data, "yhat"):
        model.compute_yhat(centile_data)

    if not model.has_batch_effect:
        batch_effects = {}

    if harmonize_data and scatter_data:
        if model.has_batch_effect:
            reference_batch_effect = {k: v[0] for k, v in batch_effects.items()}
            model.harmonize(scatter_data, reference_batch_effect=reference_batch_effect)
        else:
            model.harmonize(scatter_data)

    for response_var in model.response_vars:
        _plot_centiles(
            centile_data=centile_data,
            response_var=response_var,
            covariate=covariate,
            covariate_range=covariate_range,
            batch_effects=batch_effects,
            scatter_data=scatter_data,
            harmonize_data=harmonize_data,
            hue_data=hue_data,
            markers_data=markers_data,
            show_other_data=show_other_data,
            show_thrivelines=show_thrivelines,
            save_dir=save_dir,
            show_centile_labels=show_centile_labels,
            show_legend=show_legend,
            show_yhat=show_yhat,
            plt_kwargs=plt_kwargs,
        )


def _plot_centiles(
    centile_data: NormData,
    response_var: str,
    covariate: str = None,  # type: ignore
    covariate_range: tuple[float, float] = (None, None),  # type: ignore
    batch_effects: Dict[str, List[str]] = None,  # type: ignore
    scatter_data: NormData | None = None,
    harmonize_data: bool = True,
    hue_data: str = "site",
    markers_data: str = "sex",
    show_other_data: bool = False,
    show_thrivelines: bool = False,
    save_dir: str | None = None,
    show_centile_labels: bool = True,
    show_legend: bool = True,
    show_yhat: bool = False,
    plt_kwargs: dict = None,  # type: ignore
) -> None:
    sns.set_style("whitegrid")
    plt.figure(**plt_kwargs)

    filter_dict = {
        "covariates": covariate,
        "response_vars": response_var,
    }

    filtered = centile_data.sel(filter_dict)

    for centile in centile_data.coords["centile"][::-1]:
        d_mean = abs(centile - 0.5)
        if d_mean == 0:
            thickness = 2
        else:
            thickness = 1
        if d_mean <= 0.25:
            style = "-"

        elif d_mean <= 0.475:
            style = "--"
        else:
            style = ":"

        sns.lineplot(
            x=filtered.X,
            y=filtered.centiles.sel(centile=centile),
            color="black",
            linestyle=style,
            linewidth=thickness,
            zorder=2,
            legend="brief",
        )

        font = FontProperties()
        font.set_weight("bold")
        if show_centile_labels:
            plt.text(
                s=centile.item(),
                x=filtered.X[0] - 1,
                y=filtered.centiles.sel(centile=centile)[0],
                color="black",
                horizontalalignment="right",
                verticalalignment="center",
                fontproperties=font,
            )
            plt.text(
                s=centile.item(),
                x=filtered.X[-1] + 1,
                y=filtered.centiles.sel(centile=centile)[-1],
                color="black",
                horizontalalignment="left",
                verticalalignment="center",
                fontproperties=font,
            )
    if show_yhat:
        plt.plot(filtered.X, filtered.Yhat, color="red", linestyle="--", linewidth=thickness, zorder=2, label="$\\hat{Y}$")

    minx, maxx = plt.xlim()
    plt.xlim(minx - 0.1 * (maxx - minx), maxx + 0.1 * (maxx - minx))

    if scatter_data:
        scatter_filter = scatter_data.sel(filter_dict)
        df = scatter_filter.to_dataframe()
        scatter_data_name = "Y_harmonized" if harmonize_data else "Y"
        thriveline_data_name = "thrive_Y_harmonized" if harmonize_data else "thrive_Y"
        columns = [("X", covariate), (scatter_data_name, response_var)]
        columns.extend([("batch_effects", be.item()) for be in scatter_data.batch_effect_dims])
        df = df[columns]
        df.columns = [c[1] for c in df.columns]
        if batch_effects == {}:
            sns.scatterplot(
                df,
                x=covariate,
                y=response_var,
                label=scatter_data.name,
                color="black",
                s=20,
                alpha=0.6,
                zorder=1,
                linewidth=0,
            )
            if show_thrivelines:
                plt.plot(scatter_filter.thrive_X.to_numpy().T, scatter_filter[thriveline_data_name].to_numpy().T)
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
            if show_thrivelines:
                plt.plot(scatter_filter.thrive_X.to_numpy().T, scatter_filter[thriveline_data_name].to_numpy().T)

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
    if scatter_data:
        if harmonize_data:
            plotname = f"centiles_{response_var}_{scatter_data.name}_harmonized"
            title = f"{title}\n With harmonized {scatter_data.name} data"
        else:
            plotname = f"centiles_{response_var}_{scatter_data.name}"
            title = f"{title}\n With raw {scatter_data.name} data"

    plt.title(title)
    plt.xlabel(covariate)
    plt.ylabel(response_var)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{plotname}.png"), dpi=300)
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
                plt.plot(
                    [-3, 3], [-3 + my_offset, 3 + my_offset], color="black", linestyle="--", linewidth=1, alpha=0.8, zorder=0
                )
        else:
            plt.plot([-3, 3], [-3, 3], color="black", linestyle="--", linewidth=1, alpha =0.8, zorder=0)

    if bound != 0:
        plt.axis((-bound, bound, -bound, bound))
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"qq_{response_var}_{data.name}.png"), dpi=300)
    else:
        plt.show(block=False)
    plt.close()


def plot_ridge(data: NormData, variable: Literal["Z", "Y"], split_by: str, save_dir: str | None = None, **kwargs: Any) -> None:
    """
    Plot a ridge plot for each response variable in the data.

    Creates a density plot for the variable split by the split_by variable.

    Each density plot will be on a different row.

    The hue of the density plot will be the split_by variable.

    Parameters
    ----------
    data : NormData
        Data containing the response variable.
    variable : Literal["Z", "Y"]
        The variable to plot on the x-axis. (Z or Y)
    split_by : str
        The variable to split the data by.
    save_dir : str | None, optional
        The directory to save the plot to. Defaults to None.
    **kwargs : Any, optional
        Additional keyword arguments for the plot.

    Returns
    -------
    None
    """

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    for response_var in data.coords["response_vars"].to_numpy():
        _plot_ridge(data, variable, response_var, split_by, save_dir, **kwargs)


def _plot_ridge(data, variable, response_var, split_by, save_dir, **kwargs):
    df = data.to_dataframe()
    # Select only the Z and batch_effects columns
    df = df[[(variable, response_var), ("batch_effects", split_by)]]
    # Join column name levels with an underscore
    df.columns = [df.columns[0][0], df.columns[1][1]]

    # Initialize the FacetGrid object
    palette = kwargs.get("palette", sns.cubehelix_palette(n_colors=len(df[split_by].unique()), rot=1.5, light=0.7))
    g = sns.FacetGrid(df, row=split_by, hue=split_by, aspect=15, height=0.5, palette=palette)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, variable, bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, variable, clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, variable)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"ridge_{response_var}_{variable}_{split_by}_{data.name}.png"), dpi=300)
    else:
        plt.show(block=False)
    plt.close()
