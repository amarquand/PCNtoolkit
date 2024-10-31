"""A module for plotting functions."""

# TODO move all plotting functions to this file

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore
from matplotlib.font_manager import FontProperties

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase


def plot_centiles(
    model: NormBase,
    data: NormData,
    covariate: str | None = None,
    cummul_densities: list | None = None,
    batch_effects: Dict[str, List[str]] | None = None,
    show_data: bool = False,
    plt_kwargs: dict | None = None,
    hue_data: str = "site",
    markers_data: str = "sex",
) -> None:
    """Plot centiles for all response variables in the data.

    Args:
        model (NormBase): The normative model.
        data (NormData): Data containing the covariates and response variables.
        covariate (str | None, optional): Name of the covariate on the x-axis. Defaults to None.
        cummul_densities (list | None, optional): Which CDF values correspond to the centiles. Defaults to None.
        batch_effects (Dict[str, List[str]] | None, optional):
            Models with a random effect have different centiles for different batch effects. This parameter allows
            you to specify for which batch effects to plot the centiles, by providing a dictionary with the batch
            effect name as key and a list of batch effect values as value. The first values in the lists will be
            used for computing the centiles. If no list is provided, the batch effect that occurs first in the
            data will be used. Addtionally, if `show_data==True`, the dictionary values specify which batch effects
            are highlighted in the scatterplot.
        show_data (bool, optional): Scatter data along with centiles. Defaults to False.
        plt_kwargs (dict | None, optional): Additional kwargs to pt. Defaults to None.
        hue_data (str, optional): Column to use for coloring. Defaults to "site".
        markers_data (str, optional): Column to use for marker styling. Defaults to "sex".

    Raises:
        ValueError: _description_
    """
    if covariate is None:
        covariate = data.covariates[0].to_numpy().item()
        assert isinstance(covariate, str)

    if batch_effects is None:
        if model.has_random_effect:
            batch_effects = data.get_single_batch_effect()
        else:
            batch_effects = {}

    # Ensure that the batch effects are in the correct format
    if batch_effects:
        for k, v in batch_effects.items():
            if isinstance(v, str):
                batch_effects[k] = [v]
            elif not isinstance(v, list):
                raise ValueError(
                    f"Items of the batch_effect dict be a list or a string, not {type(v)}"
                )

    if plt_kwargs is None:
        plt_kwargs = {}
    palette = plt_kwargs.pop("cmap", "viridis")
    synth_data = data.create_synthetic_data(
        n_datapoints=150,
        range_dim=covariate,
        batch_effects_to_sample={k: [v[0]] for k, v in batch_effects.items()}
        if batch_effects
        else None,
    )
    model.compute_centiles(synth_data, cdf=cummul_densities)
    for response_var in data.coords["response_vars"].to_numpy():
        _plot_centiles(
            data=data,
            synth_data=synth_data,
            response_var=response_var,
            covariate=covariate,
            batch_effects=batch_effects,
            show_data=show_data,
            plt_kwargs=plt_kwargs,
            hue_data=hue_data,
            markers_data=markers_data,
            palette=palette,
        )


def _plot_centiles(
    data: NormData,
    synth_data: NormData,
    response_var: str,
    batch_effects: Dict[str, List[str]],
    covariate: str = None,  # type: ignore
    show_data: bool = False,
    plt_kwargs: dict = None,  # type: ignore
    hue_data: str = "site",
    markers_data: str = "sex",
    palette: str = "viridis",
) -> None:
    """Plot the centiles for a single response variable."""

    sns.set_style("whitegrid")
    plt.figure(**plt_kwargs)
    cmap = plt.get_cmap(palette)

    filter_dict = {
        "covariates": covariate,
        "response_vars": response_var,
    }
    filtered = synth_data.sel(filter_dict)

    for cdf in synth_data.coords["cdf"][::-1]:
        d_mean = abs(cdf - 0.5)
        if d_mean < 0.25:
            style = "-"
        elif d_mean < 0.475:
            style = "--"
        else:
            style = ":"

        sns.lineplot(
            x=filtered.X,
            y=filtered.centiles.sel(cdf=cdf),
            color=cmap(cdf),
            linestyle=style,
            linewidth=1,
            zorder=2,
            legend="brief",
        )
        color = cmap(cdf)
        font = FontProperties()
        font.set_weight("bold")
        plt.text(
            s=cdf.item(),
            x=filtered.X[0] - 1,
            y=filtered.centiles.sel(cdf=cdf)[0],
            color=color,
            horizontalalignment="right",
            verticalalignment="center",
            fontproperties=font,
        )
        plt.text(
            s=cdf.item(),
            x=filtered.X[-1] + 1,
            y=filtered.centiles.sel(cdf=cdf)[-1],
            color=color,
            horizontalalignment="left",
            verticalalignment="center",
            fontproperties=font,
        )

    minx, maxx = plt.xlim()
    plt.xlim(minx - 0.1 * (maxx - minx), maxx + 0.1 * (maxx - minx))

    if show_data:
        df = data.sel(filter_dict).to_dataframe()
        columns = [("X", covariate), ("y", response_var)]
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
                alpha=0.7,
                zorder=1,
            )
            non_be_df = df[~idx]
            sns.scatterplot(
                data=non_be_df,
                x=covariate,
                y=response_var,
                color="black",
                s=20,
                alpha=0.4,
                zorder=0,
            )
            legend = scatter.get_legend()
            if legend:
                handles = legend.legend_handles
                labels = [t.get_text() for t in legend.get_texts()]
                plt.legend(
                    handles,
                    labels,
                    title=data.name + " data",
                    title_fontsize=10,
                )
    plt.title(f"Centiles of {response_var}")
    plt.xlabel(covariate)
    plt.ylabel(response_var)

    plt.show()
