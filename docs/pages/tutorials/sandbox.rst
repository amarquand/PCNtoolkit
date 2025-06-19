.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    from pcntoolkit import BLR, BsplineBasisFunction, LinearBasisFunction, NormativeModel, NormData, load_fcon1000

.. code:: ipython3

    import os
    from typing import Any, Dict, List, Literal
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.font_manager import FontProperties
    
    from pcntoolkit import NormativeModel, NormData
    
    
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
        palette = plt_kwargs.pop("cmap", "viridis")
    
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
                palette=palette,
                save_dir=save_dir,
                show_centile_labels=show_centile_labels,
                show_legend=show_legend,
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
        palette: str = "viridis",
        save_dir: str | None = None,
        show_centile_labels: bool = True,
        show_legend: bool = True,
        plt_kwargs: dict = None,  # type: ignore
    ) -> None:
        sns.set_style("whitegrid")
        plt.figure(**plt_kwargs)
        cmap = plt.get_cmap(palette)
    
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
                # color=cmap(centile),
                color="black",
                linestyle=style,
                linewidth=thickness,
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
                    s=30,
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
        else:
            plotname = f"centiles_{response_var}"
    
        plt.title(title)
        plt.xlabel(covariate)
        plt.ylabel(response_var)
        plt.ylim(500, 2500)
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{plotname}.png"), dpi=300)
        else:
            plt.show(block=False)
        plt.close()


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



.. parsed-literal::

    Process: 10589 - 2025-06-18 15:35:38 - Dataset "fcon1000" created.
        - 1078 observations
        - 1078 unique subjects
        - 1 covariates
        - 217 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        


.. code:: ipython3

    model = NormativeModel.load("/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/hbr/save_dir")

.. code:: ipython3

    plot_centiles(
        model,
        covariate_range=(10, 80),
        covariate="age",  # Which covariate to plot on the x-axis
        show_centile_labels=True,
        # harmonize_data=True,  # Set this to False to see the difference
        show_legend=False,  # Don't show the legend because it crowds the plot
        save_dir="/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/plots_for_presentation/1",
    )


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:216: UserWarning: Process: 10589 - 2025-06-18 15:38:04 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:04 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 10589 - 2025-06-18 15:38:04 - Computing centiles for 4 response variables.
    Process: 10589 - 2025-06-18 15:38:04 - Computing centiles for CortexVol.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:05 - Computing centiles for Right-Lateral-Ventricle.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:06 - Computing centiles for WM-hypointensities.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:07 - Computing centiles for Right-Amygdala.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. code:: ipython3

    # synthetic = model.synthesize(n_samples=5000, covariate_range_per_batch_effect=True)  # <- will fill in the missing Y data
    # synthetic.name = "fcon1000"
    plot_centiles(
        model,
        covariate_range=(10, 80),
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=norm_data,  # Scatter the train data points
        # batch_effects="all",  # You can set this to "all" to show all batch effects
        show_other_data=True,  # Show data points that do not match any batch effects
        show_centile_labels=True,
        harmonize_data=False,
        # harmonize_data=True,  # Set this to False to see the difference
        show_legend=False,  # Don't show the legend because it crowds the plot
        save_dir="/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/plots_for_presentation/2",
    )


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:216: UserWarning: Process: 10589 - 2025-06-18 15:38:09 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:09 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 10589 - 2025-06-18 15:38:09 - Computing centiles for 4 response variables.
    Process: 10589 - 2025-06-18 15:38:09 - Computing centiles for CortexVol.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:10 - Computing centiles for Right-Lateral-Ventricle.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:11 - Computing centiles for WM-hypointensities.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:13 - Computing centiles for Right-Amygdala.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    /var/folders/m8/vtbcb7c96ms3mbjny3b70h3w0000gp/T/ipykernel_10589/920352495.py:282: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /var/folders/m8/vtbcb7c96ms3mbjny3b70h3w0000gp/T/ipykernel_10589/920352495.py:282: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /var/folders/m8/vtbcb7c96ms3mbjny3b70h3w0000gp/T/ipykernel_10589/920352495.py:282: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /var/folders/m8/vtbcb7c96ms3mbjny3b70h3w0000gp/T/ipykernel_10589/920352495.py:282: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)


.. code:: ipython3

    plot_centiles(
        model,
        covariate_range=(10, 80),
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=norm_data,  # Scatter the train data points
        batch_effects="all",  # You can set this to "all" to show all batch effects
        show_other_data=True,  # Show data points that do not match any batch effects
        show_centile_labels=True,
        harmonize_data=False,
        # harmonize_data=True,  # Set this to False to see the difference
        show_legend=False,  # Don't show the legend because it crowds the plot
        save_dir="/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/plots_for_presentation/3",
    )


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:216: UserWarning: Process: 10589 - 2025-06-18 15:38:14 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:14 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 10589 - 2025-06-18 15:38:14 - Computing centiles for 4 response variables.
    Process: 10589 - 2025-06-18 15:38:14 - Computing centiles for CortexVol.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:16 - Computing centiles for Right-Lateral-Ventricle.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:17 - Computing centiles for WM-hypointensities.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:18 - Computing centiles for Right-Amygdala.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. code:: ipython3

    plot_centiles(
        model,
        covariate_range=(10, 80),
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=norm_data,  # Scatter the train data points
        batch_effects="all",  # You can set this to "all" to show all batch effects
        show_other_data=True,  # Show data points that do not match any batch effects
        show_centile_labels=True,
        harmonize_data=True,
        # harmonize_data=True,  # Set this to False to see the difference
        show_legend=False,  # Don't show the legend because it crowds the plot
        save_dir="/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/plots_for_presentation/4",
    )


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:216: UserWarning: Process: 10589 - 2025-06-18 15:38:20 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:38:20 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 10589 - 2025-06-18 15:38:20 - Computing centiles for 4 response variables.
    Process: 10589 - 2025-06-18 15:38:20 - Computing centiles for CortexVol.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. code:: ipython3

    norm_data




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 242kB
    Dimensions:            (observations: 1078, response_vars: 4, covariates: 1,
                            batch_effect_dims: 2)
    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1074 1075 1076 1077
      * response_vars      (response_vars) &lt;U34 544B &#x27;WM-hypointensities&#x27; ... &#x27;Co...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
    Data variables:
        subjects           (observations) object 9kB &#x27;AnnArbor_a_sub04111&#x27; ... &#x27;S...
        Y                  (observations, response_vars) float64 34kB 1.687e+03 ....
        X                  (observations, covariates) float64 9kB 25.63 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 147kB &#x27;M&#x27; ... &#x27;...
        Y_harmonized       (observations, response_vars) float64 34kB 1.521e+03 ....
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000
        unique_batch_effects:           {&#x27;sex&#x27;: [&#x27;F&#x27;, &#x27;M&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;...
        batch_effect_counts:            {&#x27;sex&#x27;: {&#x27;F&#x27;: 589, &#x27;M&#x27;: 489}, &#x27;site&#x27;: {&#x27;A...
        batch_effect_covariate_ranges:  {&#x27;sex&#x27;: {&#x27;F&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;...
        covariate_ranges:               {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-af76eddd-ce92-424b-934b-55cebde37fe2' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-af76eddd-ce92-424b-934b-55cebde37fe2' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 1078</li><li><span class='xr-has-index'>response_vars</span>: 4</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-9712b3ff-1c5f-484e-95a6-a3fc3ced6f88' class='xr-section-summary-in' type='checkbox'  checked><label for='section-9712b3ff-1c5f-484e-95a6-a3fc3ced6f88' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1074 1075 1076 1077</div><input id='attrs-ea6a900a-c40c-4444-a9d4-27f1bd6f3c3a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ea6a900a-c40c-4444-a9d4-27f1bd6f3c3a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fef25324-c786-4590-9116-fe95c7e23121' class='xr-var-data-in' type='checkbox'><label for='data-fef25324-c786-4590-9116-fe95c7e23121' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1075, 1076, 1077])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U34</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Cortex...</div><input id='attrs-6e084903-66c2-4764-a0b3-2f236dce56cf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6e084903-66c2-4764-a0b3-2f236dce56cf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3e55ebf0-e70a-439c-bd54-5bcc166f9f8e' class='xr-var-data-in' type='checkbox'><label for='data-3e55ebf0-e70a-439c-bd54-5bcc166f9f8e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;], dtype=&#x27;&lt;U34&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-fc89ad56-7e29-4aa7-b17c-5727ab177d72' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fc89ad56-7e29-4aa7-b17c-5727ab177d72' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-47aa44c2-e958-4c6b-99e5-484331d54222' class='xr-var-data-in' type='checkbox'><label for='data-47aa44c2-e958-4c6b-99e5-484331d54222' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-eb2417b9-b2d8-4638-9d00-682f1ae8143a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-eb2417b9-b2d8-4638-9d00-682f1ae8143a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a3dcaf42-f53b-4600-9352-561181f2683b' class='xr-var-data-in' type='checkbox'><label for='data-a3dcaf42-f53b-4600-9352-561181f2683b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-a7423c54-ab0a-4df0-900c-7922929d4972' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a7423c54-ab0a-4df0-900c-7922929d4972' class='xr-section-summary' >Data variables: <span>(5)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subjects</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;AnnArbor_a_sub04111&#x27; ... &#x27;Saint...</div><input id='attrs-8036840a-ada8-47cf-89d9-242028cb7147' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8036840a-ada8-47cf-89d9-242028cb7147' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-31571828-8f5c-4a41-852b-22554e28e228' class='xr-var-data-in' type='checkbox'><label for='data-31571828-8f5c-4a41-852b-22554e28e228' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;AnnArbor_a_sub04111&#x27;, &#x27;AnnArbor_a_sub04619&#x27;,
           &#x27;AnnArbor_a_sub13636&#x27;, ..., &#x27;SaintLouis_sub95967&#x27;,
           &#x27;SaintLouis_sub97935&#x27;, &#x27;SaintLouis_sub99965&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 3.906e+03 ... 4.638e+05</div><input id='attrs-b31fe7af-11b8-43a6-80db-401919771421' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b31fe7af-11b8-43a6-80db-401919771421' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6d0b5ef5-2445-441e-b6db-b666336bfeea' class='xr-var-data-in' type='checkbox'><label for='data-6d0b5ef5-2445-441e-b6db-b666336bfeea' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1.68670000e+03, 3.90590000e+03, 1.24160000e+03, 4.46861118e+05],
           [1.37110000e+03, 9.50330000e+03, 1.47980000e+03, 5.32003625e+05],
           [1.41480000e+03, 9.70240000e+03, 1.68280000e+03, 4.75051320e+05],
           ...,
           [1.06100000e+03, 9.09200000e+03, 1.81310000e+03, 5.36279364e+05],
           [4.48300000e+02, 4.55260000e+03, 1.53860000e+03, 4.61674842e+05],
           [5.09100000e+02, 3.38090000e+03, 1.60610000e+03, 4.63793416e+05]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 27.0 29.0 23.0</div><input id='attrs-2b3f2816-99fa-4c7b-b8ee-fb2e5df1afd1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2b3f2816-99fa-4c7b-b8ee-fb2e5df1afd1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a086fb63-bd0f-4b26-8a01-f0e53641b837' class='xr-var-data-in' type='checkbox'><label for='data-a086fb63-bd0f-4b26-8a01-f0e53641b837' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           ...,
           [27.  ],
           [29.  ],
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;M&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;SaintLouis&#x27;</div><input id='attrs-0efcbd2b-05ee-499f-aa22-bc6b44cb1623' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0efcbd2b-05ee-499f-aa22-bc6b44cb1623' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d7eccac1-c54b-44c7-82aa-cfd2bb6d43f2' class='xr-var-data-in' type='checkbox'><label for='data-d7eccac1-c54b-44c7-82aa-cfd2bb6d43f2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;M&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;M&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;M&#x27;, &#x27;AnnArbor_a&#x27;],
           ...,
           [&#x27;M&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;F&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;F&#x27;, &#x27;SaintLouis&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y_harmonized</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.521e+03 2.612e+03 ... 4.226e+05</div><input id='attrs-0bd843f3-d4de-4eb0-8b2f-fe1637a4b5a9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0bd843f3-d4de-4eb0-8b2f-fe1637a4b5a9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a8dd4777-e274-429b-9e67-5ff9a63cbd8f' class='xr-var-data-in' type='checkbox'><label for='data-a8dd4777-e274-429b-9e67-5ff9a63cbd8f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[  1521.35329944,   2611.96005006,   1008.6890209 ,
            395430.33607247],
           [  1204.36841761,   8220.75190661,   1247.13984774,
            480676.92524313],
           [  1248.72281731,   8418.46084963,   1450.64051811,
            423660.57287229],
           ...,
           [  1773.87694028,   7723.69790864,   1368.3129986 ,
            443755.63307374],
           [  1325.6579684 ,   4466.40498074,   1325.99813402,
            420456.24624241],
           [  1386.94954869,   3296.0924113 ,   1393.76175262,
            422604.22492441]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-96f3a5f5-dc60-4a40-b41d-c5b98c595834' class='xr-section-summary-in' type='checkbox'  ><label for='section-96f3a5f5-dc60-4a40-b41d-c5b98c595834' class='xr-section-summary' >Indexes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-b7682b89-254d-4ec7-9943-ca809cea15a1' class='xr-index-data-in' type='checkbox'/><label for='index-b7682b89-254d-4ec7-9943-ca809cea15a1' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
           ...
           1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077],
          dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;, length=1078))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-91fd51fb-9a8d-4cc9-9b9e-7ad5bbe18ce5' class='xr-index-data-in' type='checkbox'/><label for='index-91fd51fb-9a8d-4cc9-9b9e-7ad5bbe18ce5' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-798e5be2-c552-4449-b85b-c54bfec613ac' class='xr-index-data-in' type='checkbox'/><label for='index-798e5be2-c552-4449-b85b-c54bfec613ac' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-df077f5a-b147-47e2-b241-8da7825abf4d' class='xr-index-data-in' type='checkbox'/><label for='index-df077f5a-b147-47e2-b241-8da7825abf4d' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-8ef224e1-74af-4b30-8c34-be691fc685d1' class='xr-section-summary-in' type='checkbox'  checked><label for='section-8ef224e1-74af-4b30-8c34-be691fc685d1' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{&#x27;sex&#x27;: [&#x27;F&#x27;, &#x27;M&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;F&#x27;: 589, &#x27;M&#x27;: 489}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;F&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}, &#x27;M&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd><dt><span>covariate_ranges :</span></dt><dd>{&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd></dl></div></li></ul></div></div>



.. code:: ipython3

    plot_centiles(
        model,
        covariate_range=(10, 80),
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=norm_data,  # Scatter the train data points
        batch_effects={"sex": ["M", "F"], "site": ["AnnArbor_a"]},  # You can set this to "all" to show all batch effects
        show_other_data=False,  # Show data points that do not match any batch effects
        show_centile_labels=True,
        harmonize_data=True,
        # harmonize_data=True,  # Set this to False to see the difference
        show_legend=False,  # Don't show the legend because it crowds the plot
        save_dir="/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/plots_for_presentation/5",
    )


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:216: UserWarning: Process: 10589 - 2025-06-18 15:37:36 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:36 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 10589 - 2025-06-18 15:37:36 - Computing centiles for 4 response variables.
    Process: 10589 - 2025-06-18 15:37:36 - Computing centiles for CortexVol.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:37 - Computing centiles for Right-Lateral-Ventricle.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:38 - Computing centiles for WM-hypointensities.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:39 - Computing centiles for Right-Amygdala.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:41 - Harmonizing data on 4 response variables.
    Process: 10589 - 2025-06-18 15:37:41 - Harmonizing data for Right-Amygdala.


.. parsed-literal::

    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:42 - Harmonizing data for Right-Lateral-Ventricle.


.. parsed-literal::

    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:43 - Harmonizing data for WM-hypointensities.


.. parsed-literal::

    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:43 - Harmonizing data for CortexVol.


.. parsed-literal::

    Sampling: []


.. code:: ipython3

    plot_centiles(
        model,
        covariate_range=(10, 80),
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=norm_data,  # Scatter the train data points
        batch_effects={
            "sex": ["M", "F"],
            "site": [
                "AnnArbor_a",
                "Beijing_Zang",
            ],
        },  # You can set this to "all" to show all batch effects
        show_other_data=False,  # Show data points that do not match any batch effects
        show_centile_labels=True,
        harmonize_data=True,
        # harmonize_data=True,  # Set this to False to see the difference
        show_legend=False,  # Don't show the legend because it crowds the plot
        save_dir="/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/plots_for_presentation/6",
    )


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:216: UserWarning: Process: 10589 - 2025-06-18 15:37:12 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:12 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 10589 - 2025-06-18 15:37:12 - Computing centiles for 4 response variables.
    Process: 10589 - 2025-06-18 15:37:12 - Computing centiles for CortexVol.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:13 - Computing centiles for Right-Lateral-Ventricle.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:14 - Computing centiles for WM-hypointensities.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:16 - Computing centiles for Right-Amygdala.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:17 - Harmonizing data on 4 response variables.
    Process: 10589 - 2025-06-18 15:37:17 - Harmonizing data for Right-Amygdala.


.. parsed-literal::

    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:18 - Harmonizing data for Right-Lateral-Ventricle.


.. parsed-literal::

    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:19 - Harmonizing data for WM-hypointensities.


.. parsed-literal::

    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:37:20 - Harmonizing data for CortexVol.


.. parsed-literal::

    Sampling: []


.. code:: ipython3

    plot_centiles(
        model,
        covariate_range=(10, 80),
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=norm_data,  # Scatter the train data points
        batch_effects={
            "sex": ["M", "F"],
            "site": [
                "AnnArbor_a",
                "Beijing_Zang",
                "Cambridge_Buckner",
            ],
        },  # You can set this to "all" to show all batch effects
        show_other_data=False,  # Show data points that do not match any batch effects
        show_centile_labels=True,
        harmonize_data=True,
        # harmonize_data=True,  # Set this to False to see the difference
        show_legend=False,  # Don't show the legend because it crowds the plot
        save_dir="/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/plots_for_presentation/7",
    )


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:216: UserWarning: Process: 10589 - 2025-06-18 15:36:22 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:22 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 10589 - 2025-06-18 15:36:22 - Computing centiles for 4 response variables.
    Process: 10589 - 2025-06-18 15:36:22 - Computing centiles for CortexVol.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:23 - Computing centiles for Right-Lateral-Ventricle.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:25 - Computing centiles for WM-hypointensities.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:26 - Computing centiles for Right-Amygdala.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:27 - Harmonizing data on 4 response variables.
    Process: 10589 - 2025-06-18 15:36:27 - Harmonizing data for Right-Amygdala.


.. parsed-literal::

    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:28 - Harmonizing data for Right-Lateral-Ventricle.


.. parsed-literal::

    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:29 - Harmonizing data for WM-hypointensities.


.. parsed-literal::

    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:30 - Harmonizing data for CortexVol.


.. parsed-literal::

    Sampling: []


.. code:: ipython3

    plot_centiles(
        model,
        covariate_range=(10, 80),
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=norm_data,  # Scatter the train data points
        batch_effects={
            "sex": ["M", "F"],
            "site": [
                "AnnArbor_a",
                "Beijing_Zang",
                "Cambridge_Buckner",
                "Milwaukee_b",
            ],
        },  # You can set this to "all" to show all batch effects
        show_other_data=False,  # Show data points that do not match any batch effects
        show_centile_labels=True,
        harmonize_data=True,
        # harmonize_data=True,  # Set this to False to see the difference
        show_legend=False,  # Don't show the legend because it crowds the plot
        save_dir="/Users/stijndeboer/Projects/PCN/PCNtoolkit/examples/resources/plots_for_presentation/8",
    )


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:216: UserWarning: Process: 10589 - 2025-06-18 15:36:31 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:31 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 10589 - 2025-06-18 15:36:31 - Computing centiles for 4 response variables.
    Process: 10589 - 2025-06-18 15:36:31 - Computing centiles for CortexVol.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:32 - Computing centiles for Right-Lateral-Ventricle.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:34 - Computing centiles for WM-hypointensities.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:35 - Computing centiles for Right-Amygdala.


.. parsed-literal::

    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:37 - Harmonizing data on 4 response variables.
    Process: 10589 - 2025-06-18 15:36:37 - Harmonizing data for Right-Amygdala.


.. parsed-literal::

    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:37 - Harmonizing data for Right-Lateral-Ventricle.


.. parsed-literal::

    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:38 - Harmonizing data for WM-hypointensities.


.. parsed-literal::

    Sampling: []
    Sampling: []


.. parsed-literal::

    Process: 10589 - 2025-06-18 15:36:39 - Harmonizing data for CortexVol.


.. parsed-literal::

    Sampling: []



