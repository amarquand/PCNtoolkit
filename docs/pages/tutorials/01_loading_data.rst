The NormData class
==================

A key component of the PCNtoolkit is the NormData object. It is a
container for the data that will be used to fit the normative model. The
NormData object keeps track of the all the dimensions of your data, the
features and response variables, batch effects, preprocessing steps, and
more.

.. code:: ipython3

    from pcntoolkit import NormData

Creating a NormData object
--------------------------

There are currently two easy ways to create a NormData object. 1. Load
from a pandas dataframe 2. Load from numpy arrays

Here are examples of both.

.. code:: ipython3

    # Creating a NormData object from a pandas dataframe
    import pandas as pd
    
    # Download an example dataset:
    data = pd.read_csv(
        "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
    )
    
    # specify the column names to use
    covariates = ["age"]
    batch_effects = ["sex", "site"]
    response_vars = ["WM-hypointensities", "Left-Lateral-Ventricle", "Brain-Stem"]
    
    # create a NormData object
    norm_data = NormData.from_dataframe(
        name="fcon1000", dataframe=data, covariates=covariates, batch_effects=batch_effects, response_vars=response_vars
    )
    norm_data.coords


.. parsed-literal::

    Process: 91027 - 2025-06-12 15:20:31 - Dataset "fcon1000" created.
        - 1078 observations
        - 1078 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:216: UserWarning: Process: 91027 - 2025-06-12 15:20:31 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)




.. parsed-literal::

    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1074 1075 1076 1077
      * response_vars      (response_vars) <U22 264B 'WM-hypointensities' ... 'Br...
      * covariates         (covariates) <U3 12B 'age'
      * batch_effect_dims  (batch_effect_dims) <U4 32B 'sex' 'site'



.. code:: ipython3

    # Creating a NormData object from numpy arrays
    import numpy as np
    from pcntoolkit import NormData
    
    # synthesize some data
    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 10)
    batch_effects = np.random.randint(0, 2, 100)
    subject_ids = np.arange(100)
    
    # Create a NormData object
    np_norm_data = NormData.from_ndarrays("fcon1000", X=X, Y=Y, batch_effects=batch_effects, subject_ids=subject_ids)
    np_norm_data.coords


.. parsed-literal::

    Process: 91027 - 2025-06-12 15:20:31 - Dataset "fcon1000" created.
        - 100 observations
        - 100 unique subjects
        - 10 covariates
        - 10 response variables
        - 1 batch effects:
        	batch_effect_0 (2)
        




.. parsed-literal::

    Coordinates:
      * observations       (observations) int64 800B 0 1 2 3 4 5 ... 95 96 97 98 99
      * covariates         (covariates) <U11 440B 'covariate_0' ... 'covariate_9'
      * response_vars      (response_vars) <U14 560B 'response_var_0' ... 'respon...
      * batch_effect_dims  (batch_effect_dims) <U14 56B 'batch_effect_0'



As you can see, it is very simple to create a NormData object.

There is an important difference though: the coordinates of the NormData
object that was created with ``from_dataframe`` have the name of the
column in the dataframe, but the ``from_ndarrays`` method creates
coordinates with generic names. This is why the from_dataframe method is
favorable.

Casting back to a pandas dataframe
----------------------------------

The NormData object can be cast back to a pandas dataframe using the
``to_dataframe`` method. This will return a pandas dataframe with a
columnar multi-index.

.. code:: ipython3

    df = norm_data.to_dataframe()
    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>X</th>
          <th colspan="3" halign="left">Y</th>
          <th colspan="3" halign="left">Y_harmonized</th>
          <th colspan="3" halign="left">Z</th>
          <th>...</th>
          <th colspan="10" halign="left">centiles</th>
        </tr>
        <tr>
          <th></th>
          <th>age</th>
          <th>Brain-Stem</th>
          <th>Left-Lateral-Ventricle</th>
          <th>WM-hypointensities</th>
          <th>Brain-Stem</th>
          <th>Left-Lateral-Ventricle</th>
          <th>WM-hypointensities</th>
          <th>Brain-Stem</th>
          <th>Left-Lateral-Ventricle</th>
          <th>WM-hypointensities</th>
          <th>...</th>
          <th>(Brain-Stem, 0.25)</th>
          <th>(WM-hypointensities, 0.5)</th>
          <th>(Left-Lateral-Ventricle, 0.5)</th>
          <th>(Brain-Stem, 0.5)</th>
          <th>(WM-hypointensities, 0.75)</th>
          <th>(Left-Lateral-Ventricle, 0.75)</th>
          <th>(Brain-Stem, 0.75)</th>
          <th>(WM-hypointensities, 0.95)</th>
          <th>(Left-Lateral-Ventricle, 0.95)</th>
          <th>(Brain-Stem, 0.95)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25.63</td>
          <td>20663.2</td>
          <td>4049.4</td>
          <td>1686.7</td>
          <td>20663.2</td>
          <td>4049.4</td>
          <td>1686.7</td>
          <td>0.019085</td>
          <td>-0.531192</td>
          <td>0.811669</td>
          <td>...</td>
          <td>18887.797687</td>
          <td>1039.652128</td>
          <td>6405.764862</td>
          <td>20614.346287</td>
          <td>1577.343220</td>
          <td>9397.799830</td>
          <td>22340.894887</td>
          <td>2350.899842</td>
          <td>13702.331738</td>
          <td>24824.817574</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.34</td>
          <td>19954.0</td>
          <td>9312.6</td>
          <td>1371.1</td>
          <td>19954.0</td>
          <td>9312.6</td>
          <td>1371.1</td>
          <td>-0.257872</td>
          <td>1.066176</td>
          <td>0.786804</td>
          <td>...</td>
          <td>18887.546534</td>
          <td>743.969366</td>
          <td>4583.764377</td>
          <td>20614.099824</td>
          <td>1281.578935</td>
          <td>7575.345493</td>
          <td>22340.653115</td>
          <td>2055.018273</td>
          <td>11879.224460</td>
          <td>24824.582550</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.20</td>
          <td>21645.2</td>
          <td>8972.6</td>
          <td>1414.8</td>
          <td>21645.2</td>
          <td>8972.6</td>
          <td>1414.8</td>
          <td>0.402664</td>
          <td>0.377463</td>
          <td>0.288927</td>
          <td>...</td>
          <td>18887.918689</td>
          <td>1184.451505</td>
          <td>7298.020243</td>
          <td>20614.466982</td>
          <td>1722.192384</td>
          <td>10290.332331</td>
          <td>22341.015276</td>
          <td>2495.820633</td>
          <td>14595.262920</td>
          <td>24824.937523</td>
        </tr>
        <tr>
          <th>3</th>
          <td>31.39</td>
          <td>20790.6</td>
          <td>6798.6</td>
          <td>1830.6</td>
          <td>20790.6</td>
          <td>6798.6</td>
          <td>1830.6</td>
          <td>0.068779</td>
          <td>-0.235935</td>
          <td>0.699007</td>
          <td>...</td>
          <td>18887.992269</td>
          <td>1273.278014</td>
          <td>7845.370183</td>
          <td>20614.541023</td>
          <td>1811.052643</td>
          <td>10837.870113</td>
          <td>22341.089776</td>
          <td>2584.729447</td>
          <td>15143.070944</td>
          <td>24825.012684</td>
        </tr>
        <tr>
          <th>4</th>
          <td>13.58</td>
          <td>17692.6</td>
          <td>6112.5</td>
          <td>1642.4</td>
          <td>17692.6</td>
          <td>6112.5</td>
          <td>1642.4</td>
          <td>-1.141237</td>
          <td>0.612944</td>
          <td>1.369500</td>
          <td>...</td>
          <td>18887.379599</td>
          <td>550.903529</td>
          <td>3394.090536</td>
          <td>20613.938897</td>
          <td>1088.474467</td>
          <td>6385.456500</td>
          <td>22340.498194</td>
          <td>1861.858226</td>
          <td>10689.025937</td>
          <td>24824.436272</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 28 columns</p>
    </div>



Inspecting the NormData
-----------------------

So let’s go over the attributes of the NormData object. Because it is a
subclass of xarray.Dataset, it has all the attributes of a
xarray.Dataset, but it has some additional attributes that are specific
to normative modelling.

The data variables
~~~~~~~~~~~~~~~~~~

The data variables of the NormData object are: - ``X``: The covariates -
``Y``: The response variables - ``batch_effects``: The batch effects -
``subjects``: The subject ids

And all these data variables are xarray.DataArrays, with corresponding
dimensions, stored in the ``data_vars`` attribute of the NormData
object.

.. code:: ipython3

    norm_data.data_vars




.. parsed-literal::

    Data variables:
        subjects       (observations) int64 9kB 0 1 2 3 4 ... 1074 1075 1076 1077
        Y              (observations, response_vars) float64 26kB 1.687e+03 ... 1...
        X              (observations, covariates) float64 9kB 25.63 18.34 ... 23.0
        batch_effects  (observations, batch_effect_dims) <U17 147kB '1' ... 'Sain...



The coordinates
~~~~~~~~~~~~~~~

Because it is a subclass of xarray.Dataset, the NormData object also
holds all the coordinates of the data, found under the ``coords``
attribute.

The coordinates are: - ``observations``: The index of the observations -
``response_vars``: The names of the response variables - ``covariates``:
The names of the covariates - ``batch_effect_dims``: The names of the
batch effect dimensions

.. code:: ipython3

    norm_data.coords




.. parsed-literal::

    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1074 1075 1076 1077
      * response_vars      (response_vars) <U22 264B 'WM-hypointensities' ... 'Br...
      * covariates         (covariates) <U3 12B 'age'
      * batch_effect_dims  (batch_effect_dims) <U4 32B 'sex' 'site'



Indexing using the coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Xarrays powerful indexing methods can also be used on NormData.

Selecting a response variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For example, to select the data for a specific response variable, you
can use the ``response_vars`` coordinate:

.. code:: python

   norm_data.sel(response_vars="WM-hypointensities")

This will return a new NormData object with only the data for the
response variable “WM-hypointensities”.

.. code:: ipython3

    norm_data.sel(response_vars="WM-hypointensities")




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 181kB
    Dimensions:            (observations: 1078, covariates: 1, batch_effect_dims: 2)
    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1074 1075 1076 1077
        response_vars      &lt;U22 88B &#x27;WM-hypointensities&#x27;
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
    Data variables:
        subjects           (observations) int64 9kB 0 1 2 3 ... 1074 1075 1076 1077
        Y                  (observations) float64 9kB 1.687e+03 1.371e+03 ... 509.1
        X                  (observations, covariates) float64 9kB 25.63 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 147kB &#x27;1&#x27; ... &#x27;...
    Attributes:
        real_ids:                       False
        is_scaled:                      False
        name:                           fcon1000
        unique_batch_effects:           {&#x27;sex&#x27;: [&#x27;0&#x27;, &#x27;1&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;...
        batch_effect_counts:            {&#x27;sex&#x27;: {&#x27;0&#x27;: 589, &#x27;1&#x27;: 489}, &#x27;site&#x27;: {&#x27;A...
        batch_effect_covariate_ranges:  {&#x27;sex&#x27;: {&#x27;0&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;...
        covariate_ranges:               {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-7f65ec49-c736-4bbd-a721-1495c8ec7a3b' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-7f65ec49-c736-4bbd-a721-1495c8ec7a3b' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 1078</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-a9d45832-5fe1-4d90-b361-090a67d3927e' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a9d45832-5fe1-4d90-b361-090a67d3927e' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1074 1075 1076 1077</div><input id='attrs-37547b42-171a-4470-a02c-fcc49ae603cb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-37547b42-171a-4470-a02c-fcc49ae603cb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6fc84c57-a993-4e44-afb6-226040e203d4' class='xr-var-data-in' type='checkbox'><label for='data-6fc84c57-a993-4e44-afb6-226040e203d4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1075, 1076, 1077])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>response_vars</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27;</div><input id='attrs-7e8c6ad9-629b-441c-9ea3-105b6c0dd996' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7e8c6ad9-629b-441c-9ea3-105b6c0dd996' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1493f7d7-3e5a-4658-80f6-50e61bda226f' class='xr-var-data-in' type='checkbox'><label for='data-1493f7d7-3e5a-4658-80f6-50e61bda226f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(&#x27;WM-hypointensities&#x27;, dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-bd5843b1-243b-4355-aee6-28862cf2c2cc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bd5843b1-243b-4355-aee6-28862cf2c2cc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e516a6a8-13b4-4a1a-ae5d-635b3606dd5d' class='xr-var-data-in' type='checkbox'><label for='data-e516a6a8-13b4-4a1a-ae5d-635b3606dd5d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-8c023cc1-943d-4d98-a4da-1cd4b4257377' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8c023cc1-943d-4d98-a4da-1cd4b4257377' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7d6652b5-f5cd-49ec-a969-08630253d642' class='xr-var-data-in' type='checkbox'><label for='data-7d6652b5-f5cd-49ec-a969-08630253d642' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ca318331-3ff4-47b8-803a-98c5dfe451a9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-ca318331-3ff4-47b8-803a-98c5dfe451a9' class='xr-section-summary' >Data variables: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subjects</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1074 1075 1076 1077</div><input id='attrs-da5125c0-0b6c-4f36-90bd-4fac43bfac0d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-da5125c0-0b6c-4f36-90bd-4fac43bfac0d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5940985b-644e-435a-a676-bed5445db949' class='xr-var-data-in' type='checkbox'><label for='data-5940985b-644e-435a-a676-bed5445db949' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1075, 1076, 1077])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 1.371e+03 ... 448.3 509.1</div><input id='attrs-f27e967e-4e17-433d-9410-2d0022eaa758' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f27e967e-4e17-433d-9410-2d0022eaa758' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5e252a01-cd85-4d72-8b11-899aa2d3cfff' class='xr-var-data-in' type='checkbox'><label for='data-5e252a01-cd85-4d72-8b11-899aa2d3cfff' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1686.7, 1371.1, 1414.8, ..., 1061. ,  448.3,  509.1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 27.0 29.0 23.0</div><input id='attrs-f29ca661-9e9c-4c30-a841-d9a49ed7f22d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f29ca661-9e9c-4c30-a841-d9a49ed7f22d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ccb886ad-226e-46c3-af97-f53274b23b66' class='xr-var-data-in' type='checkbox'><label for='data-ccb886ad-226e-46c3-af97-f53274b23b66' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           ...,
           [27.  ],
           [29.  ],
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;1&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;SaintLouis&#x27;</div><input id='attrs-d316cd80-46a2-427b-8281-5c762275ba1e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d316cd80-46a2-427b-8281-5c762275ba1e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-568382dc-a882-4cd2-8481-192191eef193' class='xr-var-data-in' type='checkbox'><label for='data-568382dc-a882-4cd2-8481-192191eef193' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           ...,
           [&#x27;1&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;0&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;0&#x27;, &#x27;SaintLouis&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-77f54b53-5ebe-4805-b131-5dc896766584' class='xr-section-summary-in' type='checkbox'  ><label for='section-77f54b53-5ebe-4805-b131-5dc896766584' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-01fca2a2-0da5-4727-9ef7-c4eebbe3f4af' class='xr-index-data-in' type='checkbox'/><label for='index-01fca2a2-0da5-4727-9ef7-c4eebbe3f4af' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
           ...
           1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077],
          dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;, length=1078))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-5d8ac61b-5bbd-4e0f-b54d-50cb60d99f8c' class='xr-index-data-in' type='checkbox'/><label for='index-5d8ac61b-5bbd-4e0f-b54d-50cb60d99f8c' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-953fe6cc-fa7a-4342-83a0-d4877e3935a2' class='xr-index-data-in' type='checkbox'/><label for='index-953fe6cc-fa7a-4342-83a0-d4877e3935a2' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-c02d460a-a212-4c5a-879b-b4143c0f548f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-c02d460a-a212-4c5a-879b-b4143c0f548f' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{&#x27;sex&#x27;: [&#x27;0&#x27;, &#x27;1&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;0&#x27;: 589, &#x27;1&#x27;: 489}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;0&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}, &#x27;1&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd><dt><span>covariate_ranges :</span></dt><dd>{&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd></dl></div></li></ul></div></div>



Selecting a number of observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

But we can also filter out a slice of the data. For example, to select
the first 10 observations, you can use the ``observations`` coordinate:

.. code:: python

   norm_data.sel(observations=slice(0, 9))

This will return a new NormData object with only the first 10
observations.

.. code:: ipython3

    norm_data.sel(observations=slice(0, 9))




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 2kB
    Dimensions:            (observations: 10, response_vars: 3, covariates: 1,
                            batch_effect_dims: 2)
    Coordinates:
      * observations       (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
      * response_vars      (response_vars) &lt;U22 264B &#x27;WM-hypointensities&#x27; ... &#x27;Br...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
    Data variables:
        subjects           (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
        Y                  (observations, response_vars) float64 240B 1.687e+03 ....
        X                  (observations, covariates) float64 80B 25.63 ... 19.88
        batch_effects      (observations, batch_effect_dims) &lt;U17 1kB &#x27;1&#x27; ... &#x27;An...
    Attributes:
        real_ids:                       False
        is_scaled:                      False
        name:                           fcon1000
        unique_batch_effects:           {&#x27;sex&#x27;: [&#x27;0&#x27;, &#x27;1&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;...
        batch_effect_counts:            {&#x27;sex&#x27;: {&#x27;0&#x27;: 589, &#x27;1&#x27;: 489}, &#x27;site&#x27;: {&#x27;A...
        batch_effect_covariate_ranges:  {&#x27;sex&#x27;: {&#x27;0&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;...
        covariate_ranges:               {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-4ec32b2e-ec2f-4cf1-b85e-af97650f3e56' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-4ec32b2e-ec2f-4cf1-b85e-af97650f3e56' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 10</li><li><span class='xr-has-index'>response_vars</span>: 3</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-c08ef123-4c58-4e9f-a108-ebb0e81b9a2b' class='xr-section-summary-in' type='checkbox'  checked><label for='section-c08ef123-4c58-4e9f-a108-ebb0e81b9a2b' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-f30d300c-5466-4c38-a801-3d4ddcc4fc86' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f30d300c-5466-4c38-a801-3d4ddcc4fc86' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1944c9f3-bf18-4a8f-93d4-9c3b4b930385' class='xr-var-data-in' type='checkbox'><label for='data-1944c9f3-bf18-4a8f-93d4-9c3b4b930385' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Brain-...</div><input id='attrs-25988a45-a03c-4b5b-9f27-1c76237a7e33' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-25988a45-a03c-4b5b-9f27-1c76237a7e33' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1a5c4866-a557-4135-9e6d-89d8c4b082c4' class='xr-var-data-in' type='checkbox'><label for='data-1a5c4866-a557-4135-9e6d-89d8c4b082c4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Left-Lateral-Ventricle&#x27;, &#x27;Brain-Stem&#x27;],
          dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-673bfcfc-75c7-462d-b700-f336cffa3ae7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-673bfcfc-75c7-462d-b700-f336cffa3ae7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8219e26a-db6d-45a4-8dde-5cc5c4966546' class='xr-var-data-in' type='checkbox'><label for='data-8219e26a-db6d-45a4-8dde-5cc5c4966546' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-0acd1e99-3173-482e-863b-58e26a103a78' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0acd1e99-3173-482e-863b-58e26a103a78' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0aeedccb-868b-430a-acfa-0058e8849f2e' class='xr-var-data-in' type='checkbox'><label for='data-0aeedccb-868b-430a-acfa-0058e8849f2e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-460e9924-424b-4ad7-9eab-88ff6453649c' class='xr-section-summary-in' type='checkbox'  checked><label for='section-460e9924-424b-4ad7-9eab-88ff6453649c' class='xr-section-summary' >Data variables: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subjects</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-d994d83a-aaa4-402c-8652-b024c9860569' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d994d83a-aaa4-402c-8652-b024c9860569' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ba829091-20b8-45ce-896c-cf6e1573c8e8' class='xr-var-data-in' type='checkbox'><label for='data-ba829091-20b8-45ce-896c-cf6e1573c8e8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 4.049e+03 ... 2.092e+04</div><input id='attrs-fd399548-a415-4355-aa5e-5d8df0f47001' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fd399548-a415-4355-aa5e-5d8df0f47001' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a67a117e-3981-4871-9bfc-e6461ba74b88' class='xr-var-data-in' type='checkbox'><label for='data-a67a117e-3981-4871-9bfc-e6461ba74b88' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1686.7,  4049.4, 20663.2],
           [ 1371.1,  9312.6, 19954. ],
           [ 1414.8,  8972.6, 21645.2],
           [ 1830.6,  6798.6, 20790.6],
           [ 1642.4,  6112.5, 17692.6],
           [ 2108.4,  7076.4, 20996.8],
           [ 2023.1,  4862.2, 20964.9],
           [ 2193.4,  9931.7, 21339.8],
           [ 1086. ,  6479.5, 18517.9],
           [ 1604.9,  5890.9, 20919.9]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 17.58 19.88</div><input id='attrs-2f7a572c-0f8f-4791-96e7-0cfe7e78c866' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2f7a572c-0f8f-4791-96e7-0cfe7e78c866' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-32d9e615-fb29-4b70-9af5-75f495bdd312' class='xr-var-data-in' type='checkbox'><label for='data-32d9e615-fb29-4b70-9af5-75f495bdd312' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           [31.39],
           [13.58],
           [29.21],
           [15.92],
           [21.46],
           [17.58],
           [19.88]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;1&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;AnnArbor_a&#x27;</div><input id='attrs-1953fb25-2264-4efe-8a76-87ed085e2d1b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1953fb25-2264-4efe-8a76-87ed085e2d1b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c46762f6-916e-4c66-bc63-55519296bff4' class='xr-var-data-in' type='checkbox'><label for='data-c46762f6-916e-4c66-bc63-55519296bff4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d81ffbff-94b7-4952-acf8-15c34b99b746' class='xr-section-summary-in' type='checkbox'  ><label for='section-d81ffbff-94b7-4952-acf8-15c34b99b746' class='xr-section-summary' >Indexes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-4c4be73c-09c2-47e2-a0f2-31f430897e23' class='xr-index-data-in' type='checkbox'/><label for='index-4c4be73c-09c2-47e2-a0f2-31f430897e23' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-cf3702d8-fcdf-41b2-824c-53b26123fd05' class='xr-index-data-in' type='checkbox'/><label for='index-cf3702d8-fcdf-41b2-824c-53b26123fd05' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;, &#x27;Left-Lateral-Ventricle&#x27;, &#x27;Brain-Stem&#x27;], dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-0d023a50-9f40-4507-8623-1d92b0f04a20' class='xr-index-data-in' type='checkbox'/><label for='index-0d023a50-9f40-4507-8623-1d92b0f04a20' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-683b5811-aaea-4de4-b32f-5b05a54b8432' class='xr-index-data-in' type='checkbox'/><label for='index-683b5811-aaea-4de4-b32f-5b05a54b8432' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-7a6c55f1-f788-4cbb-abfe-9eb126f77feb' class='xr-section-summary-in' type='checkbox'  checked><label for='section-7a6c55f1-f788-4cbb-abfe-9eb126f77feb' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{&#x27;sex&#x27;: [&#x27;0&#x27;, &#x27;1&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;0&#x27;: 589, &#x27;1&#x27;: 489}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;0&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}, &#x27;1&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd><dt><span>covariate_ranges :</span></dt><dd>{&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd></dl></div></li></ul></div></div>



NormData with predictions
-------------------------

After fitting a model and predicting on NormData, the NormData object
will have new attributes holding the predictions.

Specifically, the NormData object will be extended with new data
variables:

- ``Z``: The predicted Z scores for each response variable
- ``centiles``: The predicted centiles
- ``logp``: The predicted log-p-values for each response variable
- ``Yhat``: The predicted mean of the response variable
- ``Y_harmonized``: The harmonized response variables
- ``statistics``: An array of statistics for each response variable

And the following new coordinates: - ``centile``: The specific centile
values - ``statistic``: The name of the computed statistics

.. code:: ipython3

    from pcntoolkit import NormativeModel, BLR
    
    # We create a very simple BLR model because it is fast to fit
    model = NormativeModel(BLR())
    model.fit(norm_data)  # Fitting on the data also makes predictions for that data


.. parsed-literal::

    Process: 91027 - 2025-06-12 16:00:30 - Fitting models on 3 response variables.
    Process: 91027 - 2025-06-12 16:00:30 - Fitting model for WM-hypointensities.
    Process: 91027 - 2025-06-12 16:00:30 - Fitting model for Left-Lateral-Ventricle.
    Process: 91027 - 2025-06-12 16:00:30 - Fitting model for Brain-Stem.
    Process: 91027 - 2025-06-12 16:00:30 - Making predictions on 3 response variables.
    Process: 91027 - 2025-06-12 16:00:30 - Computing z-scores for 3 response variables.
    Process: 91027 - 2025-06-12 16:00:30 - Computing z-scores for WM-hypointensities.
    Process: 91027 - 2025-06-12 16:00:31 - Computing z-scores for Brain-Stem.
    Process: 91027 - 2025-06-12 16:00:31 - Computing z-scores for Left-Lateral-Ventricle.
    Process: 91027 - 2025-06-12 16:00:31 - Computing centiles for 3 response variables.
    Process: 91027 - 2025-06-12 16:00:31 - Computing centiles for WM-hypointensities.
    Process: 91027 - 2025-06-12 16:00:31 - Computing centiles for Brain-Stem.
    Process: 91027 - 2025-06-12 16:00:31 - Computing centiles for Left-Lateral-Ventricle.
    Process: 91027 - 2025-06-12 16:00:31 - Computing log-probabilities for 3 response variables.
    Process: 91027 - 2025-06-12 16:00:31 - Computing log-probabilities for WM-hypointensities.
    Process: 91027 - 2025-06-12 16:00:31 - Computing log-probabilities for Brain-Stem.
    Process: 91027 - 2025-06-12 16:00:31 - Computing log-probabilities for Left-Lateral-Ventricle.


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:216: UserWarning: Process: 91027 - 2025-06-12 16:00:31 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)


.. parsed-literal::

    Process: 91027 - 2025-06-12 16:00:31 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 91027 - 2025-06-12 16:00:31 - Computing centiles for 3 response variables.
    Process: 91027 - 2025-06-12 16:00:31 - Computing centiles for WM-hypointensities.
    Process: 91027 - 2025-06-12 16:00:31 - Computing centiles for Brain-Stem.
    Process: 91027 - 2025-06-12 16:00:31 - Computing centiles for Left-Lateral-Ventricle.
    Process: 91027 - 2025-06-12 16:00:32 - Harmonizing data on 3 response variables.
    Process: 91027 - 2025-06-12 16:00:32 - Harmonizing data for WM-hypointensities.
    Process: 91027 - 2025-06-12 16:00:32 - Harmonizing data for Brain-Stem.
    Process: 91027 - 2025-06-12 16:00:32 - Harmonizing data for Left-Lateral-Ventricle.


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:278: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:278: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)


.. parsed-literal::

    Process: 91027 - 2025-06-12 16:00:32 - Saving model to:
    	/Users/stijndeboer/.pcntoolkit/saves.


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:278: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      non_be_df["marker"] = ["Other data"] * len(non_be_df)


.. code:: ipython3

    norm_data.data_vars




.. parsed-literal::

    Data variables:
        subjects       (observations) int64 9kB 0 1 2 3 4 ... 1074 1075 1076 1077
        Y              (observations, response_vars) float64 26kB 1.687e+03 ... 1...
        X              (observations, covariates) float64 9kB 25.63 18.34 ... 23.0
        batch_effects  (observations, batch_effect_dims) <U17 147kB '1' ... 'Sain...
        Z              (observations, response_vars) float64 26kB 0.8117 ... -0.2591
        centiles       (centile, observations, response_vars) float64 129kB -271....
        logp           (observations, response_vars) float64 26kB -7.929 ... -8.8
        Yhat           (observations, response_vars) float64 26kB 1.04e+03 ... 2....
        statistics     (response_vars, statistic) float64 240B 0.05581 ... 0.9964
        Y_harmonized   (observations, response_vars) float64 26kB 1.687e+03 ... 1...



.. code:: ipython3

    norm_data.coords




.. parsed-literal::

    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1074 1075 1076 1077
      * response_vars      (response_vars) <U22 264B 'WM-hypointensities' ... 'Br...
      * covariates         (covariates) <U3 12B 'age'
      * batch_effect_dims  (batch_effect_dims) <U4 32B 'sex' 'site'
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) <U8 320B 'MACE' 'MAPE' ... 'SMSE' 'ShapiroW'



Indexing of predicted data
~~~~~~~~~~~~~~~~~~~~~~~~~~

All the indexing methods can still be used, and they will also slice
through the newly added data variables. So for example, to select the
first 10 observations, you can use:

.. code:: python

   norm_data.sel(observations=slice(0, 9))

This will return a new NormData object with only the first 10
observations.

.. code:: ipython3

    norm_data.sel(observations=slice(0, 9))





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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 5kB
    Dimensions:            (observations: 10, response_vars: 3, covariates: 1,
                            batch_effect_dims: 2, centile: 5, statistic: 10)
    Coordinates:
      * observations       (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
      * response_vars      (response_vars) &lt;U22 264B &#x27;WM-hypointensities&#x27; ... &#x27;Br...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) &lt;U8 320B &#x27;MACE&#x27; &#x27;MAPE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;
    Data variables:
        subjects           (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
        Y                  (observations, response_vars) float64 240B 1.687e+03 ....
        X                  (observations, covariates) float64 80B 25.63 ... 19.88
        batch_effects      (observations, batch_effect_dims) &lt;U17 1kB &#x27;1&#x27; ... &#x27;An...
        Z                  (observations, response_vars) float64 240B 0.8117 ... ...
        centiles           (centile, observations, response_vars) float64 1kB -27...
        logp               (observations, response_vars) float64 240B -7.929 ... ...
        Yhat               (observations, response_vars) float64 240B 1.04e+03 .....
        statistics         (response_vars, statistic) float64 240B 0.05581 ... 0....
        Y_harmonized       (observations, response_vars) float64 240B 1.687e+03 ....
    Attributes:
        real_ids:                       False
        is_scaled:                      False
        name:                           fcon1000
        unique_batch_effects:           {&#x27;sex&#x27;: [&#x27;0&#x27;, &#x27;1&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;...
        batch_effect_counts:            {&#x27;sex&#x27;: {&#x27;0&#x27;: 589, &#x27;1&#x27;: 489}, &#x27;site&#x27;: {&#x27;A...
        batch_effect_covariate_ranges:  {&#x27;sex&#x27;: {&#x27;0&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;...
        covariate_ranges:               {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-93c00702-d1d1-439d-a8e0-d87c04dbd41d' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-93c00702-d1d1-439d-a8e0-d87c04dbd41d' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 10</li><li><span class='xr-has-index'>response_vars</span>: 3</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>centile</span>: 5</li><li><span class='xr-has-index'>statistic</span>: 10</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-d14a87d4-4257-4440-8aac-3e5b236e94ca' class='xr-section-summary-in' type='checkbox'  checked><label for='section-d14a87d4-4257-4440-8aac-3e5b236e94ca' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-46d04f3e-c09d-41a5-a207-4f407bf1b0b4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-46d04f3e-c09d-41a5-a207-4f407bf1b0b4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9170f63c-e3cf-43ef-9b8e-c24e32ddb8c2' class='xr-var-data-in' type='checkbox'><label for='data-9170f63c-e3cf-43ef-9b8e-c24e32ddb8c2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Brain-...</div><input id='attrs-69cb6af1-d601-4c1e-a911-e3606916e21c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-69cb6af1-d601-4c1e-a911-e3606916e21c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e373f8fe-d5c8-44c5-b87a-a0161d4b43ac' class='xr-var-data-in' type='checkbox'><label for='data-e373f8fe-d5c8-44c5-b87a-a0161d4b43ac' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Left-Lateral-Ventricle&#x27;, &#x27;Brain-Stem&#x27;],
          dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-f6874ab3-9c43-431e-823c-e6dda80a4509' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f6874ab3-9c43-431e-823c-e6dda80a4509' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-06f90981-f87f-4d95-a399-88da8f6735af' class='xr-var-data-in' type='checkbox'><label for='data-06f90981-f87f-4d95-a399-88da8f6735af' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-a3942e82-65ed-4a95-aa28-03639be032c7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a3942e82-65ed-4a95-aa28-03639be032c7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b34d24d0-7a37-4a82-a347-76854086df67' class='xr-var-data-in' type='checkbox'><label for='data-b34d24d0-7a37-4a82-a347-76854086df67' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-7478c7e7-6082-416d-bd24-6d225a286d5f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7478c7e7-6082-416d-bd24-6d225a286d5f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9e7bc387-9739-411f-b538-ad7759a0d641' class='xr-var-data-in' type='checkbox'><label for='data-9e7bc387-9739-411f-b538-ad7759a0d641' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;MACE&#x27; &#x27;MAPE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-e5c8d018-c803-4eca-b001-652ed3a5aaac' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e5c8d018-c803-4eca-b001-652ed3a5aaac' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bf49f5d3-e6a8-427b-82da-4e28c9ad88ed' class='xr-var-data-in' type='checkbox'><label for='data-bf49f5d3-e6a8-427b-82da-4e28c9ad88ed' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;,
           &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-abbc5291-f5e2-4d8b-8105-3deb8f92e489' class='xr-section-summary-in' type='checkbox'  checked><label for='section-abbc5291-f5e2-4d8b-8105-3deb8f92e489' class='xr-section-summary' >Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subjects</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-9e4fb116-0bd2-47ab-82e6-85ecf203a53f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9e4fb116-0bd2-47ab-82e6-85ecf203a53f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b5b0d2a7-dc36-4c72-8aa0-ac42ae6fc82a' class='xr-var-data-in' type='checkbox'><label for='data-b5b0d2a7-dc36-4c72-8aa0-ac42ae6fc82a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 4.049e+03 ... 2.092e+04</div><input id='attrs-2aeb1e8b-41bf-487f-9550-7c59e9c01b23' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2aeb1e8b-41bf-487f-9550-7c59e9c01b23' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-45753d82-e488-4794-9519-a619619ffc5e' class='xr-var-data-in' type='checkbox'><label for='data-45753d82-e488-4794-9519-a619619ffc5e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1686.7,  4049.4, 20663.2],
           [ 1371.1,  9312.6, 19954. ],
           [ 1414.8,  8972.6, 21645.2],
           [ 1830.6,  6798.6, 20790.6],
           [ 1642.4,  6112.5, 17692.6],
           [ 2108.4,  7076.4, 20996.8],
           [ 2023.1,  4862.2, 20964.9],
           [ 2193.4,  9931.7, 21339.8],
           [ 1086. ,  6479.5, 18517.9],
           [ 1604.9,  5890.9, 20919.9]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 17.58 19.88</div><input id='attrs-82d007e8-5fd2-4fee-86cc-47523eccc39a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-82d007e8-5fd2-4fee-86cc-47523eccc39a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-758384ff-080b-4425-8a4c-174e82505cfa' class='xr-var-data-in' type='checkbox'><label for='data-758384ff-080b-4425-8a4c-174e82505cfa' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           [31.39],
           [13.58],
           [29.21],
           [15.92],
           [21.46],
           [17.58],
           [19.88]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;1&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;AnnArbor_a&#x27;</div><input id='attrs-21875c30-8eb4-4763-b88c-2b7f1802b975' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-21875c30-8eb4-4763-b88c-2b7f1802b975' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a9077b1c-ace9-4bb5-9d4a-c8c374b44bb9' class='xr-var-data-in' type='checkbox'><label for='data-a9077b1c-ace9-4bb5-9d4a-c8c374b44bb9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.8117 -0.5312 ... 0.2079 0.1194</div><input id='attrs-70080821-fabc-430f-8693-a134c815d9a7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-70080821-fabc-430f-8693-a134c815d9a7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-83440a8e-79f9-4e41-b8ea-bc67ed4d930d' class='xr-var-data-in' type='checkbox'><label for='data-83440a8e-79f9-4e41-b8ea-bc67ed4d930d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.81166894, -0.53119164,  0.01908509],
           [ 0.78680367,  1.06617572, -0.25787247],
           [ 0.2889267 ,  0.37746293,  0.40266401],
           [ 0.69900651, -0.2359351 ,  0.06877881],
           [ 1.3694996 ,  0.61294383, -1.14123688],
           [ 1.15840188, -0.05051823,  0.14936129],
           [ 1.72802298,  0.19915215,  0.13707421],
           [ 1.65960365,  1.02988645,  0.28345893],
           [ 0.46779484,  0.47024978, -0.81888426],
           [ 1.00173737,  0.20792512,  0.11944267]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-271.6 -890.8 ... 2.482e+04</div><input id='attrs-a1c089cc-3b9b-4267-9865-8222ad1a3f54' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a1c089cc-3b9b-4267-9865-8222ad1a3f54' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6dd179ad-383a-4406-b8ca-4c9b9e173025' class='xr-var-data-in' type='checkbox'><label for='data-6dd179ad-383a-4406-b8ca-4c9b9e173025' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-2.71595586e+02, -8.90802014e+02,  1.64038750e+04],
            [-5.67079541e+02, -2.71169571e+03,  1.64036171e+04],
            [-1.26917623e+02,  7.77565665e-01,  1.64039964e+04],
            [-3.81734191e+01,  5.47669422e+02,  1.64040694e+04],
            [-7.60051167e+02, -3.90084487e+03,  1.64034415e+04],
            [-1.26512385e+02,  3.27486395e+00,  1.64039968e+04],
            [-6.65183449e+02, -3.31624358e+03,  1.64035285e+04],
            [-4.40609199e+02, -1.93233827e+03,  1.64037291e+04],
            [-5.97888250e+02, -2.90154936e+03,  1.64035894e+04],
            [-5.04653567e+02, -2.32700447e+03,  1.64036727e+04]],
    
           [[ 5.01961036e+02,  3.41372989e+03,  1.88877977e+04],
            [ 2.06359797e+02,  1.59218326e+03,  1.88875465e+04],
            [ 6.46710626e+02,  4.30570816e+03,  1.88879187e+04],
            [ 7.35503385e+02,  4.85287025e+03,  1.88879923e+04],
            [ 1.33325922e+01,  4.02724571e+02,  1.88873796e+04],
            [ 6.47116078e+02,  4.30820664e+03,  1.88879190e+04],
            [ 1.08225559e+02,  9.87466482e+02,  1.88874620e+04],
            [ 3.32875572e+02,  2.37179365e+03,  1.88876547e+04],
            [ 1.75541101e+02,  1.40227399e+03,  1.88875200e+04],
    ...
            [ 1.28157894e+03,  7.57534549e+03,  2.23406531e+04],
            [ 1.72219238e+03,  1.02903323e+04,  2.23410153e+04],
            [ 1.81105264e+03,  1.08378701e+04,  2.23410898e+04],
            [ 1.08847447e+03,  6.38545650e+03,  2.23404982e+04],
            [ 1.72259813e+03,  1.02928325e+04,  2.23410156e+04],
            [ 1.18340253e+03,  6.97039391e+03,  2.23405741e+04],
            [ 1.40815787e+03,  8.35530754e+03,  2.23407559e+04],
            [ 1.25074636e+03,  7.38535891e+03,  2.23406282e+04],
            [ 1.34405638e+03,  7.96032331e+03,  2.23407037e+04]],
    
           [[ 2.35089984e+03,  1.37023317e+04,  2.48248176e+04],
            [ 2.05501827e+03,  1.18792245e+04,  2.48245825e+04],
            [ 2.49582063e+03,  1.45952629e+04,  2.48249375e+04],
            [ 2.58472945e+03,  1.51430709e+04,  2.48250127e+04],
            [ 1.86185823e+03,  1.06890259e+04,  2.48244363e+04],
            [ 2.49622660e+03,  1.45977643e+04,  2.48249379e+04],
            [ 1.95681154e+03,  1.12741040e+04,  2.48245075e+04],
            [ 2.18164264e+03,  1.26594395e+04,  2.48246815e+04],
            [ 2.02417571e+03,  1.16891823e+04,  2.48245588e+04],
            [ 2.11751725e+03,  1.22643222e+04,  2.48246311e+04]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-7.929 -9.458 ... -9.338 -8.774</div><input id='attrs-6ebfb58c-6d11-4780-b53e-4d3176dcfdaf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6ebfb58c-6d11-4780-b53e-4d3176dcfdaf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e6d45b08-917a-41f1-b692-536766a27f31' class='xr-var-data-in' type='checkbox'><label for='data-e6d45b08-917a-41f1-b692-536766a27f31' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-7.92942478, -9.45752864, -8.76679912],
           [-7.90939993, -9.88466   , -8.79986882],
           [-7.64185346, -9.3877781 , -8.84768597],
           [-7.84448195, -9.34443443, -8.76898235],
           [-8.53756263, -9.50407281, -9.417834  ],
           [-8.27106187, -9.3178153 , -8.77777122],
           [-9.09286242, -9.33608621, -8.77601599],
           [-8.9770708 , -9.84668649, -8.80679265],
           [-7.70927301, -9.42684917, -9.10190589],
           [-8.10163664, -9.33793895, -8.77375215]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.04e+03 6.406e+03 ... 2.061e+04</div><input id='attrs-09d5ed89-c28e-4fb1-b4d3-c0b163b6d22a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-09d5ed89-c28e-4fb1-b4d3-c0b163b6d22a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-362f66de-1845-4423-aae1-60d11284abb5' class='xr-var-data-in' type='checkbox'><label for='data-362f66de-1845-4423-aae1-60d11284abb5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1039.65212783,  6405.76486163, 20614.34628677],
           [  743.96936581,  4583.76437721, 20614.09982434],
           [ 1184.45150512,  7298.02024289, 20614.46698237],
           [ 1273.27801387,  7845.37018266, 20614.54102252],
           [  550.90352942,  3394.09053553, 20613.93889687],
           [ 1184.85710561,  7300.51955769, 20614.46732045],
           [  645.81404563,  3978.9301972 , 20614.01800827],
           [  870.51672075,  5363.55059277, 20614.2053062 ],
           [  713.14372806,  4393.81645291, 20614.07413003],
           [  806.43184228,  4968.6588554 , 20614.1518891 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05581 0.454 ... 1.0 0.9964</div><input id='attrs-2e4fc556-4afb-4600-9a63-d5f436dd4acd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2e4fc556-4afb-4600-9a63-d5f436dd4acd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-26e06eed-1bd3-4371-a64a-fd75b57415c4' class='xr-var-data-in' type='checkbox'><label for='data-26e06eed-1bd3-4371-a64a-fd75b57415c4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 5.58070501e-02,  4.53956023e-01, -3.25480412e-04,
             8.07756061e+00, -1.13582231e-03,  7.79664882e+02,
             9.91346248e-03,  7.45091622e-01,  1.00113582e+00,
             8.64052887e-01],
           [ 5.02782931e-02,  3.99107088e-01, -3.76018410e-03,
             9.79099158e+00, -6.84925953e-03,  4.32300403e+03,
             2.59840757e-01,  4.29131957e-18,  1.00684926e+00,
             9.47832829e-01],
           [ 8.31168831e-03,  9.77312839e-02, -6.59853255e-04,
             9.24131392e+00,  5.54977074e-06,  2.49417368e+03,
             6.10744682e-02,  4.49846682e-02,  9.99994450e-01,
             9.96388221e-01]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y_harmonized</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 4.049e+03 ... 2.092e+04</div><input id='attrs-a1625b83-d4fa-449b-a640-f2a97a00ffc3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a1625b83-d4fa-449b-a640-f2a97a00ffc3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8412bddd-9a18-49d3-9f30-045ccd7595f7' class='xr-var-data-in' type='checkbox'><label for='data-8412bddd-9a18-49d3-9f30-045ccd7595f7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1686.7,  4049.4, 20663.2],
           [ 1371.1,  9312.6, 19954. ],
           [ 1414.8,  8972.6, 21645.2],
           [ 1830.6,  6798.6, 20790.6],
           [ 1642.4,  6112.5, 17692.6],
           [ 2108.4,  7076.4, 20996.8],
           [ 2023.1,  4862.2, 20964.9],
           [ 2193.4,  9931.7, 21339.8],
           [ 1086. ,  6479.5, 18517.9],
           [ 1604.9,  5890.9, 20919.9]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-b6d60605-db6e-4cee-8acf-f50416199c35' class='xr-section-summary-in' type='checkbox'  ><label for='section-b6d60605-db6e-4cee-8acf-f50416199c35' class='xr-section-summary' >Indexes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-3a11f52d-0bbd-4cb1-8669-9376897105e0' class='xr-index-data-in' type='checkbox'/><label for='index-3a11f52d-0bbd-4cb1-8669-9376897105e0' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-fdd7bd48-736c-4453-8395-3e32cfb77dda' class='xr-index-data-in' type='checkbox'/><label for='index-fdd7bd48-736c-4453-8395-3e32cfb77dda' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;, &#x27;Left-Lateral-Ventricle&#x27;, &#x27;Brain-Stem&#x27;], dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-4e494f75-bee2-40bf-ab6c-eb5e4fdec8dc' class='xr-index-data-in' type='checkbox'/><label for='index-4e494f75-bee2-40bf-ab6c-eb5e4fdec8dc' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-fdb229fd-42f1-49d8-a624-b20f76fe280f' class='xr-index-data-in' type='checkbox'/><label for='index-fdb229fd-42f1-49d8-a624-b20f76fe280f' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>centile</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-91c18c94-9ce4-4123-aa65-626b17f3575a' class='xr-index-data-in' type='checkbox'/><label for='index-91c18c94-9ce4-4123-aa65-626b17f3575a' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0.05, 0.25, 0.5, 0.75, 0.95], dtype=&#x27;float64&#x27;, name=&#x27;centile&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-9b2e7c4c-a8be-41ed-8c74-2dcb7ac6b85e' class='xr-index-data-in' type='checkbox'/><label for='index-9b2e7c4c-a8be-41ed-8c74-2dcb7ac6b85e' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;,
           &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-90fec25e-9aba-40bd-bf52-89510d54cf8c' class='xr-section-summary-in' type='checkbox'  checked><label for='section-90fec25e-9aba-40bd-bf52-89510d54cf8c' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{&#x27;sex&#x27;: [&#x27;0&#x27;, &#x27;1&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;0&#x27;: 589, &#x27;1&#x27;: 489}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;0&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}, &#x27;1&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd><dt><span>covariate_ranges :</span></dt><dd>{&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd></dl></div></li></ul></div></div>



Or, if we want to select only the WM-hypointensities, we can use:

.. code:: python

   norm_data.sel(response_vars="WM-hypointensities")

This will return a new NormData object with only the WM-hypointensities.

.. code:: ipython3

    norm_data.sel(response_vars="WM-hypointensities").statistics




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;statistics&#x27; (statistic: 10)&gt; Size: 80B
    array([ 5.58070501e-02,  4.53956023e-01, -3.25480412e-04,  8.07756061e+00,
           -1.13582231e-03,  7.79664882e+02,  9.91346248e-03,  7.45091622e-01,
            1.00113582e+00,  8.64052887e-01])
    Coordinates:
        response_vars  &lt;U22 88B &#x27;WM-hypointensities&#x27;
      * statistic      (statistic) &lt;U8 320B &#x27;MACE&#x27; &#x27;MAPE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'statistics'</div><ul class='xr-dim-list'><li><span class='xr-has-index'>statistic</span>: 10</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-99ee3c13-64f7-4b75-a91b-34f38e3dfa7d' class='xr-array-in' type='checkbox' checked><label for='section-99ee3c13-64f7-4b75-a91b-34f38e3dfa7d' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.05581 0.454 -0.0003255 8.078 ... 0.009913 0.7451 1.001 0.8641</span></div><div class='xr-array-data'><pre>array([ 5.58070501e-02,  4.53956023e-01, -3.25480412e-04,  8.07756061e+00,
           -1.13582231e-03,  7.79664882e+02,  9.91346248e-03,  7.45091622e-01,
            1.00113582e+00,  8.64052887e-01])</pre></div></div></li><li class='xr-section-item'><input id='section-114305b5-6cd1-413d-be1f-cd2f8a6d38d4' class='xr-section-summary-in' type='checkbox'  checked><label for='section-114305b5-6cd1-413d-be1f-cd2f8a6d38d4' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>response_vars</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27;</div><input id='attrs-dd8f267d-2956-4cc0-82c2-16eed62c21ba' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dd8f267d-2956-4cc0-82c2-16eed62c21ba' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4a8d67e6-b503-4a5d-b894-e3d76d848d14' class='xr-var-data-in' type='checkbox'><label for='data-4a8d67e6-b503-4a5d-b894-e3d76d848d14' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(&#x27;WM-hypointensities&#x27;, dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;MACE&#x27; &#x27;MAPE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-c88347e7-ebc4-459b-b14b-e2d270fe4c05' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c88347e7-ebc4-459b-b14b-e2d270fe4c05' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7138be22-9db7-4fc6-b3c5-35c955f11cf4' class='xr-var-data-in' type='checkbox'><label for='data-7138be22-9db7-4fc6-b3c5-35c955f11cf4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;,
           &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f609886c-3bd0-470b-8d6e-e77db8a68ba2' class='xr-section-summary-in' type='checkbox'  ><label for='section-f609886c-3bd0-470b-8d6e-e77db8a68ba2' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-817c7c47-3fa0-4ed5-8358-59aff7e9b52b' class='xr-index-data-in' type='checkbox'/><label for='index-817c7c47-3fa0-4ed5-8358-59aff7e9b52b' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;,
           &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ff8903b7-7986-4a48-93b6-428ec0dfd564' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-ff8903b7-7986-4a48-93b6-428ec0dfd564' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



Now we can use the to_dataframe method to cast that selection back to a
pandas dataframe.

.. code:: ipython3

    new_df = norm_data.sel(response_vars="WM-hypointensities").to_dataframe()
    new_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>X</th>
          <th>Y</th>
          <th>Y_harmonized</th>
          <th>Z</th>
          <th colspan="2" halign="left">batch_effects</th>
          <th>subjects</th>
          <th colspan="5" halign="left">centiles</th>
        </tr>
        <tr>
          <th></th>
          <th>age</th>
          <th>WM-hypointensities</th>
          <th>WM-hypointensities</th>
          <th>WM-hypointensities</th>
          <th>sex</th>
          <th>site</th>
          <th>subjects</th>
          <th>(WM-hypointensities, 0.05)</th>
          <th>(WM-hypointensities, 0.25)</th>
          <th>(WM-hypointensities, 0.5)</th>
          <th>(WM-hypointensities, 0.75)</th>
          <th>(WM-hypointensities, 0.95)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25.63</td>
          <td>1686.7</td>
          <td>1686.7</td>
          <td>0.811669</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>0</td>
          <td>-271.595586</td>
          <td>501.961036</td>
          <td>1039.652128</td>
          <td>1577.343220</td>
          <td>2350.899842</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.34</td>
          <td>1371.1</td>
          <td>1371.1</td>
          <td>0.786804</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>1</td>
          <td>-567.079541</td>
          <td>206.359797</td>
          <td>743.969366</td>
          <td>1281.578935</td>
          <td>2055.018273</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.20</td>
          <td>1414.8</td>
          <td>1414.8</td>
          <td>0.288927</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>2</td>
          <td>-126.917623</td>
          <td>646.710626</td>
          <td>1184.451505</td>
          <td>1722.192384</td>
          <td>2495.820633</td>
        </tr>
        <tr>
          <th>3</th>
          <td>31.39</td>
          <td>1830.6</td>
          <td>1830.6</td>
          <td>0.699007</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>3</td>
          <td>-38.173419</td>
          <td>735.503385</td>
          <td>1273.278014</td>
          <td>1811.052643</td>
          <td>2584.729447</td>
        </tr>
        <tr>
          <th>4</th>
          <td>13.58</td>
          <td>1642.4</td>
          <td>1642.4</td>
          <td>1.369500</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>4</td>
          <td>-760.051167</td>
          <td>13.332592</td>
          <td>550.903529</td>
          <td>1088.474467</td>
          <td>1861.858226</td>
        </tr>
      </tbody>
    </table>
    </div>



This should give you a pretty good overview of how to work with
NormData. Most of the functionality is built on top of xarray, so if you
want to learn more about xarray, you can check out the `xarray
documentation <https://docs.xarray.dev/en/stable/>`__. However, the
Xarray.DataSet class does not officially support being extended, so the
API does not work completely as expected.

If you have any suggestions for improvements, please let us know!


