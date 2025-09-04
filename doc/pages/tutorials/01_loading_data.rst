The NormData class
==================

A key component of the PCNtoolkit is the NormData object. It is a
container for the data that will be used to fit the normative model. The
NormData object keeps track of the all the dimensions of your data, the
features and response variables, batch effects, preprocessing steps, and
more.

.. code:: ipython3

    import copy
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    from pcntoolkit import NormData


Creating a NormData object
--------------------------

There are currently two easy ways to create a NormData object. 1. Load
from a pandas dataframe 2. Load from numpy arrays

Here are examples of both.

.. code:: ipython3

    # Creating a NormData object from a pandas dataframe
    
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
        name="fcon1000",
        dataframe=data,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
        remove_outliers=True,
        z_threshold=10,
    )
    norm_data.coords


.. parsed-literal::

    Process: 28421 - 2025-06-30 13:54:21 - Dataset "fcon1000" created.
        - 1078 observations
        - 1078 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 28421 - 2025-06-30 13:54:21 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
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

    Process: 28421 - 2025-06-30 13:54:21 - Dataset "fcon1000" created.
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
          <th colspan="2" halign="left">batch_effects</th>
          <th>subjects</th>
        </tr>
        <tr>
          <th></th>
          <th>age</th>
          <th>Brain-Stem</th>
          <th>Left-Lateral-Ventricle</th>
          <th>WM-hypointensities</th>
          <th>sex</th>
          <th>site</th>
          <th>subjects</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25.63</td>
          <td>20663.2</td>
          <td>4049.4</td>
          <td>1686.7</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.34</td>
          <td>19954.0</td>
          <td>9312.6</td>
          <td>1371.1</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.20</td>
          <td>21645.2</td>
          <td>8972.6</td>
          <td>1414.8</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>2</td>
        </tr>
        <tr>
          <th>3</th>
          <td>31.39</td>
          <td>20790.6</td>
          <td>6798.6</td>
          <td>1830.6</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>3</td>
        </tr>
        <tr>
          <th>4</th>
          <td>13.58</td>
          <td>17692.6</td>
          <td>6112.5</td>
          <td>1642.4</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>4</td>
        </tr>
      </tbody>
    </table>
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
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}
        batch_effect_covariate_ranges:  {&#x27;sex&#x27;: {&#x27;0&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-ba4978a0-6b9b-45d4-a84d-235260240250' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-ba4978a0-6b9b-45d4-a84d-235260240250' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 1078</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-f3fdd396-2bfb-46bd-a56a-e7f31a77a11d' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f3fdd396-2bfb-46bd-a56a-e7f31a77a11d' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1074 1075 1076 1077</div><input id='attrs-14c87953-3ede-45fd-a2a9-5de425ab2652' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-14c87953-3ede-45fd-a2a9-5de425ab2652' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8195f21d-a6b8-4e28-8505-1e9d161830da' class='xr-var-data-in' type='checkbox'><label for='data-8195f21d-a6b8-4e28-8505-1e9d161830da' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1075, 1076, 1077])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>response_vars</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27;</div><input id='attrs-b03d67e1-401f-440b-be57-22f44fa9b70a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b03d67e1-401f-440b-be57-22f44fa9b70a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d1c5c2b5-3022-4fd7-8132-f53e5c953df9' class='xr-var-data-in' type='checkbox'><label for='data-d1c5c2b5-3022-4fd7-8132-f53e5c953df9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(&#x27;WM-hypointensities&#x27;, dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-bfcd9069-5684-4c0d-92f9-858ee566bc88' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bfcd9069-5684-4c0d-92f9-858ee566bc88' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-53042ca2-4d18-47fc-935a-9be0fb1c3921' class='xr-var-data-in' type='checkbox'><label for='data-53042ca2-4d18-47fc-935a-9be0fb1c3921' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-9fa8eae9-8dd6-4671-9738-6e3cd7e10f8e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9fa8eae9-8dd6-4671-9738-6e3cd7e10f8e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6d0c15fa-28a8-435e-9810-54dbc4a9fcb7' class='xr-var-data-in' type='checkbox'><label for='data-6d0c15fa-28a8-435e-9810-54dbc4a9fcb7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-8a8d64c3-ac14-45cd-9611-281265f91ab4' class='xr-section-summary-in' type='checkbox'  checked><label for='section-8a8d64c3-ac14-45cd-9611-281265f91ab4' class='xr-section-summary' >Data variables: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subjects</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1074 1075 1076 1077</div><input id='attrs-8b02929b-6e17-4224-b6e6-75a4e4957edf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8b02929b-6e17-4224-b6e6-75a4e4957edf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-913f9508-f649-4e3c-b5c6-4694cdded165' class='xr-var-data-in' type='checkbox'><label for='data-913f9508-f649-4e3c-b5c6-4694cdded165' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1075, 1076, 1077])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 1.371e+03 ... 448.3 509.1</div><input id='attrs-1fea037b-8995-4e8d-bd60-375231cda2a1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1fea037b-8995-4e8d-bd60-375231cda2a1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0b8b91dc-5739-4349-96c7-3061526b2f85' class='xr-var-data-in' type='checkbox'><label for='data-0b8b91dc-5739-4349-96c7-3061526b2f85' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1686.7, 1371.1, 1414.8, ..., 1061. ,  448.3,  509.1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 27.0 29.0 23.0</div><input id='attrs-f3ba3631-1770-41f1-9d87-c8972cddf7f1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f3ba3631-1770-41f1-9d87-c8972cddf7f1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6598f297-cdb3-4622-a23e-2b3869039769' class='xr-var-data-in' type='checkbox'><label for='data-6598f297-cdb3-4622-a23e-2b3869039769' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           ...,
           [27.  ],
           [29.  ],
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;1&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;SaintLouis&#x27;</div><input id='attrs-df8ae6fa-ed55-4bf5-843a-7fabbe66a7a4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-df8ae6fa-ed55-4bf5-843a-7fabbe66a7a4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f5159ecb-07eb-4618-b182-a3ce875f6a12' class='xr-var-data-in' type='checkbox'><label for='data-f5159ecb-07eb-4618-b182-a3ce875f6a12' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           ...,
           [&#x27;1&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;0&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;0&#x27;, &#x27;SaintLouis&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-4b33b621-dfb3-406c-8926-24b8427c31e6' class='xr-section-summary-in' type='checkbox'  ><label for='section-4b33b621-dfb3-406c-8926-24b8427c31e6' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-a670f16c-c3dd-4330-a351-719146d09b46' class='xr-index-data-in' type='checkbox'/><label for='index-a670f16c-c3dd-4330-a351-719146d09b46' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
           ...
           1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077],
          dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;, length=1078))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-0d1f2dcc-fad3-4b84-b785-e6dc9b228261' class='xr-index-data-in' type='checkbox'/><label for='index-0d1f2dcc-fad3-4b84-b785-e6dc9b228261' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-051d6704-7877-4a12-be40-edeb4205b118' class='xr-index-data-in' type='checkbox'/><label for='index-051d6704-7877-4a12-be40-edeb4205b118' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-5dd79a98-f84e-43db-8a65-3a5190916d50' class='xr-section-summary-in' type='checkbox'  checked><label for='section-5dd79a98-f84e-43db-8a65-3a5190916d50' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{&#x27;sex&#x27;: [&#x27;0&#x27;, &#x27;1&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x306987600&gt;, {&#x27;sex&#x27;: {&#x27;0&#x27;: 589, &#x27;1&#x27;: 489}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;0&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}, &#x27;1&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



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
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}
        batch_effect_covariate_ranges:  {&#x27;sex&#x27;: {&#x27;0&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-3d0dc19f-4328-430d-bf3b-ee9ed2fad930' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-3d0dc19f-4328-430d-bf3b-ee9ed2fad930' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 10</li><li><span class='xr-has-index'>response_vars</span>: 3</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-dc9af172-f94f-412e-a90e-47e388faf296' class='xr-section-summary-in' type='checkbox'  checked><label for='section-dc9af172-f94f-412e-a90e-47e388faf296' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-6f5d8d8e-716a-48f2-913f-cef894881005' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6f5d8d8e-716a-48f2-913f-cef894881005' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-02df861b-c85e-4c4a-8ee9-89a4a76df03d' class='xr-var-data-in' type='checkbox'><label for='data-02df861b-c85e-4c4a-8ee9-89a4a76df03d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Brain-...</div><input id='attrs-dbe04644-51e4-4b1c-bd71-ece4967a3719' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dbe04644-51e4-4b1c-bd71-ece4967a3719' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c43148f7-8f32-4a47-887c-2cb32216ed07' class='xr-var-data-in' type='checkbox'><label for='data-c43148f7-8f32-4a47-887c-2cb32216ed07' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Left-Lateral-Ventricle&#x27;, &#x27;Brain-Stem&#x27;],
          dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-e9d78ec4-f6be-4d31-9296-4dd60d20316f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e9d78ec4-f6be-4d31-9296-4dd60d20316f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-92d81fc3-b4f0-432a-99f5-9425cdd9f172' class='xr-var-data-in' type='checkbox'><label for='data-92d81fc3-b4f0-432a-99f5-9425cdd9f172' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-558bbb37-bf6e-47b0-8f40-2bfadc45cfe2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-558bbb37-bf6e-47b0-8f40-2bfadc45cfe2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d6a5c383-d458-44bf-9875-d60c441b81a1' class='xr-var-data-in' type='checkbox'><label for='data-d6a5c383-d458-44bf-9875-d60c441b81a1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-a0322a52-5aa9-431f-a1f7-df7faf7c4983' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a0322a52-5aa9-431f-a1f7-df7faf7c4983' class='xr-section-summary' >Data variables: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subjects</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-8246b713-85e0-4b7e-bb00-20b8660e4f41' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8246b713-85e0-4b7e-bb00-20b8660e4f41' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-41a76bc1-4caa-4a8e-88e9-d63ad8139a08' class='xr-var-data-in' type='checkbox'><label for='data-41a76bc1-4caa-4a8e-88e9-d63ad8139a08' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 4.049e+03 ... 2.092e+04</div><input id='attrs-28635243-415a-44e6-8c7a-2b66f07e6551' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-28635243-415a-44e6-8c7a-2b66f07e6551' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f6792444-806b-4cb0-884e-9895e2ac9ad9' class='xr-var-data-in' type='checkbox'><label for='data-f6792444-806b-4cb0-884e-9895e2ac9ad9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1686.7,  4049.4, 20663.2],
           [ 1371.1,  9312.6, 19954. ],
           [ 1414.8,  8972.6, 21645.2],
           [ 1830.6,  6798.6, 20790.6],
           [ 1642.4,  6112.5, 17692.6],
           [ 2108.4,  7076.4, 20996.8],
           [ 2023.1,  4862.2, 20964.9],
           [ 2193.4,  9931.7, 21339.8],
           [ 1086. ,  6479.5, 18517.9],
           [ 1604.9,  5890.9, 20919.9]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 17.58 19.88</div><input id='attrs-555337b2-e875-48d4-9801-a7ea9cf36907' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-555337b2-e875-48d4-9801-a7ea9cf36907' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7af491aa-616b-4602-86c8-c455c11020c4' class='xr-var-data-in' type='checkbox'><label for='data-7af491aa-616b-4602-86c8-c455c11020c4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           [31.39],
           [13.58],
           [29.21],
           [15.92],
           [21.46],
           [17.58],
           [19.88]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;1&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;AnnArbor_a&#x27;</div><input id='attrs-dff41737-5f7e-41bf-b228-14c7d4dbb2cd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dff41737-5f7e-41bf-b228-14c7d4dbb2cd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-08ac21a6-e690-4b4c-a58f-722128bae6d2' class='xr-var-data-in' type='checkbox'><label for='data-08ac21a6-e690-4b4c-a58f-722128bae6d2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ee8ecbac-0726-433e-b8d8-857ed2317428' class='xr-section-summary-in' type='checkbox'  ><label for='section-ee8ecbac-0726-433e-b8d8-857ed2317428' class='xr-section-summary' >Indexes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-99ce0e97-4041-48ad-8044-cc0225543a3c' class='xr-index-data-in' type='checkbox'/><label for='index-99ce0e97-4041-48ad-8044-cc0225543a3c' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-c11be6c3-a965-49e0-9812-17a15c3ed102' class='xr-index-data-in' type='checkbox'/><label for='index-c11be6c3-a965-49e0-9812-17a15c3ed102' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;, &#x27;Left-Lateral-Ventricle&#x27;, &#x27;Brain-Stem&#x27;], dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-170828cc-d620-46f6-b8a9-cf6083347b8e' class='xr-index-data-in' type='checkbox'/><label for='index-170828cc-d620-46f6-b8a9-cf6083347b8e' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-9a01c047-07a3-4205-8a28-520813536d28' class='xr-index-data-in' type='checkbox'/><label for='index-9a01c047-07a3-4205-8a28-520813536d28' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1e74a233-79d2-477e-be97-38c525a38fcc' class='xr-section-summary-in' type='checkbox'  checked><label for='section-1e74a233-79d2-477e-be97-38c525a38fcc' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{&#x27;sex&#x27;: [&#x27;0&#x27;, &#x27;1&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x306987600&gt;, {&#x27;sex&#x27;: {&#x27;0&#x27;: 589, &#x27;1&#x27;: 489}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;0&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}, &#x27;1&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



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

    from pcntoolkit import BLR, NormativeModel
    
    # We create a very simple BLR model because it is fast to fit
    model = NormativeModel(BLR())
    model.fit(norm_data)  # Fitting on the data also makes predictions for that data


.. parsed-literal::

    Process: 28421 - 2025-06-30 13:54:21 - Fitting models on 3 response variables.
    Process: 28421 - 2025-06-30 13:54:21 - Fitting model for WM-hypointensities.
    Process: 28421 - 2025-06-30 13:54:21 - Fitting model for Left-Lateral-Ventricle.
    Process: 28421 - 2025-06-30 13:54:21 - Fitting model for Brain-Stem.
    Process: 28421 - 2025-06-30 13:54:21 - Making predictions on 3 response variables.
    Process: 28421 - 2025-06-30 13:54:21 - Computing z-scores for 3 response variables.
    Process: 28421 - 2025-06-30 13:54:21 - Computing z-scores for Brain-Stem.
    Process: 28421 - 2025-06-30 13:54:21 - Computing z-scores for Left-Lateral-Ventricle.
    Process: 28421 - 2025-06-30 13:54:22 - Computing z-scores for WM-hypointensities.
    Process: 28421 - 2025-06-30 13:54:22 - Computing centiles for 3 response variables.
    Process: 28421 - 2025-06-30 13:54:22 - Computing centiles for Brain-Stem.
    Process: 28421 - 2025-06-30 13:54:22 - Computing centiles for Left-Lateral-Ventricle.
    Process: 28421 - 2025-06-30 13:54:22 - Computing centiles for WM-hypointensities.
    Process: 28421 - 2025-06-30 13:54:22 - Computing log-probabilities for 3 response variables.
    Process: 28421 - 2025-06-30 13:54:22 - Computing log-probabilities for Brain-Stem.
    Process: 28421 - 2025-06-30 13:54:22 - Computing log-probabilities for Left-Lateral-Ventricle.
    Process: 28421 - 2025-06-30 13:54:22 - Computing log-probabilities for WM-hypointensities.
    Process: 28421 - 2025-06-30 13:54:22 - Computing yhat for 3 response variables.
    Process: 28421 - 2025-06-30 13:54:22 - Computing yhat for Brain-Stem.
    Process: 28421 - 2025-06-30 13:54:22 - Computing yhat for Left-Lateral-Ventricle.
    Process: 28421 - 2025-06-30 13:54:22 - Computing yhat for WM-hypointensities.


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 28421 - 2025-06-30 13:54:23 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)


.. parsed-literal::

    Process: 28421 - 2025-06-30 13:54:23 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 28421 - 2025-06-30 13:54:23 - Computing centiles for 3 response variables.
    Process: 28421 - 2025-06-30 13:54:23 - Computing centiles for Brain-Stem.
    Process: 28421 - 2025-06-30 13:54:23 - Computing centiles for Left-Lateral-Ventricle.
    Process: 28421 - 2025-06-30 13:54:23 - Computing centiles for WM-hypointensities.
    Process: 28421 - 2025-06-30 13:54:23 - Harmonizing data on 3 response variables.
    Process: 28421 - 2025-06-30 13:54:23 - Harmonizing data for Brain-Stem.
    Process: 28421 - 2025-06-30 13:54:23 - Harmonizing data for Left-Lateral-Ventricle.
    Process: 28421 - 2025-06-30 13:54:23 - Harmonizing data for WM-hypointensities.
    Process: 28421 - 2025-06-30 13:54:24 - Saving model to:
    	/Users/stijndeboer/.pcntoolkit/saves.


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
        Yhat           (observations, response_vars) float64 26kB -25.7 ... 1.87e+04
        statistics     (response_vars, statistic) float64 264B -0.9437 ... 0.9964
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
      * statistic          (statistic) <U8 352B 'EXPV' 'MACE' ... 'SMSE' 'ShapiroW'



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
                            batch_effect_dims: 2, centile: 5, statistic: 11)
    Coordinates:
      * observations       (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
      * response_vars      (response_vars) &lt;U22 264B &#x27;WM-hypointensities&#x27; ... &#x27;Br...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) &lt;U8 352B &#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;
    Data variables:
        subjects           (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
        Y                  (observations, response_vars) float64 240B 1.687e+03 ....
        X                  (observations, covariates) float64 80B 25.63 ... 19.88
        batch_effects      (observations, batch_effect_dims) &lt;U17 1kB &#x27;1&#x27; ... &#x27;An...
        Z                  (observations, response_vars) float64 240B 0.8117 ... ...
        centiles           (centile, observations, response_vars) float64 1kB -27...
        logp               (observations, response_vars) float64 240B -7.929 ... ...
        Yhat               (observations, response_vars) float64 240B -25.7 ... 2...
        statistics         (response_vars, statistic) float64 264B -0.9437 ... 0....
        Y_harmonized       (observations, response_vars) float64 240B 1.687e+03 ....
    Attributes:
        real_ids:                       False
        is_scaled:                      False
        name:                           fcon1000
        unique_batch_effects:           {&#x27;sex&#x27;: [&#x27;0&#x27;, &#x27;1&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}
        batch_effect_covariate_ranges:  {&#x27;sex&#x27;: {&#x27;0&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-b4a256c3-f76e-4268-ae93-9e2377d062a5' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-b4a256c3-f76e-4268-ae93-9e2377d062a5' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 10</li><li><span class='xr-has-index'>response_vars</span>: 3</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>centile</span>: 5</li><li><span class='xr-has-index'>statistic</span>: 11</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-78affbdd-a28f-4dc3-b42b-067338574062' class='xr-section-summary-in' type='checkbox'  checked><label for='section-78affbdd-a28f-4dc3-b42b-067338574062' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-1473a144-a503-471c-9cb5-e656c18205db' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1473a144-a503-471c-9cb5-e656c18205db' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-def3e293-01ab-4d1d-bfa5-8ed9cb9d93c3' class='xr-var-data-in' type='checkbox'><label for='data-def3e293-01ab-4d1d-bfa5-8ed9cb9d93c3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Brain-...</div><input id='attrs-f3a3076a-2224-4463-a0ee-c1793136fedf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f3a3076a-2224-4463-a0ee-c1793136fedf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7d5fdc5c-255d-4043-ba43-2e20792e3b99' class='xr-var-data-in' type='checkbox'><label for='data-7d5fdc5c-255d-4043-ba43-2e20792e3b99' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Left-Lateral-Ventricle&#x27;, &#x27;Brain-Stem&#x27;],
          dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-350cb4e3-40fa-4316-97d2-2146955eb9c8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-350cb4e3-40fa-4316-97d2-2146955eb9c8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-94822e29-00a4-467b-9398-1d1dff6ed832' class='xr-var-data-in' type='checkbox'><label for='data-94822e29-00a4-467b-9398-1d1dff6ed832' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-bf6bcf2e-e2c6-42c8-bb09-77e06714a744' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bf6bcf2e-e2c6-42c8-bb09-77e06714a744' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4f5fc871-c727-4e5d-b092-aa0a35ce4a9a' class='xr-var-data-in' type='checkbox'><label for='data-4f5fc871-c727-4e5d-b092-aa0a35ce4a9a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-99e0dc1f-46b5-42fe-b338-1d31a0319acc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-99e0dc1f-46b5-42fe-b338-1d31a0319acc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9318f207-9ade-41ff-ad96-2b029032c057' class='xr-var-data-in' type='checkbox'><label for='data-9318f207-9ade-41ff-ad96-2b029032c057' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-dc9d8f99-c648-4a14-ba83-a5ee68b6359f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dc9d8f99-c648-4a14-ba83-a5ee68b6359f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-50bbf60a-fd00-4681-b9ef-0f412d02d297' class='xr-var-data-in' type='checkbox'><label for='data-50bbf60a-fd00-4681-b9ef-0f412d02d297' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e88feb5c-e700-42f6-8b7e-ccffe7c913f4' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e88feb5c-e700-42f6-8b7e-ccffe7c913f4' class='xr-section-summary' >Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subjects</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-f7890302-b30a-44a2-923f-9357261f770a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f7890302-b30a-44a2-923f-9357261f770a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a190826c-d78b-4a62-8830-cf832b47a371' class='xr-var-data-in' type='checkbox'><label for='data-a190826c-d78b-4a62-8830-cf832b47a371' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 4.049e+03 ... 2.092e+04</div><input id='attrs-c75101a2-f854-40ff-a2eb-9e0ec9bd6de3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c75101a2-f854-40ff-a2eb-9e0ec9bd6de3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d73c5341-5a6c-4eef-8d59-97053b0f4957' class='xr-var-data-in' type='checkbox'><label for='data-d73c5341-5a6c-4eef-8d59-97053b0f4957' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1686.7,  4049.4, 20663.2],
           [ 1371.1,  9312.6, 19954. ],
           [ 1414.8,  8972.6, 21645.2],
           [ 1830.6,  6798.6, 20790.6],
           [ 1642.4,  6112.5, 17692.6],
           [ 2108.4,  7076.4, 20996.8],
           [ 2023.1,  4862.2, 20964.9],
           [ 2193.4,  9931.7, 21339.8],
           [ 1086. ,  6479.5, 18517.9],
           [ 1604.9,  5890.9, 20919.9]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 17.58 19.88</div><input id='attrs-6a65dd06-4178-4c17-923a-f02851afe9c1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6a65dd06-4178-4c17-923a-f02851afe9c1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c5f33739-e96e-493e-ac02-89e853936aa9' class='xr-var-data-in' type='checkbox'><label for='data-c5f33739-e96e-493e-ac02-89e853936aa9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           [31.39],
           [13.58],
           [29.21],
           [15.92],
           [21.46],
           [17.58],
           [19.88]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;1&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;AnnArbor_a&#x27;</div><input id='attrs-9cf0f4b3-3f2b-4f6b-afc8-8e27774fecc8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9cf0f4b3-3f2b-4f6b-afc8-8e27774fecc8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-15ad4591-6fb6-4341-a3fe-203b05c812d5' class='xr-var-data-in' type='checkbox'><label for='data-15ad4591-6fb6-4341-a3fe-203b05c812d5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.8117 -0.5312 ... 0.2079 0.1194</div><input id='attrs-0b02a59a-0e16-4d50-b4c3-13da359bd2b7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0b02a59a-0e16-4d50-b4c3-13da359bd2b7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e6462377-3924-4377-a3fc-4604ced00468' class='xr-var-data-in' type='checkbox'><label for='data-e6462377-3924-4377-a3fc-4604ced00468' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.81166894, -0.53119164,  0.01908509],
           [ 0.78680367,  1.06617572, -0.25787247],
           [ 0.2889267 ,  0.37746293,  0.40266401],
           [ 0.69900651, -0.2359351 ,  0.06877881],
           [ 1.3694996 ,  0.61294383, -1.14123688],
           [ 1.15840188, -0.05051823,  0.14936129],
           [ 1.72802298,  0.19915215,  0.13707421],
           [ 1.65960365,  1.02988645,  0.28345893],
           [ 0.46779484,  0.47024978, -0.81888426],
           [ 1.00173737,  0.20792512,  0.11944267]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-271.6 -890.8 ... 2.482e+04</div><input id='attrs-82e670d5-4be1-49a9-ac38-09582a653112' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-82e670d5-4be1-49a9-ac38-09582a653112' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c27b8be3-f5d2-4083-affa-013e510e9119' class='xr-var-data-in' type='checkbox'><label for='data-c27b8be3-f5d2-4083-affa-013e510e9119' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-2.71595586e+02, -8.90802014e+02,  1.64038750e+04],
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
            [ 2.11751725e+03,  1.22643222e+04,  2.48246311e+04]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-7.929 -9.458 ... -9.338 -8.774</div><input id='attrs-491b5f81-9c15-4f83-ac27-058dbe3b9abe' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-491b5f81-9c15-4f83-ac27-058dbe3b9abe' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-37c6ae84-874a-46f5-9005-cb850da99889' class='xr-var-data-in' type='checkbox'><label for='data-37c6ae84-874a-46f5-9005-cb850da99889' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-7.92942478, -9.45752864, -8.76679912],
           [-7.90939993, -9.88466   , -8.79986882],
           [-7.64185346, -9.3877781 , -8.84768597],
           [-7.84448195, -9.34443443, -8.76898235],
           [-8.53756263, -9.50407281, -9.417834  ],
           [-8.27106187, -9.3178153 , -8.77777122],
           [-9.09286242, -9.33608621, -8.77601599],
           [-8.9770708 , -9.84668649, -8.80679265],
           [-7.70927301, -9.42684917, -9.10190589],
           [-8.10163664, -9.33793895, -8.77375215]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-25.7 5.861e+03 ... 2.287e+04</div><input id='attrs-bd9938cc-b903-44ed-96ab-49f7b0cb2cb1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bd9938cc-b903-44ed-96ab-49f7b0cb2cb1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0262524f-92a4-4464-bafd-61025220c422' class='xr-var-data-in' type='checkbox'><label for='data-0262524f-92a4-4464-bafd-61025220c422' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[  -25.7003655 ,  5860.55118121, 22572.74728684],
           [  358.17774376,  1149.18640639, 18106.01807573],
           [ -260.76423555,  8784.72863436, 15327.1115006 ],
           [ 2098.67004855,  4175.65003437, 16734.83922582],
           [ -292.97591623,  5459.12761497, 19413.64262776],
           [ 1517.46940692,  3678.28988828, 22842.3340228 ],
           [ 1009.94223503,  9737.61068149, 20582.57383153],
           [   86.73850127, 10466.23925775, 20159.86231803],
           [ -324.06600087, 12249.38110837, 20302.89953445],
           [  599.13110208,  -911.37801492, 22866.71104426]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.9437 0.05581 ... 2.205 0.9964</div><input id='attrs-2aa89fbe-ed93-4523-bd0b-295cbcccea0f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2aa89fbe-ed93-4523-bd0b-295cbcccea0f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ca42f94e-ff6a-4481-9048-1ac035c65797' class='xr-var-data-in' type='checkbox'><label for='data-ca42f94e-ff6a-4481-9048-1ac035c65797' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-9.43666865e-01,  5.58070501e-02,  7.79454555e-01,
            -3.25480412e-04,  8.07756061e+00, -9.63496588e-01,
             1.09188412e+03,  5.66311467e-02,  6.30696821e-02,
             1.96349659e+00,  8.64052887e-01],
           [-1.15207219e+00,  5.02782931e-02,  7.30631249e-01,
            -3.76018410e-03,  9.79099158e+00, -1.18421649e+00,
             6.36724049e+03,  1.22323117e-01,  5.65664161e-05,
             2.18421649e+00,  9.47832829e-01],
           [-1.20186811e+00,  8.31168831e-03,  1.46762698e-01,
            -6.59853255e-04,  9.24131392e+00, -1.20480541e+00,
             3.70350581e+03, -2.28418860e-02,  4.53740737e-01,
             2.20480541e+00,  9.96388221e-01]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y_harmonized</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 4.049e+03 ... 2.092e+04</div><input id='attrs-16fc2783-c464-4086-9cc0-5656f352958e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-16fc2783-c464-4086-9cc0-5656f352958e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e7db7f27-b5c2-4ec9-b295-9f2f1f356328' class='xr-var-data-in' type='checkbox'><label for='data-e7db7f27-b5c2-4ec9-b295-9f2f1f356328' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1686.7,  4049.4, 20663.2],
           [ 1371.1,  9312.6, 19954. ],
           [ 1414.8,  8972.6, 21645.2],
           [ 1830.6,  6798.6, 20790.6],
           [ 1642.4,  6112.5, 17692.6],
           [ 2108.4,  7076.4, 20996.8],
           [ 2023.1,  4862.2, 20964.9],
           [ 2193.4,  9931.7, 21339.8],
           [ 1086. ,  6479.5, 18517.9],
           [ 1604.9,  5890.9, 20919.9]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-9ccf3e7a-8b9e-41ef-a4af-48ff46da5810' class='xr-section-summary-in' type='checkbox'  ><label for='section-9ccf3e7a-8b9e-41ef-a4af-48ff46da5810' class='xr-section-summary' >Indexes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-85957758-fd16-4b55-b31b-b138a7057004' class='xr-index-data-in' type='checkbox'/><label for='index-85957758-fd16-4b55-b31b-b138a7057004' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-3863f092-8b19-44a3-87ac-1d8e277e6436' class='xr-index-data-in' type='checkbox'/><label for='index-3863f092-8b19-44a3-87ac-1d8e277e6436' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;, &#x27;Left-Lateral-Ventricle&#x27;, &#x27;Brain-Stem&#x27;], dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-ed1668c7-554c-48d2-9c68-4f19b2c5dfd9' class='xr-index-data-in' type='checkbox'/><label for='index-ed1668c7-554c-48d2-9c68-4f19b2c5dfd9' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-5a858ff4-764f-4caf-afc2-0b4a02104ba6' class='xr-index-data-in' type='checkbox'/><label for='index-5a858ff4-764f-4caf-afc2-0b4a02104ba6' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>centile</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-e985dffd-262a-4b73-8e11-7ba0eee6d601' class='xr-index-data-in' type='checkbox'/><label for='index-e985dffd-262a-4b73-8e11-7ba0eee6d601' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0.05, 0.25, 0.5, 0.75, 0.95], dtype=&#x27;float64&#x27;, name=&#x27;centile&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-424dd252-8e93-4a49-82a5-f90a48987dd9' class='xr-index-data-in' type='checkbox'/><label for='index-424dd252-8e93-4a49-82a5-f90a48987dd9' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-c66293b7-171d-49b2-a302-f8eabe55ba05' class='xr-section-summary-in' type='checkbox'  checked><label for='section-c66293b7-171d-49b2-a302-f8eabe55ba05' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{&#x27;sex&#x27;: [&#x27;0&#x27;, &#x27;1&#x27;], &#x27;site&#x27;: [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x306987600&gt;, {&#x27;sex&#x27;: {&#x27;0&#x27;: 589, &#x27;1&#x27;: 489}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{&#x27;sex&#x27;: {&#x27;0&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}, &#x27;1&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}}, &#x27;site&#x27;: {&#x27;AnnArbor_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {&#x27;age&#x27;: {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;statistics&#x27; (statistic: 11)&gt; Size: 88B
    array([-9.43666865e-01,  5.58070501e-02,  7.79454555e-01, -3.25480412e-04,
            8.07756061e+00, -9.63496588e-01,  1.09188412e+03,  5.66311467e-02,
            6.30696821e-02,  1.96349659e+00,  8.64052887e-01])
    Coordinates:
        response_vars  &lt;U22 88B &#x27;WM-hypointensities&#x27;
      * statistic      (statistic) &lt;U8 352B &#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'statistics'</div><ul class='xr-dim-list'><li><span class='xr-has-index'>statistic</span>: 11</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-f71baada-3420-45ad-8494-086db673032d' class='xr-array-in' type='checkbox' checked><label for='section-f71baada-3420-45ad-8494-086db673032d' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>-0.9437 0.05581 0.7795 -0.0003255 ... 0.05663 0.06307 1.963 0.8641</span></div><div class='xr-array-data'><pre>array([-9.43666865e-01,  5.58070501e-02,  7.79454555e-01, -3.25480412e-04,
            8.07756061e+00, -9.63496588e-01,  1.09188412e+03,  5.66311467e-02,
            6.30696821e-02,  1.96349659e+00,  8.64052887e-01])</pre></div></div></li><li class='xr-section-item'><input id='section-9be7b9ee-bcdd-4d61-b52e-8bceb1f42ca6' class='xr-section-summary-in' type='checkbox'  checked><label for='section-9be7b9ee-bcdd-4d61-b52e-8bceb1f42ca6' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>response_vars</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27;</div><input id='attrs-af75a17a-b0af-478a-ab97-0f8284100fa8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-af75a17a-b0af-478a-ab97-0f8284100fa8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-84cd89d8-dd39-4d8d-a3b2-9da6c2b5ee6e' class='xr-var-data-in' type='checkbox'><label for='data-84cd89d8-dd39-4d8d-a3b2-9da6c2b5ee6e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(&#x27;WM-hypointensities&#x27;, dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-ef8525a1-763f-49d1-bee3-51c575ca1556' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ef8525a1-763f-49d1-bee3-51c575ca1556' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8874205a-25f9-4478-86d3-6935a9c6aab5' class='xr-var-data-in' type='checkbox'><label for='data-8874205a-25f9-4478-86d3-6935a9c6aab5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-955d1514-9762-440e-9004-7e156fbbe51b' class='xr-section-summary-in' type='checkbox'  ><label for='section-955d1514-9762-440e-9004-7e156fbbe51b' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-bf181f90-c9fd-41a0-a647-49a248056949' class='xr-index-data-in' type='checkbox'/><label for='index-bf181f90-c9fd-41a0-a647-49a248056949' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-6614e705-9944-4492-a892-962f353a129c' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-6614e705-9944-4492-a892-962f353a129c' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



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

Pre-processing and split datasets
---------------------------------

Sometimes we have a dataset that is pre-split into train and test, and
we want to use that exact data split to fit the model. We can then load
the data into two NormData objects, but we have to make sure that the
two datasets are compatible. This will ensure that the fitted model is
applicable to both of them.

.. code:: ipython3

    # Download an example dataset:
    data = pd.read_csv(
        "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
    )
    # Create an arbitrary split as a placeholder for a predefined split ()
    train, test = train_test_split(data, test_size=100)


.. code:: ipython3

    # specify the column names to use
    covariates = ["age"]
    batch_effects = ["sex", "site"]
    response_vars = ["WM-hypointensities", "Left-Lateral-Ventricle", "Brain-Stem"]
    
    # create NormData objects
    norm_train = NormData.from_dataframe(
        name="train", dataframe=train, covariates=covariates, batch_effects=batch_effects, response_vars=response_vars
    )
    norm_test = NormData.from_dataframe(
        name="test", dataframe=test, covariates=covariates, batch_effects=batch_effects, response_vars=response_vars
    )


.. parsed-literal::

    Process: 28421 - 2025-06-30 13:54:25 - Dataset "train" created.
        - 978 observations
        - 978 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        
    Process: 28421 - 2025-06-30 13:54:25 - Dataset "test" created.
        - 100 observations
        - 100 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (2)
    	site (18)
        


.. parsed-literal::

    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 28421 - 2025-06-30 13:54:25 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)
    /opt/anaconda3/envs/uv_refactor/lib/python3.12/site-packages/pcntoolkit/util/output.py:218: UserWarning: Process: 28421 - 2025-06-30 13:54:25 - remove_Nan is set to False. Ensure your data does not contain NaNs in critical columns, or handle them appropriately.
      warnings.warn(message)


.. code:: ipython3

    # Should print false, because the train and test split do not contain the same sites
    print(norm_train.check_compatibility(norm_test))


.. parsed-literal::

    False


.. code:: ipython3

    norm_train.check_compatibility(norm_test)




.. parsed-literal::

    False



.. code:: ipython3

    norm_train.make_compatible(norm_test)

.. code:: ipython3

    norm_train.check_compatibility(norm_test)




.. parsed-literal::

    True


