`Predictive Clinical Neuroscience Toolkit <https://github.com/amarquand/PCNtoolkit>`__
======================================================================================

Hierarchical Bayesian Regression Normative Modelling and Transfer onto unseen site.
===================================================================================

This notebook will go through basic data preparation (training and
testing set, `see Saige’s
tutorial <https://github.com/predictive-clinical-neuroscience/PCNtoolkit-demo/blob/main/tutorials/BLR_protocol/BLR_normativemodel_protocol.ipynb>`__
on Normative Modelling for more detail), the actual training of the
models, and will finally describe how to transfer the trained models
onto unseen sites.

Created by `Saige Rutherford <https://twitter.com/being_saige>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

adapted/edited by Andre Marquand and Pierre Berthet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step 0: Install necessary libraries & grab data files
-----------------------------------------------------

.. code:: ipython3

    !pip install pcntoolkit
    !pip install nutpie


.. parsed-literal::

    Collecting https://github.com/amarquand/PCNtoolkit/archive/dev.zip
      Downloading https://github.com/amarquand/PCNtoolkit/archive/dev.zip
    [2K     [32m\[0m [32m64.9 MB[0m [31m15.9 MB/s[0m [33m0:00:05[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    Collecting bspline<0.2.0,>=0.1.1 (from pcntoolkit==0.31.0)
      Downloading bspline-0.1.1.tar.gz (84 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m84.2/84.2 kB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
    Collecting matplotlib<4.0.0,>=3.9.2 (from pcntoolkit==0.31.0)
      Downloading matplotlib-3.9.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
    Requirement already satisfied: nibabel<6.0.0,>=5.3.1 in /usr/local/lib/python3.10/dist-packages (from pcntoolkit==0.31.0) (5.3.2)
    Requirement already satisfied: numpy<2.0,>=1.26 in /usr/local/lib/python3.10/dist-packages (from pcntoolkit==0.31.0) (1.26.4)
    Requirement already satisfied: pymc<6.0.0,>=5.18.0 in /usr/local/lib/python3.10/dist-packages (from pcntoolkit==0.31.0) (5.18.0)
    Requirement already satisfied: scikit-learn<2.0.0,>=1.5.2 in /usr/local/lib/python3.10/dist-packages (from pcntoolkit==0.31.0) (1.5.2)
    Requirement already satisfied: scipy<2.0,>=1.12 in /usr/local/lib/python3.10/dist-packages (from pcntoolkit==0.31.0) (1.13.1)
    Requirement already satisfied: seaborn<0.14.0,>=0.13.2 in /usr/local/lib/python3.10/dist-packages (from pcntoolkit==0.31.0) (0.13.2)
    Requirement already satisfied: six<2.0.0,>=1.16.0 in /usr/local/lib/python3.10/dist-packages (from pcntoolkit==0.31.0) (1.16.0)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4.0.0,>=3.9.2->pcntoolkit==0.31.0) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4.0.0,>=3.9.2->pcntoolkit==0.31.0) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4.0.0,>=3.9.2->pcntoolkit==0.31.0) (4.54.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4.0.0,>=3.9.2->pcntoolkit==0.31.0) (1.4.7)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4.0.0,>=3.9.2->pcntoolkit==0.31.0) (24.2)
    Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4.0.0,>=3.9.2->pcntoolkit==0.31.0) (11.0.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4.0.0,>=3.9.2->pcntoolkit==0.31.0) (3.2.0)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4.0.0,>=3.9.2->pcntoolkit==0.31.0) (2.8.2)
    Requirement already satisfied: importlib-resources>=5.12 in /usr/local/lib/python3.10/dist-packages (from nibabel<6.0.0,>=5.3.1->pcntoolkit==0.31.0) (6.4.5)
    Requirement already satisfied: typing-extensions>=4.6 in /usr/local/lib/python3.10/dist-packages (from nibabel<6.0.0,>=5.3.1->pcntoolkit==0.31.0) (4.12.2)
    Requirement already satisfied: arviz>=0.13.0 in /usr/local/lib/python3.10/dist-packages (from pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (0.20.0)
    Requirement already satisfied: cachetools>=4.2.1 in /usr/local/lib/python3.10/dist-packages (from pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (5.5.0)
    Requirement already satisfied: cloudpickle in /usr/local/lib/python3.10/dist-packages (from pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (3.1.0)
    Requirement already satisfied: pandas>=0.24.0 in /usr/local/lib/python3.10/dist-packages (from pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (2.2.2)
    Requirement already satisfied: pytensor<2.26,>=2.25.1 in /usr/local/lib/python3.10/dist-packages (from pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (2.25.5)
    Requirement already satisfied: rich>=13.7.1 in /usr/local/lib/python3.10/dist-packages (from pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (13.9.4)
    Requirement already satisfied: threadpoolctl<4.0.0,>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (3.5.0)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn<2.0.0,>=1.5.2->pcntoolkit==0.31.0) (1.4.2)
    Requirement already satisfied: setuptools>=60.0.0 in /usr/local/lib/python3.10/dist-packages (from arviz>=0.13.0->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (75.1.0)
    Requirement already satisfied: xarray>=2022.6.0 in /usr/local/lib/python3.10/dist-packages (from arviz>=0.13.0->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (2024.10.0)
    Requirement already satisfied: h5netcdf>=1.0.2 in /usr/local/lib/python3.10/dist-packages (from arviz>=0.13.0->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (1.4.1)
    Requirement already satisfied: xarray-einstats>=0.3 in /usr/local/lib/python3.10/dist-packages (from arviz>=0.13.0->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (0.8.0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.24.0->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.24.0->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (2024.2)
    Requirement already satisfied: filelock>=3.15 in /usr/local/lib/python3.10/dist-packages (from pytensor<2.26,>=2.25.1->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (3.16.1)
    Requirement already satisfied: etuples in /usr/local/lib/python3.10/dist-packages (from pytensor<2.26,>=2.25.1->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (0.3.9)
    Requirement already satisfied: logical-unification in /usr/local/lib/python3.10/dist-packages (from pytensor<2.26,>=2.25.1->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (0.4.6)
    Requirement already satisfied: miniKanren in /usr/local/lib/python3.10/dist-packages (from pytensor<2.26,>=2.25.1->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (1.0.3)
    Requirement already satisfied: cons in /usr/local/lib/python3.10/dist-packages (from pytensor<2.26,>=2.25.1->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (0.4.6)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=13.7.1->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (3.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=13.7.1->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (2.18.0)
    Requirement already satisfied: h5py in /usr/local/lib/python3.10/dist-packages (from h5netcdf>=1.0.2->arviz>=0.13.0->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (3.12.1)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=13.7.1->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (0.1.2)
    Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from logical-unification->pytensor<2.26,>=2.25.1->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (0.12.1)
    Requirement already satisfied: multipledispatch in /usr/local/lib/python3.10/dist-packages (from logical-unification->pytensor<2.26,>=2.25.1->pymc<6.0.0,>=5.18.0->pcntoolkit==0.31.0) (1.0.0)
    Downloading matplotlib-3.9.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m8.3/8.3 MB[0m [31m55.1 MB/s[0m eta [36m0:00:00[0m
    [?25hBuilding wheels for collected packages: pcntoolkit, bspline
      Building wheel for pcntoolkit (pyproject.toml) ... [?25l[?25hdone
      Created wheel for pcntoolkit: filename=pcntoolkit-0.31.0-py3-none-any.whl size=114835 sha256=40635c10c24ccf2c319ee965aaf1038272cd5578f14d9cb3dd14598ddab31d00
      Stored in directory: /tmp/pip-ephem-wheel-cache-f502unec/wheels/9e/c4/29/3bca3a5facf8ef69b8622461d8520d24a19d3745aefa093d1e
      Building wheel for bspline (setup.py) ... [?25l[?25hdone
      Created wheel for bspline: filename=bspline-0.1.1-py3-none-any.whl size=84482 sha256=150d24f295ccda92c9789d421e52c3858d43c66874deec4a463a87b4e5533448
      Stored in directory: /root/.cache/pip/wheels/3c/ab/0a/70927853a6d9166bc777922736063a6f99c43a327c802f9326
    Successfully built pcntoolkit bspline
    Installing collected packages: bspline, matplotlib, pcntoolkit
      Attempting uninstall: matplotlib
        Found existing installation: matplotlib 3.8.0
        Uninstalling matplotlib-3.8.0:
          Successfully uninstalled matplotlib-3.8.0
    Successfully installed bspline-0.1.1 matplotlib-3.9.2 pcntoolkit-0.31.0
    Collecting nutpie
      Downloading nutpie-0.13.2-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (5.4 kB)
    Requirement already satisfied: pyarrow>=12.0.0 in /usr/local/lib/python3.10/dist-packages (from nutpie) (17.0.0)
    Requirement already satisfied: pandas>=2.0 in /usr/local/lib/python3.10/dist-packages (from nutpie) (2.2.2)
    Requirement already satisfied: xarray>=2023.6.0 in /usr/local/lib/python3.10/dist-packages (from nutpie) (2024.10.0)
    Requirement already satisfied: arviz>=0.15.0 in /usr/local/lib/python3.10/dist-packages (from nutpie) (0.20.0)
    Requirement already satisfied: setuptools>=60.0.0 in /usr/local/lib/python3.10/dist-packages (from arviz>=0.15.0->nutpie) (75.1.0)
    Requirement already satisfied: matplotlib>=3.5 in /usr/local/lib/python3.10/dist-packages (from arviz>=0.15.0->nutpie) (3.9.2)
    Requirement already satisfied: numpy>=1.23.0 in /usr/local/lib/python3.10/dist-packages (from arviz>=0.15.0->nutpie) (1.26.4)
    Requirement already satisfied: scipy>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from arviz>=0.15.0->nutpie) (1.13.1)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from arviz>=0.15.0->nutpie) (24.2)
    Requirement already satisfied: h5netcdf>=1.0.2 in /usr/local/lib/python3.10/dist-packages (from arviz>=0.15.0->nutpie) (1.4.1)
    Requirement already satisfied: typing-extensions>=4.1.0 in /usr/local/lib/python3.10/dist-packages (from arviz>=0.15.0->nutpie) (4.12.2)
    Requirement already satisfied: xarray-einstats>=0.3 in /usr/local/lib/python3.10/dist-packages (from arviz>=0.15.0->nutpie) (0.8.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas>=2.0->nutpie) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=2.0->nutpie) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas>=2.0->nutpie) (2024.2)
    Requirement already satisfied: h5py in /usr/local/lib/python3.10/dist-packages (from h5netcdf>=1.0.2->arviz>=0.15.0->nutpie) (3.12.1)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.5->arviz>=0.15.0->nutpie) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.5->arviz>=0.15.0->nutpie) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.5->arviz>=0.15.0->nutpie) (4.54.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.5->arviz>=0.15.0->nutpie) (1.4.7)
    Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.5->arviz>=0.15.0->nutpie) (11.0.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.5->arviz>=0.15.0->nutpie) (3.2.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas>=2.0->nutpie) (1.16.0)
    Downloading nutpie-0.13.2-cp310-cp310-manylinux_2_28_x86_64.whl (1.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.5/1.5 MB[0m [31m16.5 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: nutpie
    Successfully installed nutpie-0.13.2


For this tutorial we will use data from the `Functional Connectom
Project FCON1000 <http://fcon_1000.projects.nitrc.org/>`__ to create a
multi-site dataset.

The dataset contains some cortical measures (eg thickness), processed by
Freesurfer 6.0, and some covariates (eg age, site, gender).

First we import the required package, and create a working directory.

.. code:: ipython3

    import os
    import pandas as pd
    import pcntoolkit as ptk
    import numpy as np
    import pickle
    from matplotlib import pyplot as plt

.. code:: ipython3

    processing_dir = "HBR_demo"    # replace with desired working directory
    if not os.path.isdir(processing_dir):
        os.makedirs(processing_dir)
    os.chdir(processing_dir)
    processing_dir = os.getcwd()

Overview
^^^^^^^^

Here we get the FCON dataset, remove the ICBM site for later transfer,
assign some site id to the different scanner sites and print an overview
of the left hemisphere mean raw cortical thickness as a function of age,
color coded by the various sites:

.. code:: ipython3

    fcon = pd.read_csv('https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000.csv')
    
    # extract the ICBM site for transfer
    icbm = fcon.loc[fcon['site'] == 'ICBM']
    icbm['sitenum'] = 0
    
    # remove from the training set (also Pittsburgh because it only has 3 samples)
    fcon = fcon.loc[fcon['site'] != 'ICBM']
    fcon = fcon.loc[fcon['site'] != 'Pittsburgh']
    
    sites = fcon['site'].unique()
    fcon['sitenum'] = 0
    
    f, ax = plt.subplots(figsize=(12, 12))
    
    for i,s in enumerate(sites):
        idx = fcon['site'] == s
        fcon['sitenum'].loc[idx] = i
    
        print('site',s, sum(idx))
        ax.scatter(fcon['age'].loc[idx], fcon['lh_MeanThickness_thickness'].loc[idx])
    
    ax.legend(sites)
    ax.set_ylabel('LH mean cortical thickness [mm]')
    ax.set_xlabel('age')



.. parsed-literal::

    <ipython-input-4-a7d14b9f2beb>:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      icbm['sitenum'] = 0
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      fcon['sitenum'].loc[idx] = i
    <ipython-input-4-a7d14b9f2beb>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      fcon['sitenum'].loc[idx] = i


.. parsed-literal::

    site AnnArbor_a 24
    site AnnArbor_b 32
    site Atlanta 28
    site Baltimore 23
    site Bangor 20
    site Beijing_Zang 198
    site Berlin_Margulies 26
    site Cambridge_Buckner 198
    site Cleveland 31
    site Leiden_2180 12
    site Leiden_2200 19
    site Milwaukee_b 46
    site Munchen 15
    site NewYork_a 83
    site NewYork_a_ADHD 25
    site Newark 19
    site Oulu 102
    site Oxford 22
    site PaloAlto 17
    site Queensland 19
    site SaintLouis 31




.. parsed-literal::

    Text(0.5, 0, 'age')




.. image:: HBR_NormativeModel_FCONdata_Tutorial_files/HBR_NormativeModel_FCONdata_Tutorial_10_3.png


Step 1: Prepare training and testing sets
-----------------------------------------

Then we randomly split half of the samples (participants) to be either
in the training or in the testing samples. We do this for the remaing
FCON dataset and for the ICBM data. The transfer function will also
require a training and a test sample.

The numbers of samples per sites used for training and for testing are
then displayed.

.. code:: ipython3

    tr = np.random.uniform(size=fcon.shape[0]) > 0.5
    te = ~tr
    
    fcon_tr = fcon.loc[tr]
    fcon_te = fcon.loc[te]
    
    tr = np.random.uniform(size=icbm.shape[0]) > 0.5
    te = ~tr
    
    icbm_tr = icbm.loc[tr]
    icbm_te = icbm.loc[te]
    
    print('sample size check')
    for i,s in enumerate(sites):
        idx = fcon_tr['site'] == s
        idxte = fcon_te['site'] == s
        print(i,s, sum(idx), sum(idxte))
    
    fcon_tr.to_csv(processing_dir + '/fcon1000_tr.csv')
    fcon_te.to_csv(processing_dir + '/fcon1000_te.csv')
    icbm_tr.to_csv(processing_dir + '/fcon1000_icbm_tr.csv')
    icbm_te.to_csv(processing_dir + '/fcon1000_icbm_te.csv')


.. parsed-literal::

    sample size check
    0 AnnArbor_a 10 14
    1 AnnArbor_b 19 13
    2 Atlanta 12 16
    3 Baltimore 12 11
    4 Bangor 10 10
    5 Beijing_Zang 91 107
    6 Berlin_Margulies 9 17
    7 Cambridge_Buckner 96 102
    8 Cleveland 13 18
    9 Leiden_2180 5 7
    10 Leiden_2200 11 8
    11 Milwaukee_b 18 28
    12 Munchen 9 6
    13 NewYork_a 38 45
    14 NewYork_a_ADHD 15 10
    15 Newark 9 10
    16 Oulu 50 52
    17 Oxford 9 13
    18 PaloAlto 8 9
    19 Queensland 10 9
    20 SaintLouis 18 13


Otherwise you can just load these pre defined subsets:

.. code:: ipython3

    # Optional
    #fcon_tr = pd.read_csv('https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000_tr.csv')
    #fcon_te = pd.read_csv('https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000_te.csv')
    #icbm_tr = pd.read_csv('https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000_icbm_tr.csv')
    #icbm_te = pd.read_csv('https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000_icbm_te.csv')

Step 2: Configure HBR inputs: covariates, measures and batch effects
--------------------------------------------------------------------

We will here only use the mean cortical thickness for the Right and Left
hemisphere: two idps.

.. code:: ipython3

    idps = ['rh_MeanThickness_thickness','lh_MeanThickness_thickness']

As input to the model, we need covariates (used to describe predictable
source of variability (fixed effects), here ‘age’), measures (here
cortical thickness on two idps), and batch effects (random source of
variability, here ‘scanner site’ and ‘sex’).

``X`` corresponds to the covariate(s)

``Y`` to the measure(s)

``batch_effects`` to the random effects

We need these values both for the training (``_train``) and for the
testing set (``_test``).

.. code:: ipython3

    X_train = (fcon_tr['age']/100).to_numpy(dtype=float)
    Y_train = fcon_tr[idps].to_numpy(dtype=float)
    
    # configure batch effects for site and sex
    #batch_effects_train = fcon_tr[['sitenum','sex']].to_numpy(dtype=int)
    
    # or only site
    batch_effects_train = fcon_tr[['sitenum']].to_numpy(dtype=int)
    
    with open('X_train.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(X_train), file)
    with open('Y_train.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(Y_train), file)
    with open('trbefile.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(batch_effects_train), file)
    
    
    X_test = (fcon_te['age']/100).to_numpy(dtype=float)
    Y_test = fcon_te[idps].to_numpy(dtype=float)
    #batch_effects_test = fcon_te[['sitenum','sex']].to_numpy(dtype=int)
    batch_effects_test = fcon_te[['sitenum']].to_numpy(dtype=int)
    
    with open('X_test.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(X_test), file)
    with open('Y_test.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(Y_test), file)
    with open('tsbefile.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(batch_effects_test), file)
    
    # a simple function to quickly load pickle files
    def ldpkl(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)

.. code:: ipython3

    batch_effects_test




.. parsed-literal::

    array([[ 0],
           [ 0],
           [ 0],
           [ 0],
           [ 0],
           [ 0],
           [ 0],
           [ 0],
           [ 0],
           [ 0],
           [ 0],
           [ 0],
           [ 0],
           [ 0],
           [ 1],
           [ 1],
           [ 1],
           [ 1],
           [ 1],
           [ 1],
           [ 1],
           [ 1],
           [ 1],
           [ 1],
           [ 1],
           [ 1],
           [ 1],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 2],
           [ 3],
           [ 3],
           [ 3],
           [ 3],
           [ 3],
           [ 3],
           [ 3],
           [ 3],
           [ 3],
           [ 3],
           [ 3],
           [ 4],
           [ 4],
           [ 4],
           [ 4],
           [ 4],
           [ 4],
           [ 4],
           [ 4],
           [ 4],
           [ 4],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 5],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 6],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 7],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 8],
           [ 9],
           [ 9],
           [ 9],
           [ 9],
           [ 9],
           [ 9],
           [ 9],
           [10],
           [10],
           [10],
           [10],
           [10],
           [10],
           [10],
           [10],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [11],
           [12],
           [12],
           [12],
           [12],
           [12],
           [12],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [13],
           [14],
           [14],
           [14],
           [14],
           [14],
           [14],
           [14],
           [14],
           [14],
           [14],
           [15],
           [15],
           [15],
           [15],
           [15],
           [15],
           [15],
           [15],
           [15],
           [15],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [16],
           [17],
           [17],
           [17],
           [17],
           [17],
           [17],
           [17],
           [17],
           [17],
           [17],
           [17],
           [17],
           [17],
           [18],
           [18],
           [18],
           [18],
           [18],
           [18],
           [18],
           [18],
           [18],
           [19],
           [19],
           [19],
           [19],
           [19],
           [19],
           [19],
           [19],
           [19],
           [20],
           [20],
           [20],
           [20],
           [20],
           [20],
           [20],
           [20],
           [20],
           [20],
           [20],
           [20],
           [20]])



Step 3: Files and Folders grooming
----------------------------------

.. code:: ipython3

    respfile = os.path.join(processing_dir, 'Y_train.pkl')       # measurements  (eg cortical thickness) of the training samples (columns: the various features/ROIs, rows: observations or subjects)
    covfile = os.path.join(processing_dir, 'X_train.pkl')        # covariates (eg age) the training samples (columns: covariates, rows: observations or subjects)
    
    testrespfile_path = os.path.join(processing_dir, 'Y_test.pkl')       # measurements  for the testing samples
    testcovfile_path = os.path.join(processing_dir, 'X_test.pkl')        # covariate file for the testing samples
    
    trbefile = os.path.join(processing_dir, 'trbefile.pkl')      # training batch effects file (eg scanner_id, gender)  (columns: the various batch effects, rows: observations or subjects)
    tsbefile = os.path.join(processing_dir, 'tsbefile.pkl')      # testing batch effects file
    
    output_path = os.path.join(processing_dir, 'Models/')    #  output path, where the models will be written
    log_dir = os.path.join(processing_dir, 'log/')           #
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    
    outputsuffix = '_estimate'      # a string to name the output files, of use only to you, so adapt it for your needs.

Step 4: Estimating the models
-----------------------------

Now we have everything ready to estimate the normative models. The
``estimate`` function only needs the training and testing sets, each
divided in three datasets: covariates, measures and batch effects. We
obviously specify ``alg=hbr`` to use the hierarchical bayesian
regression method, well suited for the multi sites datasets. The
remaining arguments are basic data management: where the models, logs,
and output files will be written and how they will be named.

.. code:: ipython3

    ptk.normative.estimate(covfile=covfile,
                           respfile=respfile,
                           tsbefile=tsbefile,
                           trbefile=trbefile,
                           inscaler='standardize',
                           outscaler='standardize',
                           linear_mu='True',
                           random_intercept_mu='True',
                           centered_intercept_mu='True',
                           alg='hbr',
                           log_path=log_dir,
                           binary=True,
                           output_path=output_path,
                           testcov= testcovfile_path,
                           testresp = testrespfile_path,
                           outputsuffix=outputsuffix,
                           savemodel=True,
                           nuts_sampler='nutpie')


.. parsed-literal::

    inscaler: standardize
    outscaler: standardize
    Processing data in /content/HBR_demo/Y_train.pkl
    Estimating model  1 of 2



.. raw:: html

    
    <style>
        :root {
            --column-width-1: 40%; /* Progress column width */
            --column-width-2: 15%; /* Chain column width */
            --column-width-3: 15%; /* Divergences column width */
            --column-width-4: 15%; /* Step Size column width */
            --column-width-5: 15%; /* Gradients/Draw column width */
        }
    
        .nutpie {
            max-width: 800px;
            margin: 10px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            //color: #333;
            //background-color: #fff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            font-size: 14px; /* Smaller font size for a more compact look */
        }
        .nutpie table {
            width: 100%;
            border-collapse: collapse; /* Remove any extra space between borders */
        }
        .nutpie th, .nutpie td {
            padding: 8px 10px; /* Reduce padding to make table more compact */
            text-align: left;
            border-bottom: 1px solid #888;
        }
        .nutpie th {
            //background-color: #f0f0f0;
        }
    
        .nutpie th:nth-child(1) { width: var(--column-width-1); }
        .nutpie th:nth-child(2) { width: var(--column-width-2); }
        .nutpie th:nth-child(3) { width: var(--column-width-3); }
        .nutpie th:nth-child(4) { width: var(--column-width-4); }
        .nutpie th:nth-child(5) { width: var(--column-width-5); }
    
        .nutpie progress {
            width: 100%;
            height: 15px; /* Smaller progress bars */
            border-radius: 5px;
        }
        progress::-webkit-progress-bar {
            background-color: #eee;
            border-radius: 5px;
        }
        progress::-webkit-progress-value {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        progress::-moz-progress-bar {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        .nutpie .progress-cell {
            width: 100%;
        }
    
        .nutpie p strong { font-size: 16px; font-weight: bold; }
    
        @media (prefers-color-scheme: dark) {
            .nutpie {
                //color: #ddd;
                //background-color: #1e1e1e;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
            .nutpie table, .nutpie th, .nutpie td {
                border-color: #555;
                color: #ccc;
            }
            .nutpie th {
                background-color: #2a2a2a;
            }
            .nutpie progress::-webkit-progress-bar {
                background-color: #444;
            }
            .nutpie progress::-webkit-progress-value {
                background-color: #3178c6;
            }
            .nutpie progress::-moz-progress-bar {
                background-color: #3178c6;
            }
        }
    </style>




.. raw:: html

    
    <div class="nutpie">
        <p><strong>Sampler Progress</strong></p>
        <p>Total Chains: <span id="total-chains">1</span></p>
        <p>Active Chains: <span id="active-chains">0</span></p>
        <p>
            Finished Chains:
            <span id="active-chains">1</span>
        </p>
        <p>Sampling for now</p>
        <p>
            Estimated Time to Completion:
            <span id="eta">now</span>
        </p>
    
        <progress
            id="total-progress-bar"
            max="1500"
            value="1500">
        </progress>
        <table>
            <thead>
                <tr>
                    <th>Progress</th>
                    <th>Draws</th>
                    <th>Divergences</th>
                    <th>Step Size</th>
                    <th>Gradients/Draw</th>
                </tr>
            </thead>
            <tbody id="chain-details">
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="1500"
                                value="1500">
                            </progress>
                        </td>
                        <td>1500</td>
                        <td>0</td>
                        <td>0.34</td>
                        <td>15</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. parsed-literal::

    Output()


.. parsed-literal::

    Normal



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. parsed-literal::

    Estimating model  2 of 2



.. raw:: html

    
    <style>
        :root {
            --column-width-1: 40%; /* Progress column width */
            --column-width-2: 15%; /* Chain column width */
            --column-width-3: 15%; /* Divergences column width */
            --column-width-4: 15%; /* Step Size column width */
            --column-width-5: 15%; /* Gradients/Draw column width */
        }
    
        .nutpie {
            max-width: 800px;
            margin: 10px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            //color: #333;
            //background-color: #fff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            font-size: 14px; /* Smaller font size for a more compact look */
        }
        .nutpie table {
            width: 100%;
            border-collapse: collapse; /* Remove any extra space between borders */
        }
        .nutpie th, .nutpie td {
            padding: 8px 10px; /* Reduce padding to make table more compact */
            text-align: left;
            border-bottom: 1px solid #888;
        }
        .nutpie th {
            //background-color: #f0f0f0;
        }
    
        .nutpie th:nth-child(1) { width: var(--column-width-1); }
        .nutpie th:nth-child(2) { width: var(--column-width-2); }
        .nutpie th:nth-child(3) { width: var(--column-width-3); }
        .nutpie th:nth-child(4) { width: var(--column-width-4); }
        .nutpie th:nth-child(5) { width: var(--column-width-5); }
    
        .nutpie progress {
            width: 100%;
            height: 15px; /* Smaller progress bars */
            border-radius: 5px;
        }
        progress::-webkit-progress-bar {
            background-color: #eee;
            border-radius: 5px;
        }
        progress::-webkit-progress-value {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        progress::-moz-progress-bar {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        .nutpie .progress-cell {
            width: 100%;
        }
    
        .nutpie p strong { font-size: 16px; font-weight: bold; }
    
        @media (prefers-color-scheme: dark) {
            .nutpie {
                //color: #ddd;
                //background-color: #1e1e1e;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
            .nutpie table, .nutpie th, .nutpie td {
                border-color: #555;
                color: #ccc;
            }
            .nutpie th {
                background-color: #2a2a2a;
            }
            .nutpie progress::-webkit-progress-bar {
                background-color: #444;
            }
            .nutpie progress::-webkit-progress-value {
                background-color: #3178c6;
            }
            .nutpie progress::-moz-progress-bar {
                background-color: #3178c6;
            }
        }
    </style>




.. raw:: html

    
    <div class="nutpie">
        <p><strong>Sampler Progress</strong></p>
        <p>Total Chains: <span id="total-chains">1</span></p>
        <p>Active Chains: <span id="active-chains">0</span></p>
        <p>
            Finished Chains:
            <span id="active-chains">1</span>
        </p>
        <p>Sampling for now</p>
        <p>
            Estimated Time to Completion:
            <span id="eta">now</span>
        </p>
    
        <progress
            id="total-progress-bar"
            max="1500"
            value="1500">
        </progress>
        <table>
            <thead>
                <tr>
                    <th>Progress</th>
                    <th>Draws</th>
                    <th>Divergences</th>
                    <th>Step Size</th>
                    <th>Gradients/Draw</th>
                </tr>
            </thead>
            <tbody id="chain-details">
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="1500"
                                value="1500">
                            </progress>
                        </td>
                        <td>1500</td>
                        <td>0</td>
                        <td>0.33</td>
                        <td>15</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. parsed-literal::

    Normal



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. parsed-literal::

    Saving model meta-data...
    Evaluating the model ...
    Writing outputs ...


Here some analyses can be done, there are also some error metrics that
could be of interest. This is covered in step 6 and in `Saige’s
tutorial <https://github.com/predictive-clinical-neuroscience/PCNtoolkit-demo/blob/main/tutorials/BLR_protocol/BLR_normativemodel_protocol.ipynb>`__
on Normative Modelling.

Step 5: Transfering the models to unseen sites
----------------------------------------------

Similarly to what was done before for the FCON data, we also need to
prepare the ICBM specific data, in order to run the transfer function:
training and testing set of covariates, measures and batch effects:

.. code:: ipython3

    X_adapt = (icbm_tr['age']/100).to_numpy(dtype=float)
    Y_adapt = icbm_tr[idps].to_numpy(dtype=float)
    #batch_effects_adapt = icbm_tr[['sitenum','sex']].to_numpy(dtype=int)
    batch_effects_adapt = icbm_tr[['sitenum']].to_numpy(dtype=int)
    
    with open('X_adaptation.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(X_adapt), file)
    with open('Y_adaptation.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(Y_adapt), file)
    with open('adbefile.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(batch_effects_adapt), file)
    
    # Test data (new dataset)
    X_test_txfr = (icbm_te['age']/100).to_numpy(dtype=float)
    Y_test_txfr = icbm_te[idps].to_numpy(dtype=float)
    #batch_effects_test_txfr = icbm_te[['sitenum','sex']].to_numpy(dtype=int)
    batch_effects_test_txfr = icbm_te[['sitenum']].to_numpy(dtype=int)
    
    with open('X_test_txfr.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(X_test_txfr), file)
    with open('Y_test_txfr.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(Y_test_txfr), file)
    with open('txbefile.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(batch_effects_test_txfr), file)


.. code:: ipython3

    respfile = os.path.join(processing_dir, 'Y_adaptation.pkl')
    covfile = os.path.join(processing_dir, 'X_adaptation.pkl')
    testrespfile_path = os.path.join(processing_dir, 'Y_test_txfr.pkl')
    testcovfile_path = os.path.join(processing_dir, 'X_test_txfr.pkl')
    trbefile = os.path.join(processing_dir, 'adbefile.pkl')
    tsbefile = os.path.join(processing_dir, 'txbefile.pkl')
    
    log_dir = os.path.join(processing_dir, 'log_transfer/')
    output_path = os.path.join(processing_dir, 'Transfer/')
    model_path = os.path.join(processing_dir, 'Models/')  # path to the previously trained models
    outputsuffix = '_transfer'  # suffix added to the output files from the transfer function

Here, the difference is that the transfer function needs a model path,
which points to the models we just trained, and new site data (training
and testing). That is basically the only difference.

.. code:: ipython3

    yhat, s2, z_scores = ptk.normative.transfer(covfile=covfile,
                                                respfile=respfile,
                                                tsbefile=tsbefile,
                                                trbefile=trbefile,
                                                inscaler='standardize',
                                                outscaler='standardize',
                                                linear_mu='True',
                                                random_intercept_mu='True',
                                                centered_intercept_mu='True',
                                                model_path = model_path,
                                                alg='hbr',
                                                log_path=log_dir,
                                                binary=True,
                                                output_path=output_path,
                                                testcov= testcovfile_path,
                                                testresp = testrespfile_path,
                                                outputsuffix=outputsuffix,
                                                savemodel=True,
                                                nuts_sampler='nutpie')


.. parsed-literal::

    Loading data ...
    Using HBR transform...
    Transferring model  1 of 2



.. raw:: html

    
    <style>
        :root {
            --column-width-1: 40%; /* Progress column width */
            --column-width-2: 15%; /* Chain column width */
            --column-width-3: 15%; /* Divergences column width */
            --column-width-4: 15%; /* Step Size column width */
            --column-width-5: 15%; /* Gradients/Draw column width */
        }
    
        .nutpie {
            max-width: 800px;
            margin: 10px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            //color: #333;
            //background-color: #fff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            font-size: 14px; /* Smaller font size for a more compact look */
        }
        .nutpie table {
            width: 100%;
            border-collapse: collapse; /* Remove any extra space between borders */
        }
        .nutpie th, .nutpie td {
            padding: 8px 10px; /* Reduce padding to make table more compact */
            text-align: left;
            border-bottom: 1px solid #888;
        }
        .nutpie th {
            //background-color: #f0f0f0;
        }
    
        .nutpie th:nth-child(1) { width: var(--column-width-1); }
        .nutpie th:nth-child(2) { width: var(--column-width-2); }
        .nutpie th:nth-child(3) { width: var(--column-width-3); }
        .nutpie th:nth-child(4) { width: var(--column-width-4); }
        .nutpie th:nth-child(5) { width: var(--column-width-5); }
    
        .nutpie progress {
            width: 100%;
            height: 15px; /* Smaller progress bars */
            border-radius: 5px;
        }
        progress::-webkit-progress-bar {
            background-color: #eee;
            border-radius: 5px;
        }
        progress::-webkit-progress-value {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        progress::-moz-progress-bar {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        .nutpie .progress-cell {
            width: 100%;
        }
    
        .nutpie p strong { font-size: 16px; font-weight: bold; }
    
        @media (prefers-color-scheme: dark) {
            .nutpie {
                //color: #ddd;
                //background-color: #1e1e1e;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
            .nutpie table, .nutpie th, .nutpie td {
                border-color: #555;
                color: #ccc;
            }
            .nutpie th {
                background-color: #2a2a2a;
            }
            .nutpie progress::-webkit-progress-bar {
                background-color: #444;
            }
            .nutpie progress::-webkit-progress-value {
                background-color: #3178c6;
            }
            .nutpie progress::-moz-progress-bar {
                background-color: #3178c6;
            }
        }
    </style>




.. raw:: html

    
    <div class="nutpie">
        <p><strong>Sampler Progress</strong></p>
        <p>Total Chains: <span id="total-chains">1</span></p>
        <p>Active Chains: <span id="active-chains">0</span></p>
        <p>
            Finished Chains:
            <span id="active-chains">1</span>
        </p>
        <p>Sampling for now</p>
        <p>
            Estimated Time to Completion:
            <span id="eta">now</span>
        </p>
    
        <progress
            id="total-progress-bar"
            max="1500"
            value="1500">
        </progress>
        <table>
            <thead>
                <tr>
                    <th>Progress</th>
                    <th>Draws</th>
                    <th>Divergences</th>
                    <th>Step Size</th>
                    <th>Gradients/Draw</th>
                </tr>
            </thead>
            <tbody id="chain-details">
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="1500"
                                value="1500">
                            </progress>
                        </td>
                        <td>1500</td>
                        <td>2</td>
                        <td>0.47</td>
                        <td>7</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. parsed-literal::

    Using HBR transform...
    Transferring model  2 of 2



.. raw:: html

    
    <style>
        :root {
            --column-width-1: 40%; /* Progress column width */
            --column-width-2: 15%; /* Chain column width */
            --column-width-3: 15%; /* Divergences column width */
            --column-width-4: 15%; /* Step Size column width */
            --column-width-5: 15%; /* Gradients/Draw column width */
        }
    
        .nutpie {
            max-width: 800px;
            margin: 10px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            //color: #333;
            //background-color: #fff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            font-size: 14px; /* Smaller font size for a more compact look */
        }
        .nutpie table {
            width: 100%;
            border-collapse: collapse; /* Remove any extra space between borders */
        }
        .nutpie th, .nutpie td {
            padding: 8px 10px; /* Reduce padding to make table more compact */
            text-align: left;
            border-bottom: 1px solid #888;
        }
        .nutpie th {
            //background-color: #f0f0f0;
        }
    
        .nutpie th:nth-child(1) { width: var(--column-width-1); }
        .nutpie th:nth-child(2) { width: var(--column-width-2); }
        .nutpie th:nth-child(3) { width: var(--column-width-3); }
        .nutpie th:nth-child(4) { width: var(--column-width-4); }
        .nutpie th:nth-child(5) { width: var(--column-width-5); }
    
        .nutpie progress {
            width: 100%;
            height: 15px; /* Smaller progress bars */
            border-radius: 5px;
        }
        progress::-webkit-progress-bar {
            background-color: #eee;
            border-radius: 5px;
        }
        progress::-webkit-progress-value {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        progress::-moz-progress-bar {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        .nutpie .progress-cell {
            width: 100%;
        }
    
        .nutpie p strong { font-size: 16px; font-weight: bold; }
    
        @media (prefers-color-scheme: dark) {
            .nutpie {
                //color: #ddd;
                //background-color: #1e1e1e;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
            .nutpie table, .nutpie th, .nutpie td {
                border-color: #555;
                color: #ccc;
            }
            .nutpie th {
                background-color: #2a2a2a;
            }
            .nutpie progress::-webkit-progress-bar {
                background-color: #444;
            }
            .nutpie progress::-webkit-progress-value {
                background-color: #3178c6;
            }
            .nutpie progress::-moz-progress-bar {
                background-color: #3178c6;
            }
        }
    </style>




.. raw:: html

    
    <div class="nutpie">
        <p><strong>Sampler Progress</strong></p>
        <p>Total Chains: <span id="total-chains">1</span></p>
        <p>Active Chains: <span id="active-chains">0</span></p>
        <p>
            Finished Chains:
            <span id="active-chains">1</span>
        </p>
        <p>Sampling for now</p>
        <p>
            Estimated Time to Completion:
            <span id="eta">now</span>
        </p>
    
        <progress
            id="total-progress-bar"
            max="1500"
            value="1500">
        </progress>
        <table>
            <thead>
                <tr>
                    <th>Progress</th>
                    <th>Draws</th>
                    <th>Divergences</th>
                    <th>Step Size</th>
                    <th>Gradients/Draw</th>
                </tr>
            </thead>
            <tbody id="chain-details">
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="1500"
                                value="1500">
                            </progress>
                        </td>
                        <td>1500</td>
                        <td>1</td>
                        <td>0.40</td>
                        <td>15</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. parsed-literal::

    Evaluating the model ...
    Writing outputs ...


.. code:: ipython3

    output_path




.. parsed-literal::

    '/content/HBR_demo/Transfer/'



.. code:: ipython3

    EV = pd.read_pickle('EXPV_estimate.pkl')
    print(EV)


.. parsed-literal::

              0
    0  0.438215
    1  0.439181


And that is it, you now have models that benefited from prior knowledge
about different scanner sites to learn on unseen sites.

Step 6: Interpreting model performance
--------------------------------------

Output evaluation metrics definitions: \* yhat - predictive mean \* ys2
- predictive variance \* nm - normative model \* Z - deviance scores \*
Rho - Pearson correlation between true and predicted responses \* pRho -
parametric p-value for this correlation \* RMSE - root mean squared
error between true/predicted responses \* SMSE - standardised mean
squared error \* EV - explained variance \* MSLL - mean standardized log
loss \* See page 23 in
http://www.gaussianprocess.org/gpml/chapters/RW2.pdf

