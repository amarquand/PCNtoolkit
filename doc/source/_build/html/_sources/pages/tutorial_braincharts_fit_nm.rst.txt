Braincharts: fit model
------------------------------------

This notebook provides a complete walkthrough for an analysis of
normative modelling in a large sample as described in the accompanying
paper. Note that this script is provided principally for completeness
(e.g. to assist in fitting normative models to new datasets). All
pre-estimated normative models are already provided.

 `Open/Run in Google Colab <https://colab.research.google.com/github/predictive-clinical-neuroscience/braincharts/blob/master/scripts/fit_normative_models.ipynb>`__
 
First, if necessary, we install PCNtoolkit (note: this tutorial requires
at least version 0.20)

.. code:: ipython3

    !pip install pcntoolkit==0.20

Then we import the required libraries

.. code:: ipython3

    import os
    import numpy as np
    import pandas as pd
    import pickle
    from matplotlib import pyplot as plt
    import seaborn as sns
    
    from pcntoolkit.normative import estimate, predict, evaluate
    from pcntoolkit.util.utils import compute_MSLL, create_design_matrix
    from nm_utils import calibration_descriptives, remove_bad_subjects, load_2d

Now, we configure the locations in which the data are stored. You will
need to configure this for your specific installation

**Notes:** - The data are assumed to be in CSV format and will be loaded
as pandas dataframes - Generally the raw data will be in a different
location to the analysis - The data can have arbitrary columns but some
are required by the script, i.e. ‘age’, ‘sex’ and ‘site’, plus the
phenotypes you wish to estimate (see below)

.. code:: ipython3

    # where the raw data are stored
    data_dir = '<path-to-your>/data'
    
    # where the analysis takes place
    root_dir = '<path-to-your>/braincharts'
    out_dir = os.path.join(root_dir,'models','test')
    
    # create the output directory if it does not already exist
    os.makedirs(out_dir, exist_ok=True)

Now we load the data.

We will load one pandas dataframe for the training set and one dataframe
for the test set. We will also filter out low quality scans on the basis
of the Freesurfer `Euler
Characteristic <https://surfer.nmr.mgh.harvard.edu/fswiki/EulerNumber>`__
(EC). This is a proxy for scan quality and is described in the
publications below. Note that this requires the column ‘avg_en’ in the
pandas dataframe, which is simply the average EC of left and right
hemisphere.

We also configrure a list of site ids

**References** - `Kia et al
2021 <https://www.biorxiv.org/content/10.1101/2021.05.28.446120v1.abstract>`__
- `Rosen et al
2018 <https://www.sciencedirect.com/science/article/abs/pii/S1053811917310832?via%3Dihub>`__

.. code:: ipython3

    df_tr = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_tr_mqc.csv'), index_col=0) 
    df_te = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_te_mqc.csv'), index_col=0)
    
    # remove some bad subjects
    df_tr, bad_sub = remove_bad_subjects(df_tr, df_tr)
    df_te, bad_sub = remove_bad_subjects(df_te, df_te)
    
    # extract a list of unique site ids from the training set
    site_ids =  sorted(set(df_tr['site'].to_list()))


.. parsed-literal::

    362 subjects are removed!
    356 subjects are removed!


Configure which models to fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we load the image derived phenotypes (IDPs) which we will process
in this analysis. This is effectively just a list of columns in your
dataframe. Here we estimate normative models for the left hemisphere,
right hemisphere and cortical structures.

.. code:: ipython3

    # load the idps to process
    with open(os.path.join(root_dir,'docs','phenotypes_lh.txt')) as f:
        idp_ids_lh = f.read().splitlines()
    with open(os.path.join(root_dir,'docs','phenotypes_rh.txt')) as f:
        idp_ids_rh = f.read().splitlines()
    with open(os.path.join(root_dir,'docs','phenotypes_sc.txt')) as f:
        idp_ids_sc = f.read().splitlines()
    
    # we choose here to process all idps
    idp_ids = idp_ids_lh + idp_ids_rh + idp_ids_sc
    
    # we could also just specify a list of IDPs
    #idp_ids = ['lh_MeanThickness_thickness', 'rh_MeanThickness_thickness']

Configure model parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, we configure some parameters for the regression model we use to fit
the normative model. Here we will use a ‘warped’ Bayesian linear
regression model. To model non-Gaussianity, we select a sin arcsinh warp
and to model non-linearity, we stick with the default value for the
basis expansion (a cubic b-spline basis set with 5 knot points). Since
we are sticking with the default value, we do not need to specify any
parameters for this, but we do need to specify the limits. We choose to
pad the input by a few years either side of the input range. We will
also set a couple of options that control the estimation of the model

For further details about the likelihood warping approach, see `Fraza et
al
2021 <https://www.biorxiv.org/content/10.1101/2021.04.05.438429v1>`__.

.. code:: ipython3

    # which data columns do we wish to use as covariates? 
    cols_cov = ['age','sex']
    
    # which warping function to use? We can set this to None in order to fit a vanilla Gaussian noise model
    warp =  'WarpSinArcsinh'
    
    # limits for cubic B-spline basis 
    xmin = -5 
    xmax = 110
    
    # Do we want to force the model to be refit every time? 
    force_refit = True
    
    # Absolute Z treshold above which a sample is considered to be an outlier (without fitting any model)
    outlier_thresh = 7

Fit the models
~~~~~~~~~~~~~~

Now we fit the models. This involves looping over the IDPs we have
selected. We will use a module from PCNtoolkit to set up the design
matrices, containing the covariates, fixed effects for site and
nonlinear basis expansion.

.. code:: ipython3

    for idp_num, idp in enumerate(idp_ids): 
        print('Running IDP', idp_num, idp, ':')
       
        # set output dir 
        idp_dir = os.path.join(out_dir, idp)
        os.makedirs(os.path.join(idp_dir), exist_ok=True)
        os.chdir(idp_dir)
        
        # extract the response variables for training and test set
        y_tr = df_tr[idp].to_numpy() 
        y_te = df_te[idp].to_numpy()
        
        # remove gross outliers and implausible values
        yz_tr = (y_tr - np.mean(y_tr)) / np.std(y_tr)
        yz_te = (y_te - np.mean(y_te)) / np.std(y_te)
        nz_tr = np.bitwise_and(np.abs(yz_tr) < outlier_thresh, y_tr > 0)
        nz_te = np.bitwise_and(np.abs(yz_te) < outlier_thresh, y_te > 0)
        y_tr = y_tr[nz_tr]
        y_te = y_te[nz_te]
        
        # write out the response variables for training and test
        resp_file_tr = os.path.join(idp_dir, 'resp_tr.txt')
        resp_file_te = os.path.join(idp_dir, 'resp_te.txt') 
        np.savetxt(resp_file_tr, y_tr)
        np.savetxt(resp_file_te, y_te)
            
        # configure the design matrix
        X_tr = create_design_matrix(df_tr[cols_cov].loc[nz_tr], 
                                    site_ids = df_tr['site'].loc[nz_tr],
                                    basis = 'bspline', 
                                    xmin = xmin, 
                                    xmax = xmax)
        X_te = create_design_matrix(df_te[cols_cov].loc[nz_te], 
                                    site_ids = df_te['site'].loc[nz_te], 
                                    all_sites=site_ids,
                                    basis = 'bspline', 
                                    xmin = xmin, 
                                    xmax = xmax)
    
        # configure and save the covariates
        cov_file_tr = os.path.join(idp_dir, 'cov_bspline_tr.txt')
        cov_file_te = os.path.join(idp_dir, 'cov_bspline_te.txt')
        np.savetxt(cov_file_tr, X_tr)
        np.savetxt(cov_file_te, X_te)
    
        if not force_refit and os.path.exists(os.path.join(idp_dir, 'Models', 'NM_0_0_estimate.pkl')):
            print('Making predictions using a pre-existing model...')
            suffix = 'predict'
            
            # Make prdictsion with test data
            predict(cov_file_te, 
                    alg='blr', 
                    respfile=resp_file_te, 
                    model_path=os.path.join(idp_dir,'Models'),
                    outputsuffix=suffix)
        else:
            print('Estimating the normative model...')
            estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te, 
                     testcov=cov_file_te, alg='blr', optimizer = 'l-bfgs-b', 
                     savemodel=True, warp=warp, warp_reparam=True)
            suffix = 'estimate'
        

Compute error metrics
~~~~~~~~~~~~~~~~~~~~~

In this section we compute the following error metrics for all IDPs (all
evaluated on the test set):

-  Negative log likelihood (NLL)
-  Explained variance (EV)
-  Mean standardized log loss (MSLL)
-  Bayesian information Criteria (BIC)
-  Skew and Kurtosis of the Z-distribution

.. code:: ipython3

    # initialise dataframe we will use to store quantitative metrics 
    blr_metrics = pd.DataFrame(columns = ['eid', 'NLL', 'EV', 'MSLL', 'BIC','Skew','Kurtosis'])
    
    for idp_num, idp in enumerate(idp_ids): 
        idp_dir = os.path.join(out_dir, idp)
        
        # load the predictions and true data. We use a custom function that ensures 2d arrays
        # equivalent to: y = np.loadtxt(filename); y = y[:, np.newaxis]
        yhat_te = load_2d(os.path.join(idp_dir, 'yhat_' + suffix + '.txt'))
        s2_te = load_2d(os.path.join(idp_dir, 'ys2_' + suffix + '.txt'))
        y_te = load_2d(os.path.join(idp_dir, 'resp_te.txt'))
        
        with open(os.path.join(idp_dir,'Models', 'NM_0_0_estimate.pkl'), 'rb') as handle:
            nm = pickle.load(handle) 
        
        # compute error metrics
        if warp is None:
            metrics = evaluate(y_te, yhat_te)  
            
            # compute MSLL manually as a sanity check
            y_tr_mean = np.array( [[np.mean(y_tr)]] )
            y_tr_var = np.array( [[np.var(y_tr)]] )
            MSLL = compute_MSLL(y_te, yhat_te, s2_te, y_tr_mean, y_tr_var)         
        else:
            warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1] 
            W = nm.blr.warp
            
            # warp predictions
            med_te = W.warp_predictions(np.squeeze(yhat_te), np.squeeze(s2_te), warp_param)[0]
            med_te = med_te[:, np.newaxis]
           
            # evaluation metrics
            metrics = evaluate(y_te, med_te)
            
            # compute MSLL manually
            y_te_w = W.f(y_te, warp_param)
            y_tr_w = W.f(y_tr, warp_param)
            y_tr_mean = np.array( [[np.mean(y_tr_w)]] )
            y_tr_var = np.array( [[np.var(y_tr_w)]] )
            MSLL = compute_MSLL(y_te_w, yhat_te, s2_te, y_tr_mean, y_tr_var)     
        
        Z = np.loadtxt(os.path.join(idp_dir, 'Z_' + suffix + '.txt'))
        [skew, sdskew, kurtosis, sdkurtosis, semean, sesd] = calibration_descriptives(Z)
        
        BIC = len(nm.blr.hyp) * np.log(y_tr.shape[0]) + 2 * nm.neg_log_lik
        
        blr_metrics.loc[len(blr_metrics)] = [idp, nm.neg_log_lik, metrics['EXPV'][0], 
                                             MSLL[0], BIC, skew, kurtosis]
        
    display(blr_metrics)
    
    blr_metrics.to_pickle(os.path.join(out_dir,'blr_metrics.pkl'))



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>eid</th>
          <th>NLL</th>
          <th>EV</th>
          <th>MSLL</th>
          <th>BIC</th>
          <th>Skew</th>
          <th>Kurtosis</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>lh_G&amp;S_frontomargin_thickness</td>
          <td>-3808.584381</td>
          <td>0.314419</td>
          <td>-35.106351</td>
          <td>-7579.659922</td>
          <td>0.252934</td>
          <td>1.087225</td>
        </tr>
        <tr>
          <th>1</th>
          <td>lh_G&amp;S_occipital_inf_thickness</td>
          <td>-3468.296931</td>
          <td>0.230447</td>
          <td>-35.096839</td>
          <td>-6899.085023</td>
          <td>0.030063</td>
          <td>0.430915</td>
        </tr>
        <tr>
          <th>2</th>
          <td>lh_G&amp;S_paracentral_thickness</td>
          <td>-2977.898155</td>
          <td>0.337686</td>
          <td>-35.035891</td>
          <td>-5918.287470</td>
          <td>-0.001040</td>
          <td>0.755307</td>
        </tr>
        <tr>
          <th>3</th>
          <td>lh_G&amp;S_subcentral_thickness</td>
          <td>-3471.667467</td>
          <td>0.332549</td>
          <td>-34.990710</td>
          <td>-6905.826095</td>
          <td>0.072970</td>
          <td>0.560048</td>
        </tr>
        <tr>
          <th>4</th>
          <td>lh_G&amp;S_transv_frontopol_thickness</td>
          <td>-1565.916398</td>
          <td>0.358683</td>
          <td>-34.900294</td>
          <td>-3094.323956</td>
          <td>0.270502</td>
          <td>1.269709</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>183</th>
          <td>TotalGrayVol</td>
          <td>146369.741818</td>
          <td>0.615736</td>
          <td>-3.067824</td>
          <td>292776.992475</td>
          <td>-0.490089</td>
          <td>3.996252</td>
        </tr>
        <tr>
          <th>184</th>
          <td>SupraTentorialVol</td>
          <td>152270.636605</td>
          <td>0.345575</td>
          <td>-1.442556</td>
          <td>304578.782049</td>
          <td>-0.302217</td>
          <td>2.920578</td>
        </tr>
        <tr>
          <th>185</th>
          <td>SupraTentorialVolNotVent</td>
          <td>162984.467798</td>
          <td>0.347517</td>
          <td>-1.014633</td>
          <td>326006.444436</td>
          <td>-5.035215</td>
          <td>63.806125</td>
        </tr>
        <tr>
          <th>186</th>
          <td>avg_thickness</td>
          <td>-10627.007679</td>
          <td>0.581347</td>
          <td>-36.109891</td>
          <td>-21216.506518</td>
          <td>-0.343804</td>
          <td>1.197945</td>
        </tr>
        <tr>
          <th>187</th>
          <td>EstimatedTotalIntraCranialVol</td>
          <td>168794.712119</td>
          <td>0.253537</td>
          <td>-0.262857</td>
          <td>337626.933077</td>
          <td>-5.151926</td>
          <td>66.531844</td>
        </tr>
      </tbody>
    </table>
    <p>188 rows × 7 columns</p>
    </div>


.. code:: ipython3

    blr_metrics.to_csv(os.path.join(out_dir,'blr_metrics.csv'))

