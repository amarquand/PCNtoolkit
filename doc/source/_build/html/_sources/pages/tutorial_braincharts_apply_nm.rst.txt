Braincharts: apply (transfer to new data)
-----------------------------------------------------

This notebook shows how to apply the coefficients from pre-estimated
normative models to new data. This can be done in two different ways:
(i) using a new set of data derived from the same sites used to estimate
the model and (ii) on a completely different set of sites. In the latter
case, we also need to estimate the site effect, which requires some
calibration/adaptation data. As an illustrative example, we use a
dataset derived from the `1000 functional connectomes
project <https://www.nitrc.org/forum/forum.php?thread_id=2907&forum_id=1383>`__
and adapt the learned model to make predictions on these data.

 `Open/Run in Google Colab <https://colab.research.google.com/github/predictive-clinical-neuroscience/braincharts/blob/master/scripts/apply_normative_models.ipynb>`__

First, if necessary, we install PCNtoolkit (note: this tutorial requires
at least version 0.20)

.. code:: ipython3

    !pip install pcntoolkit==0.20

Now we import the required libraries

.. code:: ipython3

    import os
    import numpy as np
    import pandas as pd
    import pickle
    from matplotlib import pyplot as plt
    import seaborn as sns
    
    from pcntoolkit.normative import estimate, predict, evaluate
    from pcntoolkit.util.utils import compute_MSLL, create_design_matrix
    from nm_utils import remove_bad_subjects, load_2d

Next, we configure some basic variables, like where we want the analysis
to be done and which model we want to use.

**Note:** We maintain a list of site ids for each dataset, which
describe the site names in the training and test data (``site_ids_tr``
and ``site_ids_te``), plus also the adaptation data . The training site
ids are provided as a text file in the distribution and the test ids are
extracted automatically from the pandas dataframe (see below). If you
use additional data from the sites (e.g. later waves from ABCD), it may
be necessary to adjust the site names to match the names in the training
set. See the accompanying paper for more details

.. code:: ipython3

    # which model do we wish to use?
    model_name = 'lifespan_29K_82sites_train'
    site_names = 'site_ids_82sites.txt'
    
    # where the analysis takes place
    root_dir = '<path-to-your>/braincharts'
    out_dir = os.path.join(root_dir, 'models', model_name)
    
    # load a set of site ids from this model. This must match the training data
    with open(os.path.join(root_dir,'docs', site_names)) as f:
        site_ids_tr = f.read().splitlines()

Download test dataset
~~~~~~~~~~~~~~~~~~~~~

As mentioned above, to demonstrate this tool we will use a test dataset
derived from the FCON 1000 dataset. We provide a prepackaged
training/test split of these data in the required format (also after
removing sites with only a few data points),
`here <https://github.com/predictive-clinical-neuroscience/PCNtoolkit-demo/tree/main/data>`__.
you can get these data by running the following commmands:

.. code:: ipython3

    os.chdir(root_dir)
    !wget -nc https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000_tr.csv
    !wget -nc https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/main/data/fcon1000_te.csv

Load test data
~~~~~~~~~~~~~~

Now we load the test data and remove some subjects that may have poor
scan quality. This asssesment is based on the Freesurfer Euler
characteristic as described in the papers below.

**References** - `Kia et al
2021 <https://www.biorxiv.org/content/10.1101/2021.05.28.446120v1.abstract>`__
- `Rosen et al
2018 <https://www.sciencedirect.com/science/article/abs/pii/S1053811917310832?via%3Dihub>`__

.. code:: ipython3

    test_data = os.path.join(root_dir, 'fcon1000_te.csv')
    
    df_te = pd.read_csv(test_data, index_col=0)
    
    # remove some bad subjects
    df_te, bad_sub = remove_bad_subjects(df_te, df_te)
    
    # extract a list of unique site ids from the test set
    site_ids_te =  sorted(set(df_te['site'].to_list()))


.. parsed-literal::

    16 subjects are removed!


(Optional) Load adaptation data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the data you wish to make predictions for is not derived from the
same scanning sites as those in the trainig set, it is necessary to
learn the site effect so that we can account for it in the predictions.
In order to do this in an unbiased way, we use a separate dataset, which
we refer to as ‘adaptation’ data. This must contain data for all the
same sites as in the test dataset and we assume these are coded in the
same way, based on a the ‘sitenum’ column in the dataframe.

.. code:: ipython3

    adaptation_data = os.path.join(root_dir, 'fcon1000_tr.csv')
    
    df_ad = pd.read_csv(adaptation_data, index_col=0)
    
    # remove some bad subjects
    df_ad, bad_sub = remove_bad_subjects(df_ad, df_ad)
    
    # extract a list of unique site ids from the test set
    site_ids_ad =  sorted(set(df_ad['site'].to_list()))
    
    if not all(elem in site_ids_ad for elem in site_ids_te):
        print('Warning: some of the testing sites are not in the adaptation data')


.. parsed-literal::

    11 subjects are removed!


Configure which models to fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, we configure which imaging derived phenotypes (IDPs) we would like
to process. This is just a list of column names in the dataframe we have
loaded above.

We could load the whole set (i.e. all phenotypes for which we have
models for …

.. code:: ipython3

    # load the list of idps for left and right hemispheres, plus subcortical regions
    with open(os.path.join(root_dir,'docs','phenotypes_lh.txt')) as f:
        idp_ids_lh = f.read().splitlines()
    with open(os.path.join(root_dir,'docs','phenotypes_rh.txt')) as f:
        idp_ids_rh = f.read().splitlines()
    with open(os.path.join(root_dir,'docs','phenotypes_sc.txt')) as f:
        idp_ids_sc = f.read().splitlines()
    
    # we choose here to process all idps
    idp_ids = idp_ids_lh + idp_ids_rh + idp_ids_sc

… or alternatively, we could just specify a list

.. code:: ipython3

    idp_ids = [ 'Left-Thalamus-Proper', 'Left-Lateral-Ventricle', 'rh_MeanThickness_thickness']

Configure covariates
~~~~~~~~~~~~~~~~~~~~

Now, we configure some parameters to fit the model. First, we choose
which columns of the pandas dataframe contain the covariates (age and
sex). The site parameters are configured automatically later on by the
``configure_design_matrix()`` function, when we loop through the IDPs in
the list

The supplied coefficients are derived from a ‘warped’ Bayesian linear
regression model, which uses a nonlinear warping function to model
non-Gaussianity (``sinarcsinh``) plus a non-linear basis expansion (a
cubic b-spline basis set with 5 knot points, which is the default value
in the PCNtoolkit package). Since we are sticking with the default
value, we do not need to specify any parameters for this, but we do need
to specify the limits. We choose to pad the input by a few years either
side of the input range. We will also set a couple of options that
control the estimation of the model

For further details about the likelihood warping approach, see the
accompanying paper and `Fraza et al
2021 <https://www.biorxiv.org/content/10.1101/2021.04.05.438429v1>`__.

.. code:: ipython3

    # which data columns do we wish to use as covariates? 
    cols_cov = ['age','sex']
    
    # limits for cubic B-spline basis 
    xmin = -5 
    xmax = 110
    
    # Absolute Z treshold above which a sample is considered to be an outlier (without fitting any model)
    outlier_thresh = 7

Make predictions
~~~~~~~~~~~~~~~~

This will make predictions for each IDP separately. This is done by
extracting a column from the dataframe (i.e. specifying the IDP as the
response variable) and saving it as a numpy array. Then, we configure
the covariates, which is a numpy data array having the number of rows
equal to the number of datapoints in the test set. The columns are
specified as follows:

-  A global intercept (column of ones)
-  The covariate columns (here age and sex, coded as 0=female/1=male)
-  Dummy coded columns for the sites in the training set (one column per
   site)
-  Columns for the basis expansion (seven columns for the default
   parameterisation)

Once these are saved as numpy arrays in ascii format (as here) or
(alternatively) in pickle format, these are passed as inputs to the
``predict()`` method in the PCNtoolkit normative modelling framework.
These are written in the same format to the location specified by
``idp_dir``. At the end of this step, we have a set of predictions and
Z-statistics for the test dataset that we can take forward to further
analysis.

Note that when we need to make predictions on new data, the procedure is
more involved, since we need to prepare, process and store covariates,
response variables and site ids for the adaptation data.

.. code:: ipython3

    for idp_num, idp in enumerate(idp_ids): 
        print('Running IDP', idp_num, idp, ':')
        idp_dir = os.path.join(out_dir, idp)
        os.chdir(idp_dir)
        
        # extract and save the response variables for the test set
        y_te = df_te[idp].to_numpy()
        
        # save the variables
        resp_file_te = os.path.join(idp_dir, 'resp_te.txt') 
        np.savetxt(resp_file_te, y_te)
            
        # configure and save the design matrix
        cov_file_te = os.path.join(idp_dir, 'cov_bspline_te.txt')
        X_te = create_design_matrix(df_te[cols_cov], 
                                    site_ids = df_te['site'],
                                    all_sites = site_ids_tr,
                                    basis = 'bspline', 
                                    xmin = xmin, 
                                    xmax = xmax)
        np.savetxt(cov_file_te, X_te)
        
        # check whether all sites in the test set are represented in the training set
        if all(elem in site_ids_tr for elem in site_ids_te):
            print('All sites are present in the training data')
            
            # just make predictions
            yhat_te, s2_te, Z = predict(cov_file_te, 
                                        alg='blr', 
                                        respfile=resp_file_te, 
                                        model_path=os.path.join(idp_dir,'Models'))
        else:
            print('Some sites missing from the training data. Adapting model')
            
            # save the covariates for the adaptation data
            X_ad = create_design_matrix(df_ad[cols_cov], 
                                        site_ids = df_ad['site'],
                                        all_sites = site_ids_tr,
                                        basis = 'bspline', 
                                        xmin = xmin, 
                                        xmax = xmax)
            cov_file_ad = os.path.join(idp_dir, 'cov_bspline_ad.txt')          
            np.savetxt(cov_file_ad, X_ad)
            
            # save the responses for the adaptation data
            resp_file_ad = os.path.join(idp_dir, 'resp_ad.txt') 
            y_ad = df_ad[idp].to_numpy()
            np.savetxt(resp_file_ad, y_ad)
           
            # save the site ids for the adaptation data
            sitenum_file_ad = os.path.join(idp_dir, 'sitenum_ad.txt') 
            site_num_ad = df_ad['sitenum'].to_numpy(dtype=int)
            np.savetxt(sitenum_file_ad, site_num_ad)
            
            # save the site ids for the test data 
            sitenum_file_te = os.path.join(idp_dir, 'sitenum_te.txt')
            site_num_te = df_te['sitenum'].to_numpy(dtype=int)
            np.savetxt(sitenum_file_te, site_num_te)
             
            yhat_te, s2_te, Z = predict(cov_file_te, 
                                        alg = 'blr', 
                                        respfile = resp_file_te, 
                                        model_path = os.path.join(idp_dir,'Models'),
                                        adaptrespfile = resp_file_ad,
                                        adaptcovfile = cov_file_ad,
                                        adaptvargroupfile = sitenum_file_ad,
                                        testvargroupfile = sitenum_file_te)


.. parsed-literal::

    Running IDP 0 Left-Thalamus-Proper :
    Some sites missing from the training data. Adapting model
    Loading data ...
    Prediction by model  1 of 1
    Evaluating the model ...
    Evaluations Writing outputs ...
    Writing outputs ...
    Running IDP 1 Left-Lateral-Ventricle :
    Some sites missing from the training data. Adapting model
    Loading data ...
    Prediction by model  1 of 1
    Evaluating the model ...
    Evaluations Writing outputs ...
    Writing outputs ...
    Running IDP 2 rh_MeanThickness_thickness :
    Some sites missing from the training data. Adapting model
    Loading data ...
    Prediction by model  1 of 1
    Evaluating the model ...
    Evaluations Writing outputs ...
    Writing outputs ...


Preparing dummy data for plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, we plot the centiles of variation estimated by the normative model.

We do this by making use of a set of dummy covariates that span the
whole range of the input space (for age) for a fixed value of the other
covariates (e.g. sex) so that we can make predictions for these dummy
data points, then plot them. We configure these dummy predictions using
the same procedure as we used for the real data. We can use the same
dummy data for all the IDPs we wish to plot

.. code:: ipython3

    # which sex do we want to plot? 
    sex = 1 # 1 = male 0 = female
    if sex == 1: 
        clr = 'blue';
    else:
        clr = 'red'
    
    # create dummy data for visualisation
    print('configuring dummy data ...')
    xx = np.arange(xmin, xmax, 0.5)
    X0_dummy = np.zeros((len(xx), 2))
    X0_dummy[:,0] = xx
    X0_dummy[:,1] = sex
    
    # create the design matrix
    X_dummy = create_design_matrix(X0_dummy, xmin=xmin, xmax=xmax, site_ids=None, all_sites=site_ids_tr)
    
    # save the dummy covariates
    cov_file_dummy = os.path.join(out_dir,'cov_bspline_dummy_mean.txt')
    np.savetxt(cov_file_dummy, X_dummy)


.. parsed-literal::

    configuring dummy data ...


Plotting the normative models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we loop through the IDPs, plotting each one separately. The outputs
of this step are a set of quantitative regression metrics for each IDP
and a set of centile curves which we plot the test data against.

This part of the code is relatively complex because we need to keep
track of many quantities for the plotting. We also need to remember
whether the data need to be warped or not. By default in PCNtoolkit,
predictions in the form of ``yhat, s2`` are always in the warped
(Gaussian) space. If we want predictions in the input (non-Gaussian)
space, then we need to warp them with the inverse of the estimated
warping function. This can be done using the function
``nm.blr.warp.warp_predictions()``.

**Note:** it is necessary to update the intercept for each of the sites.
For purposes of visualisation, here we do this by adjusting the median
of the data to match the dummy predictions, but note that all the
quantitative metrics are estimated using the predictions that are
adjusted properly using a learned offset (or adjusted using a hold-out
adaptation set, as above). Note also that for the calibration data we
require at least two data points of the same sex in each site to be able
to estimate the variance. Of course, in a real example, you would want
many more than just two since we need to get a reliable estimate of the
variance for each site.

.. code:: ipython3

    sns.set(style='whitegrid')
    
    for idp_num, idp in enumerate(idp_ids): 
        print('Running IDP', idp_num, idp, ':')
        idp_dir = os.path.join(out_dir, idp)
        os.chdir(idp_dir)
        
        # load the true data points
        yhat_te = load_2d(os.path.join(idp_dir, 'yhat_predict.txt'))
        s2_te = load_2d(os.path.join(idp_dir, 'ys2_predict.txt'))
        y_te = load_2d(os.path.join(idp_dir, 'resp_te.txt'))
                
        # set up the covariates for the dummy data
        print('Making predictions with dummy covariates (for visualisation)')
        yhat, s2 = predict(cov_file_dummy, 
                           alg = 'blr', 
                           respfile = None, 
                           model_path = os.path.join(idp_dir,'Models'), 
                           outputsuffix = '_dummy')
        
        # load the normative model
        with open(os.path.join(idp_dir,'Models', 'NM_0_0_estimate.pkl'), 'rb') as handle:
            nm = pickle.load(handle) 
        
        # get the warp and warp parameters
        W = nm.blr.warp
        warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1] 
            
        # first, we warp predictions for the true data and compute evaluation metrics
        med_te = W.warp_predictions(np.squeeze(yhat_te), np.squeeze(s2_te), warp_param)[0]
        med_te = med_te[:, np.newaxis]
        print('metrics:', evaluate(y_te, med_te))
        
        # then, we warp dummy predictions to create the plots
        med, pr_int = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param)
        
        # extract the different variance components to visualise
        beta, junk1, junk2 = nm.blr._parse_hyps(nm.blr.hyp, X_dummy)
        s2n = 1/beta # variation (aleatoric uncertainty)
        s2s = s2-s2n # modelling uncertainty (epistemic uncertainty)
        
        # plot the data points
        y_te_rescaled_all = np.zeros_like(y_te)
        for sid, site in enumerate(site_ids_te):
            # plot the true test data points 
            if all(elem in site_ids_tr for elem in site_ids_te):
                # all data in the test set are present in the training set
                
                # first, we select the data points belonging to this particular site
                idx = np.where(np.bitwise_and(X_te[:,2] == sex, X_te[:,sid+len(cols_cov)+1] !=0))[0]
                if len(idx) == 0:
                    print('No data for site', sid, site, 'skipping...')
                    continue
                
                # then directly adjust the data
                idx_dummy = np.bitwise_and(X_dummy[:,1] > X_te[idx,1].min(), X_dummy[:,1] < X_te[idx,1].max())
                y_te_rescaled = y_te[idx] - np.median(y_te[idx]) + np.median(med[idx_dummy])
            else:
                # we need to adjust the data based on the adaptation dataset 
                
                # first, select the data point belonging to this particular site
                idx = np.where(np.bitwise_and(X_te[:,2] == sex, (df_te['site'] == site).to_numpy()))[0]
                
                # load the adaptation data
                y_ad = load_2d(os.path.join(idp_dir, 'resp_ad.txt'))
                X_ad = load_2d(os.path.join(idp_dir, 'cov_bspline_ad.txt'))
                idx_a = np.where(np.bitwise_and(X_ad[:,2] == sex, (df_ad['site'] == site).to_numpy()))[0]
                if len(idx) < 2 or len(idx_a) < 2:
                    print('Insufficent data for site', sid, site, 'skipping...')
                    continue
                
                # adjust and rescale the data
                y_te_rescaled, s2_rescaled = nm.blr.predict_and_adjust(nm.blr.hyp, 
                                                                       X_ad[idx_a,:], 
                                                                       np.squeeze(y_ad[idx_a]), 
                                                                       Xs=None, 
                                                                       ys=np.squeeze(y_te[idx]))
            # plot the (adjusted) data points
            plt.scatter(X_te[idx,1], y_te_rescaled, s=4, color=clr, alpha = 0.1)
           
        # plot the median of the dummy data
        plt.plot(xx, med, clr)
        
        # fill the gaps in between the centiles
        junk, pr_int25 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.25,0.75])
        junk, pr_int95 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.05,0.95])
        junk, pr_int99 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.01,0.99])
        plt.fill_between(xx, pr_int25[:,0], pr_int25[:,1], alpha = 0.1,color=clr)
        plt.fill_between(xx, pr_int95[:,0], pr_int95[:,1], alpha = 0.1,color=clr)
        plt.fill_between(xx, pr_int99[:,0], pr_int99[:,1], alpha = 0.1,color=clr)
                
        # make the width of each centile proportional to the epistemic uncertainty
        junk, pr_int25l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.25,0.75])
        junk, pr_int95l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.05,0.95])
        junk, pr_int99l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.01,0.99])
        junk, pr_int25u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.25,0.75])
        junk, pr_int95u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.05,0.95])
        junk, pr_int99u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.01,0.99])    
        plt.fill_between(xx, pr_int25l[:,0], pr_int25u[:,0], alpha = 0.3,color=clr)
        plt.fill_between(xx, pr_int95l[:,0], pr_int95u[:,0], alpha = 0.3,color=clr)
        plt.fill_between(xx, pr_int99l[:,0], pr_int99u[:,0], alpha = 0.3,color=clr)
        plt.fill_between(xx, pr_int25l[:,1], pr_int25u[:,1], alpha = 0.3,color=clr)
        plt.fill_between(xx, pr_int95l[:,1], pr_int95u[:,1], alpha = 0.3,color=clr)
        plt.fill_between(xx, pr_int99l[:,1], pr_int99u[:,1], alpha = 0.3,color=clr)
    
        # plot actual centile lines
        plt.plot(xx, pr_int25[:,0],color=clr, linewidth=0.5)
        plt.plot(xx, pr_int25[:,1],color=clr, linewidth=0.5)
        plt.plot(xx, pr_int95[:,0],color=clr, linewidth=0.5)
        plt.plot(xx, pr_int95[:,1],color=clr, linewidth=0.5)
        plt.plot(xx, pr_int99[:,0],color=clr, linewidth=0.5)
        plt.plot(xx, pr_int99[:,1],color=clr, linewidth=0.5)
        
        plt.xlabel('Age')
        plt.ylabel(idp) 
        plt.title(idp)
        plt.xlim((0,90))
        plt.savefig(os.path.join(idp_dir, 'centiles_' + str(sex)),  bbox_inches='tight')
        plt.show()


.. parsed-literal::

    Running IDP 0 Left-Thalamus-Proper :
    Making predictions with dummy covariates (for visualisation)
    Loading data ...
    Prediction by model  1 of 1
    Writing outputs ...
    metrics: {'RMSE': array([704.24906029]), 'Rho': array([0.6136885]), 'pRho': array([7.63644502e-59]), 'SMSE': array([0.63500304]), 'EXPV': array([0.37380003])}
    Insufficent data for site 8 Cleveland skipping...
    Insufficent data for site 19 PaloAlto skipping...



.. image:: apply_normative_models_files/apply_normative_models_23_1.png


.. parsed-literal::

    Running IDP 1 Left-Lateral-Ventricle :
    Making predictions with dummy covariates (for visualisation)
    Loading data ...
    Prediction by model  1 of 1
    Writing outputs ...
    metrics: {'RMSE': array([3939.29791125]), 'Rho': array([0.42275398]), 'pRho': array([1.86615581e-24]), 'SMSE': array([0.85019218]), 'EXPV': array([0.1786487])}
    Insufficent data for site 8 Cleveland skipping...
    Insufficent data for site 19 PaloAlto skipping...



.. image:: apply_normative_models_files/apply_normative_models_23_3.png


.. parsed-literal::

    Running IDP 2 rh_MeanThickness_thickness :
    Making predictions with dummy covariates (for visualisation)
    Loading data ...
    Prediction by model  1 of 1
    Writing outputs ...
    metrics: {'RMSE': array([0.07307275]), 'Rho': array([0.64482158]), 'pRho': array([2.29573893e-67]), 'SMSE': array([0.60735348]), 'EXPV': array([0.40563038])}
    Insufficent data for site 8 Cleveland skipping...
    Insufficent data for site 19 PaloAlto skipping...



.. image:: apply_normative_models_files/apply_normative_models_23_5.png


