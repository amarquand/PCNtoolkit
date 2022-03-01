.. title:: post-hoc tutorial

Post-hoc analysis on normative modeling outputs
===================================================

The Normative Modeling Framework for Computational Psychiatry. Nature Protocols. https://doi.org/10.1101/2021.08.08.455583.

Created by `Saige Rutherford <https://twitter.com/being_saige>`__


.. image:: https://colab.research.google.com/assets/colab-badge.svg 
    :target: https://colab.research.google.com/github/predictive-clinical-neuroscience/PCNtoolkit-demo/blob/main/tutorials/BLR_protocol/post_hoc_analysis.ipynb


SVM classification 
----------------------------------------------

Classify schizophrenia group from controls using cortical thickness
deviation scores (z-scores) and then the true cortical thickness data to
see which type of data better separates the groups.

.. code:: ipython3

    ! git clone https://github.com/predictive-clinical-neuroscience/PCNtoolkit-demo.git


.. parsed-literal::

    Cloning into 'PCNtoolkit-demo'...
    remote: Enumerating objects: 855, done.[K
    remote: Counting objects: 100% (855/855), done.[K
    remote: Compressing objects: 100% (737/737), done.[K
    remote: Total 855 (delta 278), reused 601 (delta 101), pack-reused 0[K
    Receiving objects: 100% (855/855), 18.07 MiB | 16.65 MiB/s, done.
    Resolving deltas: 100% (278/278), done.


.. code:: ipython3

    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    os.chdir('/content/PCNtoolkit-demo/')

.. code:: ipython3

    Z_df = pd.read_csv('data/fcon1000_te_Z.csv')

.. code:: ipython3

    from sklearn import svm
    from sklearn.metrics import auc
    from sklearn.metrics import plot_roc_curve
    from sklearn.model_selection import StratifiedKFold

.. code:: ipython3

    Z_df.dropna(subset=['group'], inplace=True)

.. code:: ipython3

    Z_df['group'] = Z_df['group'].replace("SZ",0)

.. code:: ipython3

    Z_df['group'] = Z_df['group'].replace("Control",1)

.. code:: ipython3

    deviations = Z_df.loc[:, Z_df.columns.str.contains('Z_predict')]

.. code:: ipython3

    cortical_thickness = Z_df.loc[:, Z_df.columns.str.endswith('_thickness')]

.. code:: ipython3

    # Data IO and generation
    X1 = deviations
    X2 = cortical_thickness
    y = Z_df['group']
    n_samples, n_features = X1.shape
    random_state = np.random.RandomState(0)

.. code:: ipython3

    X1 = X1.to_numpy()

.. code:: ipython3

    X2 = X2.to_numpy()

.. code:: ipython3

    y = y.astype(int)

.. code:: ipython3

    y = y.to_numpy()

SVM using deviation scores as features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3
    
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=5)
    classifier = svm.SVC(kernel='linear', probability=True,
                         random_state=random_state)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots(figsize=(15,15))
    parameters = {'axes.labelsize': 20,
              'axes.titlesize': 25, 'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':14,'legend.title_fontsize':16}
    plt.rcParams.update(parameters)
    
    for i, (train, test) in enumerate(cv.split(X1, y)):
        classifier.fit(X1[train], y[train])
        viz = plot_roc_curve(classifier, X1[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.set_title('Receiver operating characteristic SZ vs. HC (deviations)', fontweight="bold", size=20)
    ax.legend(loc="lower right")
    plt.show()

.. image:: post_hoc_analysis_files/post_hoc_analysis_17_1.png


SVM using true cortical thickness data as features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=5)
    classifier = svm.SVC(kernel='linear', probability=True,
                         random_state=random_state)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots(figsize=(15,15))
    parameters = {'axes.labelsize': 20,
              'axes.titlesize': 25, 'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':14,'legend.title_fontsize':16}
    plt.rcParams.update(parameters)
    
    for i, (train, test) in enumerate(cv.split(X2, y)):
        classifier.fit(X2[train], y[train])
        viz = plot_roc_curve(classifier, X2[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.set_title('Receiver operating characteristic SZ vs. HC (cortical thickness)', fontweight="bold", size=20)
    ax.legend(loc="lower right")
    plt.show()


.. image:: post_hoc_analysis_files/post_hoc_analysis_19_1.png


Classical case-control testing 
-----------------------------------------------------

.. code:: ipython3

    ! pip install statsmodels

.. code:: ipython3

    from scipy.stats import ttest_ind
    from statsmodels.stats import multitest


.. code:: ipython3

    SZ = Z_df.query('group == 0')
    HC = Z_df.query('group == 1')

Mass univariate two sample t-tests on deviation score maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code:: ipython3

    SZ_deviations = SZ.loc[:, SZ.columns.str.contains('Z_predict')]

.. code:: ipython3

    HC_deviations = HC.loc[:, HC.columns.str.contains('Z_predict')]

.. code:: ipython3

    z_cols = SZ_deviations.columns

.. code:: ipython3

    sz_hc_pvals_z = pd.DataFrame(columns={'roi','pval', 'tstat','fdr_pval'})
    for index, column in enumerate(z_cols):
        test = ttest_ind(SZ_deviations[column], HC_deviations[column])
        sz_hc_pvals_z.loc[index, 'pval'] = test.pvalue
        sz_hc_pvals_z.loc[index, 'tstat'] = test.statistic
        sz_hc_pvals_z.loc[index, 'roi'] = column

.. code:: ipython3

    sz_hc_fdr_z = multitest.fdrcorrection(sz_hc_pvals_z['pval'], alpha=0.05, method='indep', is_sorted=False)

.. code:: ipython3

    sz_hc_pvals_z['fdr_pval'] = sz_hc_fdr_z[1]

.. code:: ipython3

    sz_hc_z_sig_diff = sz_hc_pvals_z.query('pval < 0.05')

.. code:: ipython3

    sz_hc_z_sig_diff


.. raw:: html

    
      <div id="df-eca46e49-c67f-4030-b124-1bbef7358cac">
        <div class="colab-df-container">
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
          <th>roi</th>
          <th>fdr_pval</th>
          <th>pval</th>
          <th>tstat</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>Left-Amygdala_Z_predict</td>
          <td>0.089187</td>
          <td>0.04314</td>
          <td>-2.043665</td>
        </tr>
        <tr>
          <th>3</th>
          <td>rh_MeanThickness_thickness_Z_predict</td>
          <td>0.001476</td>
          <td>0.000047</td>
          <td>-4.219322</td>
        </tr>
        <tr>
          <th>4</th>
          <td>lh_G&amp;S_frontomargin_thickness_Z_predict</td>
          <td>0.066297</td>
          <td>0.027299</td>
          <td>-2.234088</td>
        </tr>
        <tr>
          <th>5</th>
          <td>rh_Pole_temporal_thickness_Z_predict</td>
          <td>0.046111</td>
          <td>0.016768</td>
          <td>-2.425135</td>
        </tr>
        <tr>
          <th>7</th>
          <td>rh_G_occipital_middle_thickness_Z_predict</td>
          <td>0.08663</td>
          <td>0.040304</td>
          <td>-2.072725</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>176</th>
          <td>Left-Lateral-Ventricle_Z_predict</td>
          <td>0.035835</td>
          <td>0.010348</td>
          <td>2.604355</td>
        </tr>
        <tr>
          <th>177</th>
          <td>rh_G_front_inf-Orbital_thickness_Z_predict</td>
          <td>0.067346</td>
          <td>0.029075</td>
          <td>-2.20854</td>
        </tr>
        <tr>
          <th>179</th>
          <td>lh_S_temporal_inf_thickness_Z_predict</td>
          <td>0.011567</td>
          <td>0.001484</td>
          <td>-3.251486</td>
        </tr>
        <tr>
          <th>180</th>
          <td>rh_G_precentral_thickness_Z_predict</td>
          <td>0.007984</td>
          <td>0.00079</td>
          <td>-3.442643</td>
        </tr>
        <tr>
          <th>185</th>
          <td>rh_G_temporal_inf_thickness_Z_predict</td>
          <td>0.055785</td>
          <td>0.021777</td>
          <td>-2.324048</td>
        </tr>
      </tbody>
    </table>
    <p>96 rows × 4 columns</p>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-eca46e49-c67f-4030-b124-1bbef7358cac')"
                  title="Convert this dataframe to an interactive table."
                  style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
           width="24px">
        <path d="M0 0h24v24H0V0z" fill="none"/>
        <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
      </svg>
          </button>
    
      <style>
        .colab-df-container {
          display:flex;
          flex-wrap:wrap;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
          <script>
            const buttonEl =
              document.querySelector('#df-eca46e49-c67f-4030-b124-1bbef7358cac button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';
    
            async function convertToInteractive(key) {
              const element = document.querySelector('#df-eca46e49-c67f-4030-b124-1bbef7358cac');
              const dataTable =
                await google.colab.kernel.invokeFunction('convertToInteractive',
                                                         [key], {});
              if (!dataTable) return;
    
              const docLinkHtml = 'Like what you see? Visit the ' +
                '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
                + ' to learn more about interactive tables.';
              element.innerHTML = '';
              dataTable['output_type'] = 'display_data';
              await google.colab.output.renderOutput(dataTable, element);
              const docLink = document.createElement('div');
              docLink.innerHTML = docLinkHtml;
              element.appendChild(docLink);
            }
          </script>
        </div>
      </div>




.. code:: ipython3

    sz_hc_z_sig_diff.shape




.. parsed-literal::

    (96, 4)


Mass univariate two sample t-tests on true cortical thickness data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    SZ_cortical_thickness = SZ.loc[:, SZ.columns.str.endswith('_thickness')]

.. code:: ipython3

    HC_cortical_thickness = HC.loc[:, HC.columns.str.endswith('_thickness')]

.. code:: ipython3

    ct_cols = SZ_cortical_thickness.columns

.. code:: ipython3

    sz_hc_pvals_ct = pd.DataFrame(columns={'roi','pval', 'tstat','fdr_pval'})
    for index, column in enumerate(ct_cols):
        test = ttest_ind(SZ_cortical_thickness[column], HC_cortical_thickness[column])
        sz_hc_pvals_ct.loc[index, 'pval'] = test.pvalue
        sz_hc_pvals_ct.loc[index, 'tstat'] = test.statistic
        sz_hc_pvals_ct.loc[index, 'roi'] = column

.. code:: ipython3

    sz_hc_fdr_ct = multitest.fdrcorrection(sz_hc_pvals_ct['pval'], alpha=0.05, method='indep', is_sorted=False)

.. code:: ipython3

    sz_hc_pvals_ct['fdr_pval'] = sz_hc_fdr_ct[1]

.. code:: ipython3

    sz_hc_ct_sig_diff = sz_hc_pvals_ct.query('pval < 0.05')

.. code:: ipython3

    sz_hc_ct_sig_diff




.. raw:: html

    
      <div id="df-378bc888-2e27-48f6-bb04-2993f86d8a98">
        <div class="colab-df-container">
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
          <th>roi</th>
          <th>fdr_pval</th>
          <th>pval</th>
          <th>tstat</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>lh_G&amp;S_occipital_inf_thickness</td>
          <td>0.025994</td>
          <td>0.002599</td>
          <td>-3.074854</td>
        </tr>
        <tr>
          <th>5</th>
          <td>lh_G&amp;S_cingul-Ant_thickness</td>
          <td>0.01673</td>
          <td>0.000558</td>
          <td>-3.54496</td>
        </tr>
        <tr>
          <th>6</th>
          <td>lh_G&amp;S_cingul-Mid-Ant_thickness</td>
          <td>0.066125</td>
          <td>0.01613</td>
          <td>-2.439868</td>
        </tr>
        <tr>
          <th>7</th>
          <td>lh_G&amp;S_cingul-Mid-Post_thickness</td>
          <td>0.1104</td>
          <td>0.046162</td>
          <td>-2.014447</td>
        </tr>
        <tr>
          <th>11</th>
          <td>lh_G_front_inf-Opercular_thickness</td>
          <td>0.070606</td>
          <td>0.021034</td>
          <td>-2.337646</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>135</th>
          <td>rh_S_oc-temp_med&amp;Lingual_thickness</td>
          <td>0.076018</td>
          <td>0.026761</td>
          <td>-2.24211</td>
        </tr>
        <tr>
          <th>141</th>
          <td>rh_S_postcentral_thickness</td>
          <td>0.070606</td>
          <td>0.019369</td>
          <td>-2.369738</td>
        </tr>
        <tr>
          <th>142</th>
          <td>rh_S_precentral-inf-part_thickness</td>
          <td>0.019935</td>
          <td>0.001409</td>
          <td>-3.267676</td>
        </tr>
        <tr>
          <th>143</th>
          <td>rh_S_precentral-sup-part_thickness</td>
          <td>0.046377</td>
          <td>0.006802</td>
          <td>-2.753296</td>
        </tr>
        <tr>
          <th>149</th>
          <td>rh_MeanThickness_thickness</td>
          <td>0.019935</td>
          <td>0.001658</td>
          <td>-3.217126</td>
        </tr>
      </tbody>
    </table>
    <p>67 rows × 4 columns</p>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-378bc888-2e27-48f6-bb04-2993f86d8a98')"
                  title="Convert this dataframe to an interactive table."
                  style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
           width="24px">
        <path d="M0 0h24v24H0V0z" fill="none"/>
        <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
      </svg>
          </button>
    
      <style>
        .colab-df-container {
          display:flex;
          flex-wrap:wrap;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
          <script>
            const buttonEl =
              document.querySelector('#df-378bc888-2e27-48f6-bb04-2993f86d8a98 button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';
    
            async function convertToInteractive(key) {
              const element = document.querySelector('#df-378bc888-2e27-48f6-bb04-2993f86d8a98');
              const dataTable =
                await google.colab.kernel.invokeFunction('convertToInteractive',
                                                         [key], {});
              if (!dataTable) return;
    
              const docLinkHtml = 'Like what you see? Visit the ' +
                '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
                + ' to learn more about interactive tables.';
              element.innerHTML = '';
              dataTable['output_type'] = 'display_data';
              await google.colab.output.renderOutput(dataTable, element);
              const docLink = document.createElement('div');
              docLink.innerHTML = docLinkHtml;
              element.appendChild(docLink);
            }
          </script>
        </div>
      </div>




.. code:: ipython3

    sz_hc_ct_sig_diff.shape




.. parsed-literal::

    (67, 4)


