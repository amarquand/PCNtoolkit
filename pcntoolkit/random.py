
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:34:31 2024

@author: johbay
"""

import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import matlib as mb
from scipy import special as spp
from sklearn.linear_model import LinearRegression
from utilities_thirve import *
from utils import sc_idp_cols

import pcntoolkit as pkt
from pcntoolkit.normative_model.norm_hbr import quantile
from pcntoolkit.util.utils import scaler

#%%

model_type ="SHASHb_1"
atlas = 'SC'
idp_nr = 0
measures = sc_idp_cols()
measure = measures[idp_nr]

base_dir= ('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/')
data_dir = os.path.join(base_dir, "Velocity/DATA_pre_processed")
test_raw = pd.read_pickle(
    '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/test_sc_adults.pkl')

batch_dir = f'/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/{model_type}_estimate_scaled_fixed_{atlas}_demented_adults2/batch_{idp_nr +1}/'

test_raw["age"] = test_raw["age"].astype(float)
test_raw["ID_subject"] = test_raw["ID_subject"].astype(str)
test_raw["sex"] = test_raw["sex"].astype(str)
test_raw["site_id"] = test_raw["site_id"].astype(str)
test_raw["ID_visit"].unique()
test_raw["ID_visit"] = test_raw["ID_visit"].astype(str)
test_raw["ID_visit"] = test_raw["ID_visit"].str.replace('v', '')
test_raw["ID_visit"] = test_raw["ID_visit"].astype(float)
test_raw["which_dataset"] = test_raw["which_dataset"].astype(int)
test_raw.reset_index(drop=True, inplace=True)
#
#  get the z-scores for
Z_measures = pd.read_pickle(os.path.join(batch_dir, 'Z_fit.pkl'))

test_raw.reset_index(drop=True, inplace=True)

# add covarites
data_all = pd.concat(
    [Z_measures, test_raw[["ID_subject", "ID_visit", "age", "sex", "site_id", "which_dataset"]]],  axis=1)

data_all.rename(columns={data_all.columns[0]: measure}, inplace=True )

#
# OAS2_0157 is a good example
#OASIS2_demented = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Longitudinal/Data/OASIS2_sc_demented.pkl")

data = data_all[data_all["which_dataset"]==2]

data["age"] = data["age"].round()
data["age"] =  data["age"].astype(int)
    
data["ID_visit"].astype(int)
data["ID_subject"].astype(str)


data_demented = data_all[data_all["which_dataset"]==4]
plot_data = data_demented[(data_demented["age"] >= 73) & (data_demented["age"] <= 75)]
OAS2_0157 = data_all[data_all["ID_subject"]=="OAS2_0157"]

for _, group_data in plot_data.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'])
plt.show()

test_raw["which_dataset"] = pd.to_numeric(test_raw["which_dataset"], errors="coerce")

test_raw_demented = test_raw.loc[test_raw["which_dataset"] == 4,["Left-Lateral-Ventricle", "ID_subject","age"]]
for _, group_data in test_raw_demented.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'])
plt.show()


filtered_subjects = data_demented.groupby("ID_subject").filter(lambda x: len(x) > 2)
for _, group_data in filtered_subjects.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'])
plt.show()

#%%

def make_velocity_plots(idp_nr, measure, data= data, seletced_sex = 'male', model_type='SHAHSb_1', re_estimate = True):

    batch_dir = f'/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/{model_type}_estimate_scaled_fixed_{atlas}_demented_adults2/batch_{idp_nr +1 }/'
    f_idx = idp_nr
        
    #idps= sc_idp_cols()
    #measure = idps[f_idx]

    
    velocity_dir = os.path.join(batch_dir,'Velocity/')
    if not os.path.exists(velocity_dir):
        os.mkdir(velocity_dir)
    
    
    A, Phi = make_correlation_matrix(data = data, measure = measure)
      
    # Create regression matrices for different gaps
    reg_matrices = [create_regression_matrix(Phi, gap) for gap in range(1, 6)]
    reg_matrix = pd.concat(reg_matrices).dropna()
    reg_matrix["V0"] = 1

    # Separate dependent and independent variables
    y = reg_matrix["y"]
    X = reg_matrix.drop(columns=["y"])

    # Train regression model
    reg_model = LinearRegression(fit_intercept=False).fit(X, y)
    
    coef = reg_model.coef_
    
    # Prediction matrix setup
    Age = np.linspace(0, 97, 98)[:, None]
    X_pred = pd.DataFrame({"V1": np.log(Age.flatten() + 0.5), "V2": 0, "V3": 1})
    X_pred["V4"] = X_pred["V1"] * X_pred["V2"]
    X_pred["V5"] = X_pred["V1"] ** 2
    X_pred["V0"] = 1

    pred_diagonal = reg_model.predict(X_pred)
    
    pred_r = recursive_r_transform(pred_diagonal)
    
    #step_results = recursive_multiply(pred_r)
    #print("Steps:", step_results)
 
    velocity_list_age, velocity = generate_thrive_lines(
        pred_r, start_year=0, end_year=90, velocity_length=3, start_z=-3, end_z=4, space_between_anchors=1, z_thrive=1.96
    )


    
    Z_train = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/trbefile_SC_demented_adults.pkl')
    X_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/X_train_SC_demented_adults.pkl",'rb')).to_numpy()
    Y_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/Y_train_SC_demented_adults.pkl",'rb'))
    #Z_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/trbefile_delta_fixed_DK.pkl",'rb')).to_numpy()
    #X_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/X_test_cross_DK.pkl",'rb')).to_numpy()
    #Y_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/Y_test_cross_L_bankkssts_DK.pkl",'rb'))
    #Z_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/tebefile_cross_DK.pkl",'rb')).to_numpy()
    Z_train.shape
    X_train.shape
    
    
    
    model_path = os.path.join(batch_dir, "Models/NM_0_0_fit.pkl")
    with open(model_path,'rb') as file:
        model = pickle.load(file)
    
    this_Y_train = Y_train[f_idx].to_numpy()
        #this_Y_test = Y_test[f_idx].to_numpy()
    inscaler = scaler("standardize")
    outscaler = scaler("standardize")
    this_scaled_X_train = inscaler.fit_transform(X_train)
    this_scaled_Y_train = outscaler.fit_transform(this_Y_train)
        #this_scaled_X_test = inscaler.transform(X_test)
        #this_scaled_Y_test = outscaler.transform(this_Y_test)
        
    selected_sex_id = 0 if selected_sex == 'female' else 1
    train_sex_idx = np.where(X_train[:,1]==selected_sex_id)
        #test_sex_idx = np.where(X_test[:,1]==selected_sex_id)
    
        # select a model batch effect (69,1)
    model_be = [1]
    #mu_intercept_mu = model.hbr.idata.posterior['mu_intercept_mu'].to_numpy().mean()
    sigma_intercept_mu = model.hbr.idata.posterior['sigma_intercept_mu'].to_numpy().mean()
    offsets = model.hbr.idata.posterior['offset_intercept_mu'].to_numpy().mean(axis = (0,1))
    #model_offset_intercept_mu_be = offsets[model_be]
    
        
    #Make an empty array
    centered_Y_train = np.zeros_like(this_scaled_Y_train)
    #centered_Y_test = np.zeros_like(this_scaled_Y_test)
    
    #For each batch effect
    for i, be in enumerate(np.unique(Z_train)):
        this_offset_intercept = offsets[i]          
        idx = (Z_train == be).all(1) 
    
        centered_Y_train[idx] = this_scaled_Y_train[idx]-sigma_intercept_mu*this_offset_intercept
        #idx = (Z_test == be).all(1) 
        #centered_Y_test[idx] = this_scaled_Y_test[idx]-sigma_intercept_mu*this_offset_intercept
    
    fig = plt.figure(figsize=(5,4))
    
    ytrain_inv = outscaler.inverse_transform(centered_Y_train[train_sex_idx,None])
    maxy = np.max(ytrain_inv)
    miny = np.min(ytrain_inv)
    dify = maxy - miny
    plt.ylim(miny - 0.1*dify, maxy + 0.1*dify )
    #plt.scatter(inscaler.inverse_transform(this_scaled_X_train)[train_sex_idx,0], outscaler.inverse_transform(centered_Y_train[train_sex_idx,None]), alpha = 0.1, s = 12, color=cols[0])
    #plt.scatter(inscaler.inverse_transform(this_scaled_X_test)[test_sex_idx,0], outscaler.inverse_transform(centered_Y_test[test_sex_idx,None]), alpha = 0.1, s = 12, color=cols[1])
    
    be_map =np.unique(Z_train)
    difX = np.max(this_scaled_X_train[:,0])-np.min(this_scaled_X_train[:,0])
    min0 = np.min(this_scaled_X_train[:,0]) + 0.01*difX
    max0 = np.max(this_scaled_X_train[:,0]) - 0.01*difX
    #sex = np.unique(this_scaled_X_train[:,1])[selected_sex_id]
    
    
    synthetic_X0 = np.linspace(min0, max0, 100)[:,None]
    #synthetic_X0 = np.linspace(min0, max0, 59)[:,None]
    #synthetic_X0 = np.linspace(39.5, 98.5, 59)[:,None]
    sex = 1
    #synthetic_X0 = np.linspace(0, 0.96, 97)[:,None]
    #     plt.xlim(min0,max0)
    synthetic_X = np.concatenate((synthetic_X0, np.full(synthetic_X0.shape,sex)),axis = 1)
    #synthetic_X = synthetic_X[41:]
    
    
    ran = np.arange(-3,4)
        
        # q = get_single_quantiles(synthetic_X,ran, model, model_be,MAP)-sigma_intercept_mu*offsets[tuple(model_be)]
    model_be_long = np.repeat(np.array(be_map[model_be]),synthetic_X.shape[0])
    q = model.get_mcmc_quantiles(synthetic_X, model_be_long) - sigma_intercept_mu * offsets[model_be]
    q = outscaler.inverse_transform(q).T
    #q = outscaler.inverse_transform(q)
    
    x_for_thrivelines = velocity_list_age.astype(float)
    x_for_thrivelines = np.stack((x_for_thrivelines, np.full(x_for_thrivelines.shape,sex)),axis = 1)
    x_for_thrivelines = inscaler.transform(x_for_thrivelines)
    z_for_thrivelines = velocity[:,None]
    be_for_thrivelines = np.ones_like(z_for_thrivelines)* be_map[model_be]
    
    
    x = inscaler.inverse_transform(synthetic_X)
    
    ## thrivelines
    q_for_thrivelines = model.get_mcmc_quantiles(X = x_for_thrivelines,batch_effects=be_for_thrivelines, z_scores=z_for_thrivelines) - sigma_intercept_mu * offsets[model_be]
    q_for_thrivelines = outscaler.inverse_transform(q_for_thrivelines[0])
    
    # #demented
    x_for_thrivelines_demented = np.array([73,75]).astype(float)
    x_for_thrivelines_demented = np.stack((x_for_thrivelines_demented, np.full(x_for_thrivelines_demented.shape,0)),axis = 1)
    x_for_thrivelines_demented = inscaler.transform(x_for_thrivelines_demented)
    z_for_thrivelines_demented = np.array(OAS2_0157["Left-Lateral-Ventricle"])[:,None]
    be_for_thrivelines_demented = np.ones_like(z_for_thrivelines_demented)* be_map[model_be]
    
    
    
    # ## thrivelines
    q_for_thrivelines_demented = model.get_mcmc_quantiles(X = x_for_thrivelines_demented,batch_effects=be_for_thrivelines_demented, z_scores=z_for_thrivelines_demented) - sigma_intercept_mu * offsets[model_be]
    q_for_thrivelines_demented = outscaler.inverse_transform(q_for_thrivelines_demented[0])
    
    
    
    #plt.xlim(np.min(x),np.max(x))
    plt.xlim(3,90)
    
    from itertools import batched
    b_size = 3
    for (this_x, this_y) in zip(batched(velocity_list_age, b_size), batched(q_for_thrivelines, b_size)):
         plt.plot(this_x, this_y, linewidth=0.5, color='blue')
    
    for ir, r in enumerate(ran):
        
        if r == 0:
            plt.plot(x[:,0], q[:,ir], color = 'black')
            #plt.plot(x[:,0], test_q2[0:100,ir+1], color = 'red')
            #plt.plot(synth_velocity , test[0:7: ((i+1)*10, i])
            
        elif abs(r) == 3:
            plt.plot(x[:,0], q[:,ir], color = 'black', alpha =0.6, linestyle="--", linewidth = 1)
            #plt.plot(x[:,0], test_q2[0:100,ir+1], color = 'red', alpha =0.6, linestyle="--", linewidth = 1)
            # if add_lattice:
            #     plt.plot(x[20:40, 0], test[ir], color = "red",  alpha =0.6, linestyle="--", linewidth = 1)
        else:
            plt.plot(x[:,0], q[:,ir], color = 'black', alpha =0.6, linewidth = 1)
            #plt.plot(x[:,0], test_q2[0:100,ir+1], color = 'red')
            
    lmap = {'blr':'W-BLR','SHASHo':'$\mathcal{S}_o$', 'SHASHb_1':'$\mathcal{S}_{b1}$','SHASHb_2':'$\mathcal{S}_{b2}$',  'Normal':'$\mathcal{N}$'}
    
    plt.plot(np.array([73,75]), q_for_thrivelines_demented, color = 'black', linestyle="--", linewidth = 2)
    #for range(:
              #plt.plot(synth_velocity , test_q2[i*10: ((i+1)*10, i])
        # for area in areas:
        #     n,a,b,c,d,e,f = area
        #     rect = patches.Rectangle((a,b),c,d,label=n, linewidth=1,edgecolor = 'red',facecolor='None',zorder=10)
        #     plt.gca().add_patch(rect)
        #     plt.text(a+e*c, b+f*d, n, color='red',fontsize=16)
        # if len(areas) > 0:
        #     suffix = '_ann'
        # else:
        #     suffix = ''
    #return q  
    plt.title(lmap[model_type] + " " + measure, fontsize = 16)
    plt.xlabel('Age',fontsize=15)
    #     plt.ylabel(feature,fontsize=12)
    plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(linestyle=":", linewidth=1, alpha=0.7)
    fig.axes[0].yaxis.offsetText.set_fontsize(14)
    plt.savefig(f"{velocity_dir}/velocity_plot_{measure}_{model_type}.png",bbox_inches='tight',dpi=300)
    plt.show()
    # del model 
    # del model_be
    # del model_be_long
    # del this_Y_train
    # #del this_Y_test
    # del this_offset_intercept
    # #del this_scaled_X_test
    # del this_scaled_X_train
    # #del this_scaled_Y_test
    # del this_scaled_Y_train
    # #gc.collect()

#%%
import math

import numpy


def recursive_multiply(lst, step_results=None, current_product=1):
    #lst.insert(0, start_z)
    if step_results is None:
        step_results = []

    # Base case: if the list is empty, return the list of step results
    if len(lst) == 0:
        return step_results

    
    # Multiply current product with the first element
    current_product *= lst[0]

    # Save the current product in step results
    step_results.append(current_product)

    # Recursively process the rest of the list
    return recursive_multiply(lst[1:], step_results, current_product)
#

def get_thrive_lines(lst, step_results=None, current_result=1, z_thrive = (-1.96) ):
    if step_results is None:
        step_results = []
        step_results.append(current_result)

    # Base case: if the list is empty, return the list of step results
    if len(lst) == 0:
        return step_results

    # Apply the custom formula to the first element
    list_item = lst[0]
    current_result = current_result * list_item + math.sqrt(1 - list_item**2) * z_thrive

    # Save the current result in step results
    step_results.append(current_result)

    # Recursively process the rest of the list
    return get_thrive_lines(lst[1:], step_results, current_result, z_thrive=z_thrive)


#
def flatten(xss):
    return [x for xs in xss for x in xs]

#

def reverse_z_transform(z, mu, sigma):
    x = mu + z*sigma
    return x


def fisher_transform(cor: float) -> float:
    """Apply Fisher transformation to correlation values."""
    if cor <= -1:
        cor = -0.9999999999999
    elif cor >= 1:
        cor = 0.9999999999999
    return 0.5 * np.log((1 + cor) / (1 - cor))

def make_correlation_matrix(data: pd.DataFrame, measure: str, re_estimate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a correlation matrix for age-based data.

    For each pair i, j, of ages:
        - Find all subjects that have data for both ages
        - Compute the correlation between the two measures for those subjects
        - Store the correlation in the matrix at position (i, j)

    Parameters:
    - data (pd.DataFrame): Data containing age, subject ID, visits, and measure columns.
    - measure (str): The column name of the measure to compute correlations.
    - re_estimate (bool): Whether to compute the correlation matrix.

    Returns:
    - A (np.ndarray): The correlation matrix.
    - Phi (np.ndarray): The Fisher-transformed correlation matrix.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")
    if not isinstance(measure, str):
        raise TypeError("Input 'measure' must be a string.")
    if not isinstance(re_estimate, bool):
        raise TypeError("Input 're_estimate' must be a boolean.")

    if not re_estimate:
        return np.array([]), np.array([])

    num_ages = 99  # Assuming ages go from 0 to 98
    A = np.full((num_ages, num_ages), np.nan, dtype=float)
    Phi = np.full((num_ages, num_ages), np.nan, dtype=float)

    for i in range(num_ages):
        for j in range(num_ages):
            vec1 = data[data["age"] == i]
            vec2 = data[data["age"] == j]

            overlap = vec1.merge(vec2, on="ID_subject", how="inner")

            overlap_x = overlap.filter(regex='_x')
            overlap_x["ID_subject"] = overlap["ID_subject"]
            overlap_y = overlap.filter(regex='_y')
            overlap_y["ID_subject"] = overlap["ID_subject"]

            overlap_x = overlap_x.sort_values(['ID_subject', 'ID_visit_x']).reset_index(drop=True)
            overlap_y = overlap_y.sort_values(['ID_subject', 'ID_visit_y']).reset_index(drop=True)

            overlap_combined = pd.concat([overlap_x, overlap_y], axis=1).drop_duplicates()

            if len(overlap_combined) < 4:
                cor = np.nan
            else:
                cor = overlap_combined[f"{measure}_x"].corr(overlap_combined[f"{measure}_y"])

            A[i, j] = cor
            Phi[i, j] = fisher_transform(cor)

    # Visualize the correlation matrix
    plt.imshow(A, cmap='viridis')
    plt.colorbar(label="Correlation")
    plt.title("Correlation Matrix")
    plt.xlabel("Age")
    plt.ylabel("Age")
    plt.show()
    
    with open(os.path.join(velocity_dir,'Cor_matrix.pkl'),'wb') as f: pickle.dump(A, f)
    with open(os.path.join(velocity_dir,'Phi_matrix.pkl'),'wb') as f: pickle.dump(Phi, f)

    return A, Phi

# step 5 van Buuren correlation
def Fisher_transform(r):
    '''Takes a correlation r and returns the Fisher-transformed value Phi.'''
    Phi = 0.5*math.log((1+r)/(1-r))
    return Phi


def get_subdiagonal(A, gap=1):
    """   

    Parameters
    ----------
    A : correlation matrix.
    gap : offset

    Returns
    -------
    sub_diagonal : diagonal below the sub-diagnonal

    """
    sub_diagonal = list()
    for j in range(99-gap):
        number = A[j+gap, j]
        sub_diagonal.append(number)
    return sub_diagonal 

def predicted_r(Phi):
    """
    
    Parameters
    ----------
    Phi : member of type list
        Fisher transformed correlation valuee.

    Returns
    -------
    r_pred : single value
        correlation value in r space.

    """
    r_pred = (np.exp(2* Phi)-1)/(np.exp(2* Phi)+1)
    return r_pred

def create_regression_matrix(Phi: np.ndarray, gap: int) -> pd.DataFrame:
    """Create a regression matrix for a given diagonal gap."""
    diagonal = get_subdiagonal(Phi, gap=gap)
    reg_matrix = pd.DataFrame(diagonal, columns=["y"])
    
    # Define age range
    Age = np.linspace(0, 98 - gap, 99 - gap)[:, None]
    
    reg_matrix["V1"] = np.log(Age + (gap / 2))
    reg_matrix["V2"] = np.log(gap)
    reg_matrix["V3"] = 1 / gap
    reg_matrix["V4"] = reg_matrix["V1"] * reg_matrix["V2"]
    reg_matrix["V5"] = reg_matrix["V1"] ** 2
    return reg_matrix

def recursive_r_transform(predicted_values: np.ndarray) -> list[float]:
    """Apply inverse Fisher transformation and return correlation coefficients."""
    return [predicted_r(val) for val in predicted_values]


def generate_thrive_lines(
    pred_r: list,
    start_year: int,
    end_year: int,
    velocity_length: int,
    start_z: int,
    end_z: int,
    space_between_anchors: int,
    z_thrive: float = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate thrive lines for downward or upward trends.

    Parameters:
    - pred_r (list): List of predicted correlation values.
    - start_year (int): Starting year.
    - end_year (int): Ending year.
    - velocity_length (int): Length of years thirve lines should predict in the futurw
    - start_z (int): Lowest (starting) z value, for example -3
    - end_z (int): Higest (ending)  z value, for example 4 
    - space_between_anchors (int): Spacing between anchors, for example thrive lines start every (1), every second (2) year etc.
    - z_thrive (float, optional): Additional z thrive factor for upward thrive. default is -1.96

    Returns:
    - velocity_list_age (np.ndarray): Flattened array of years for velocity.
    - velocity (np.ndarray): Flattened array of thrive line results.
    """
    if not isinstance(pred_r, list):
        raise TypeError("pred_r must be a list.")
    if not all(isinstance(arg, int) for arg in [start_year, end_year, velocity_length, start_z, end_z, space_between_anchors]):
        raise TypeError("start_year, end_year, velocity_length, start_z, end_z, and space_between_anchors must be integers.")
    if z_thrive is not None and not isinstance(z_thrive, (int, float)):
        raise TypeError("z_thrive must be a float or an integer if provided.")

    lists_collection = []
    year_collection_long = []

    for i in range(start_year, end_year, space_between_anchors):
        pred_r_snippet = pred_r[i : i + (velocity_length - 1)]
        age_range = np.arange(i, i + velocity_length)

        for j in range(start_z, end_z):
            if z_thrive is not None:
                step_results_thrive = list(get_thrive_lines(pred_r_snippet, current_result=j, z_thrive=z_thrive))
            else:
                step_results_thrive = list(get_thrive_lines(pred_r_snippet, current_result=j))

            lists_collection.append(step_results_thrive)
            year_collection_long.append(age_range)

    velocity_list_age = np.array(flatten(year_collection_long))
    velocity = np.array(flatten(lists_collection))

    # Trim velocity_list_age to match the length of velocity
    velocity_list_age = velocity_list_age[:len(velocity)]

    return velocity_list_age, velocity

idps= sc_idp_cols()
for idp_nr, measure in enumerate(idps):
    if  idp_nr == 1:
        break
    make_velocity_plots(idp_nr, measure, seletced_sex = 'male', model_type="SHASHb_1", re_estimate=True)
