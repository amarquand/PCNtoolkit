#!/.../anaconda/bin/python/

# -----------------------------------------------------------------------------
# Run parallel normative modelling.
# All processing takes place in the processing directory (processing_dir)
# All inputs should be text files or binaries and space seperated
#
# It is possible to run these functions using...
#
# * k-fold cross-validation
# * estimating a training dataset then applying to a second test dataset
#
# First,the data is split for parallel processing.
# Second, the splits are submitted to the cluster.
# Third, the output is collected and combined.
#
# witten by (primarily) T Wolfers, (adaptated) SM Kia, H Huijsdens, L Parks, 
# AF Marquand
# -----------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division

import os
import sys
import glob
import shutil
import pickle
import fileinput
import numpy as np
import pandas as pd
from subprocess import call

try:
    import pcntoolkit as ptk
    import pcntoolkit.dataio.fileio as fileio
    from pcntoolkit import configs
    ptkpath = ptk.__path__[0] 
except ImportError:
    pass
    ptkpath = os.path.abspath(os.path.dirname(__file__))
    if ptkpath not in sys.path:
        sys.path.append(ptkpath)
    import dataio.fileio as fileio
    import configs
    
    
PICKLE_PROTOCOL = configs.PICKLE_PROTOCOL


def execute_nm(processing_dir,
               python_path,
               job_name,
               covfile_path,
               respfile_path,
               batch_size,
               memory,
               duration,
               normative_path=None,
               func='estimate',
               **kwargs):

    """
    This function is a mother function that executes all parallel normative
    modelling routines. Different specifications are possible using the sub-
    functions.

    :Parameters:
        * processing_dir     -> Full path to the processing dir
        * python_path        -> Full path to the python distribution
        * normative_path     -> Full path to the normative.py. If None (default)
                                then it will automatically retrieves the path from 
                                the installed packeage.
        * job_name           -> Name for the bash script that is the output of
                                this function
        * covfile_path       -> Full path to a .txt file that contains all
                                covariats (subjects x covariates) for the
                                responsefile
        * respfile_path      -> Full path to a .txt that contains all features
                                (subjects x features)
        * batch_size         -> Number of features in each batch
        * memory             -> Memory requirements written as string
                                for example 4gb or 500mb
        * duation            -> The approximate duration of the job, a string
                                with HH:MM:SS for example 01:01:01
        * cv_folds           -> Number of cross validations
        * testcovfile_path   -> Full path to a .txt file that contains all
                                covariats (subjects x covariates) for the
                                testresponse file
        * testrespfile_path  -> Full path to a .txt file that contains all
                                test features
        * log_path           -> Pathfor saving log files
        * binary             -> If True uses binary format for response file
                                otherwise it is text

    written by (primarily) T Wolfers, (adapted) SM Kia
    """
    
    if normative_path is None:
        normative_path = ptkpath + '/normative.py'
        
    cv_folds = kwargs.get('cv_folds', None)
    testcovfile_path = kwargs.get('testcovfile_path', None)
    testrespfile_path= kwargs.get('testrespfile_path', None)
    cluster_spec = kwargs.pop('cluster_spec', 'torque')
    log_path = kwargs.pop('log_path', None)
    binary = kwargs.pop('binary', False)
    
    split_nm(processing_dir,
             respfile_path,
             batch_size,
             binary,
             **kwargs)

    batch_dir = glob.glob(processing_dir + 'batch_*')
    # print(batch_dir)
    number_of_batches = len(batch_dir)
    # print(number_of_batches)

    if binary:
        file_extentions = '.pkl'
    else:
        file_extentions = '.txt'
    
    kwargs.update({'batch_size':str(batch_size)})
    for n in range(1, number_of_batches+1):
        print(n)
        kwargs.update({'job_id':str(n)})
        if testrespfile_path is not None:
            if cv_folds is not None:
                raise(ValueError, """If the response file is specified
                                     cv_folds must be equal to None""")
            else:
                # specified train/test split
                batch_processing_dir = processing_dir + 'batch_' + str(n) + '/'
                batch_job_name = job_name + '_' + str(n) + '.sh'
                batch_respfile_path = (batch_processing_dir + 'resp_batch_' +
                                       str(n) + file_extentions)
                batch_testrespfile_path = (batch_processing_dir +
                                           'testresp_batch_' +
                                           str(n) + file_extentions)
                batch_job_path = batch_processing_dir + batch_job_name
                if cluster_spec == 'torque':
                    # update the response file 
                    kwargs.update({'testrespfile_path': \
                                   batch_testrespfile_path})
                    bashwrap_nm(batch_processing_dir,
                                python_path,
                                normative_path,
                                batch_job_name,
                                covfile_path,
                                batch_respfile_path,
                                func=func,
                                **kwargs)
                    qsub_nm(job_path=batch_job_path,
                            log_path=log_path,
                            memory=memory,
                            duration=duration)
                elif cluster_spec == 'sbatch':
                    # update the response file 
                    kwargs.update({'testrespfile_path': \
                                   batch_testrespfile_path})
                    sbatchwrap_nm(batch_processing_dir,
                                python_path,
                                normative_path,
                                batch_job_name,
                                covfile_path,
                                batch_respfile_path,
                                func=func,
                                memory=memory,
                                duration=duration,
                                **kwargs)
                    sbatch_nm(job_path=batch_job_path,
                            log_path=log_path)
                elif cluster_spec == 'new':
                    # this part requires addition in different envioronment [
                    sbatchwrap_nm(processing_dir=batch_processing_dir, 
                                  func=func, **kwargs)
                    sbatch_nm(processing_dir=batch_processing_dir)
                    # ]
        if testrespfile_path is None:
            if testcovfile_path is not None:
                # forward model
                batch_processing_dir = processing_dir + 'batch_' + str(n) + '/'
                batch_job_name = job_name + '_' + str(n) + '.sh'
                batch_respfile_path = (batch_processing_dir + 'resp_batch_' +
                                       str(n) + file_extentions)
                batch_job_path = batch_processing_dir + batch_job_name
                if cluster_spec == 'torque':
                    bashwrap_nm(batch_processing_dir,
                                python_path,
                                normative_path,
                                batch_job_name,
                                covfile_path,
                                batch_respfile_path,
                                func=func,
                                **kwargs)
                    qsub_nm(job_path=batch_job_path,
                            log_path=log_path,
                            memory=memory,
                            duration=duration)
                elif cluster_spec == 'sbatch':
                    sbatchwrap_nm(batch_processing_dir,
                                python_path,
                                normative_path,
                                batch_job_name,
                                covfile_path,
                                batch_respfile_path,
                                func=func,
                                memory=memory,
                                duration=duration,
                                **kwargs)
                    sbatch_nm(job_path=batch_job_path,
                              log_path=log_path)
                elif cluster_spec == 'new':
                    # this part requires addition in different envioronment [
                    bashwrap_nm(processing_dir=batch_processing_dir, func=func,
                                **kwargs)
                    qsub_nm(processing_dir=batch_processing_dir)
                    # ]
            else:
                # cross-validation
                batch_processing_dir = (processing_dir + 'batch_' +
                                        str(n) + '/')
                batch_job_name = job_name + '_' + str(n) + '.sh'
                batch_respfile_path = (batch_processing_dir +
                                       'resp_batch_' + str(n) +
                                       file_extentions)
                batch_job_path = batch_processing_dir + batch_job_name
                if cluster_spec == 'torque':
                    bashwrap_nm(batch_processing_dir,
                                python_path,
                                normative_path,
                                batch_job_name,
                                covfile_path,
                                batch_respfile_path,
                                func=func,
                                **kwargs)
                    qsub_nm(job_path=batch_job_path,
                            log_path=log_path,
                            memory=memory,
                            duration=duration)
                elif cluster_spec == 'sbatch':
                    sbatchwrap_nm(batch_processing_dir,
                                python_path,
                                normative_path,
                                batch_job_name,
                                covfile_path,
                                batch_respfile_path,
                                func=func,
                                memory=memory,
                                duration=duration,
                                **kwargs)
                    sbatch_nm(job_path=batch_job_path,
                            log_path=log_path)
                elif cluster_spec == 'new':
                    # this part requires addition in different envioronment [
                    bashwrap_nm(processing_dir=batch_processing_dir, func=func,
                                **kwargs)
                    qsub_nm(processing_dir=batch_processing_dir)
                    # ]


"""routines that are environment independent"""


def split_nm(processing_dir,
             respfile_path,
             batch_size,
             binary,
             **kwargs):

    """ This function prepares the input files for normative_parallel.

    :Parameters:
        * processing_dir    -> Full path to the folder of processing
        * respfile_path     -> Full path to the responsefile.txt
                               (subjects x features)
        * batch_size        -> Number of features in each batch
        * testrespfile_path -> Full path to the test responsefile.txt
                               (subjects x features)
        * binary            -> If True binary file

    :outputs:
        * The creation of a folder struture for batch-wise processing

    witten by (primarily) T Wolfers (adapted) SM Kia
    """
    
    testrespfile_path = kwargs.pop('testrespfile_path', None)

    dummy, respfile_extension = os.path.splitext(respfile_path)
    if (binary and respfile_extension != '.pkl'):
        raise(ValueError, """If binary is True the file format for the
              testrespfile file must be .pkl""")
    elif (binary==False and respfile_extension != '.txt'):
        raise(ValueError, """If binary is False the file format for the
              testrespfile file must be .txt""")

    # splits response into batches
    if testrespfile_path is None:
        if (binary==False):
            respfile = fileio.load_ascii(respfile_path)
        else:
            respfile = pd.read_pickle(respfile_path)

        respfile = pd.DataFrame(respfile)

        numsub = respfile.shape[1]
        batch_vec = np.arange(0,
                              numsub,
                              batch_size)
        batch_vec = np.append(batch_vec,
                              numsub)
        
        for n in range(0, (len(batch_vec) - 1)):
            resp_batch = respfile.iloc[:, (batch_vec[n]): batch_vec[n + 1]]
            os.chdir(processing_dir)
            resp = str('resp_batch_' + str(n+1))
            batch = str('batch_' + str(n+1))
            if not os.path.exists(processing_dir + batch):
                os.makedirs(processing_dir + batch)
                os.makedirs(processing_dir + batch + '/Models/')
            if (binary==False):
                fileio.save_pd(resp_batch,
                               processing_dir + batch + '/' +
                               resp + '.txt')
            else:
                resp_batch.to_pickle(processing_dir + batch + '/' +
                                     resp + '.pkl', protocol=PICKLE_PROTOCOL)

    # splits response and test responsefile into batches
    else:
        dummy, testrespfile_extension = os.path.splitext(testrespfile_path)
        if (binary and testrespfile_extension != '.pkl'):
            raise(ValueError, """If binary is True the file format for the
                  testrespfile file must be .pkl""")
        elif(binary==False and testrespfile_extension != '.txt'):
            raise(ValueError, """If binary is False the file format for the
                  testrespfile file must be .txt""")

        if (binary==False):
            respfile = fileio.load_ascii(respfile_path)
            testrespfile = fileio.load_ascii(testrespfile_path)
        else:
            respfile = pd.read_pickle(respfile_path)
            testrespfile = pd.read_pickle(testrespfile_path)

        respfile = pd.DataFrame(respfile)
        testrespfile = pd.DataFrame(testrespfile)

        numsub = respfile.shape[1]
        batch_vec = np.arange(0, numsub,
                              batch_size)
        batch_vec = np.append(batch_vec,
                              numsub)
        for n in range(0, (len(batch_vec) - 1)):
            resp_batch = respfile.iloc[:, (batch_vec[n]): batch_vec[n + 1]]
            testresp_batch = testrespfile.iloc[:, (batch_vec[n]): batch_vec[n +
                                             1]]
            os.chdir(processing_dir)
            resp = str('resp_batch_' + str(n+1))
            testresp = str('testresp_batch_' + str(n+1))
            batch = str('batch_' + str(n+1))
            if not os.path.exists(processing_dir + batch):
                os.makedirs(processing_dir + batch)
                os.makedirs(processing_dir + batch + '/Models/')
            if (binary==False):
                fileio.save_pd(resp_batch,
                               processing_dir + batch + '/' +
                               resp + '.txt')
                fileio.save_pd(testresp_batch,
                               processing_dir + batch + '/' + testresp +
                               '.txt')
            else:
                resp_batch.to_pickle(processing_dir + batch + '/' +
                                     resp + '.pkl', protocol=PICKLE_PROTOCOL)
                testresp_batch.to_pickle(processing_dir + batch + '/' +
                                         testresp + '.pkl', 
                                         protocol=PICKLE_PROTOCOL)


def collect_nm(processing_dir,
               job_name,
               func='estimate',
               collect=False,
               binary=False,
               batch_size=None,
               outputsuffix='_estimate'):
    
    """This function checks and collects all batches.

    :Parameters:
        * processing_dir        -> Full path to the processing directory
        * collect               -> If True data is checked for failed batches
                                and collected; if False data is just checked
        * binary                -> Results in pkl format? 

    :ouptuts:
        * Text files containing all results accross all batches the combined
          output (written to disk)
        * returns 0 if batches fail, 1 otherwise

    written by (primarily) T Wolfers, (adapted) SM Kia
    """

    if binary:
        file_extentions = '.pkl'
    else:
        file_extentions = '.txt'

    # detect number of subjects, batches, hyperparameters and CV
    batches = glob.glob(processing_dir + 'batch_*/')
    
    count = 0
    batch_fail = []
    
    if func != 'fit':
        file_example = []
        # TODO: Collect_nm only depends on yhat, thus does not work when no 
        # prediction is made (when test cov is not specified). 
        for batch in batches:
            if file_example == []:
                file_example = glob.glob(batch + 'yhat' + outputsuffix 
                                         + file_extentions)
            else:
                break
        if binary is False:
            file_example = fileio.load(file_example[0])
        else:
            file_example = pd.read_pickle(file_example[0])
        numsubjects = file_example.shape[0]
        batch_size = file_example.shape[1]
    
        # artificially creates files for batches that were not executed
        batch_dirs = glob.glob(processing_dir + 'batch_*/')
        batch_dirs = fileio.sort_nicely(batch_dirs)
        for batch in batch_dirs:
            filepath = glob.glob(batch + 'yhat' + outputsuffix + '*')
            if filepath == []:
                count = count+1
                batch1 = glob.glob(batch + '/' + job_name + '*.sh')
                print(batch1)
                batch_fail.append(batch1)
                if collect is True:
                    pRho = np.ones(batch_size)
                    pRho = pRho.transpose()
                    pRho = pd.Series(pRho)
                    fileio.save(pRho, batch + 'pRho' + outputsuffix + 
                                file_extentions)
                    
                    Rho = np.zeros(batch_size)
                    Rho = Rho.transpose()
                    Rho = pd.Series(Rho)
                    fileio.save(Rho, batch + 'Rho' + outputsuffix + 
                                file_extentions)
                    
                    rmse = np.zeros(batch_size)
                    rmse = rmse.transpose()
                    rmse = pd.Series(rmse)
                    fileio.save(rmse, batch + 'RMSE' + outputsuffix + 
                                file_extentions)
                    
                    smse = np.zeros(batch_size)
                    smse = smse.transpose()
                    smse = pd.Series(smse)
                    fileio.save(smse, batch + 'SMSE' + outputsuffix + 
                                file_extentions)
                    
                    expv = np.zeros(batch_size)
                    expv = expv.transpose()
                    expv = pd.Series(expv)
                    fileio.save(expv, batch + 'EXPV' + outputsuffix + 
                                file_extentions)
                    
                    msll = np.zeros(batch_size)
                    msll = msll.transpose()
                    msll = pd.Series(msll)
                    fileio.save(msll, batch + 'MSLL' + outputsuffix + 
                                file_extentions)
    
                    yhat = np.zeros([numsubjects, batch_size])
                    yhat = pd.DataFrame(yhat)
                    fileio.save(yhat, batch + 'yhat' + outputsuffix + 
                                file_extentions)
    
                    ys2 = np.zeros([numsubjects, batch_size])
                    ys2 = pd.DataFrame(ys2)
                    fileio.save(ys2, batch + 'ys2' + outputsuffix + 
                                file_extentions)
    
                    Z = np.zeros([numsubjects, batch_size])
                    Z = pd.DataFrame(Z)
                    fileio.save(Z, batch + 'Z' + outputsuffix + 
                                file_extentions)
                    
                    nll = np.zeros(batch_size)
                    nll = nll.transpose()
                    nll = pd.Series(nll)
                    fileio.save(nll, batch + 'NLL' + outputsuffix + 
                                file_extentions)
                    
                    bic = np.zeros(batch_size)
                    bic = bic.transpose()
                    bic = pd.Series(bic)
                    fileio.save(bic, batch + 'BIC' + outputsuffix + 
                                file_extentions)
    
                    if not os.path.isdir(batch + 'Models'):
                        os.mkdir('Models')
                        
                        
            else: # if more than 10% of yhat is nan then it is a failed batch
                yhat = fileio.load(filepath[0])
                if np.count_nonzero(~np.isnan(yhat))/(np.prod(yhat.shape))<0.9:
                    count = count+1
                    batch1 = glob.glob(batch + '/' + job_name + '*.sh')
                    print('More than 10% nans in '+ batch1[0])
                    batch_fail.append(batch1)
    
    else:
        batch_dirs = glob.glob(processing_dir + 'batch_*/')
        batch_dirs = fileio.sort_nicely(batch_dirs)
        for batch in batch_dirs:
            filepath = glob.glob(batch + 'Models/' + 'NM_' + '*' + outputsuffix 
                                 + '*')
            if len(filepath) < batch_size:
                count = count+1
                batch1 = glob.glob(batch + '/' + job_name + '*.sh')
                print(batch1)
                batch_fail.append(batch1)
    
    # combines all output files across batches
    if collect is True:
        pRho_filenames = glob.glob(processing_dir + 'batch_*/' + 'pRho' + 
                                   outputsuffix + '*')
        if pRho_filenames:
            pRho_filenames = fileio.sort_nicely(pRho_filenames)
            pRho_dfs = []
            for pRho_filename in pRho_filenames:
                pRho_dfs.append(pd.DataFrame(fileio.load(pRho_filename)))
            pRho_dfs = pd.concat(pRho_dfs, ignore_index=True, axis=0)
            fileio.save(pRho_dfs, processing_dir + 'pRho' + outputsuffix +
                        file_extentions)
            del pRho_dfs

        Rho_filenames = glob.glob(processing_dir + 'batch_*/' + 'Rho' + 
                                   outputsuffix + '*')
        if Rho_filenames:
            Rho_filenames = fileio.sort_nicely(Rho_filenames)
            Rho_dfs = []
            for Rho_filename in Rho_filenames:
                Rho_dfs.append(pd.DataFrame(fileio.load(Rho_filename)))
            Rho_dfs = pd.concat(Rho_dfs, ignore_index=True, axis=0)
            fileio.save(Rho_dfs, processing_dir + 'Rho' + outputsuffix +
                        file_extentions)
            del Rho_dfs

        Z_filenames = glob.glob(processing_dir + 'batch_*/' + 'Z' + 
                                   outputsuffix + '*')
        if Z_filenames:
            Z_filenames = fileio.sort_nicely(Z_filenames)
            Z_dfs = []
            for Z_filename in Z_filenames:
                Z_dfs.append(pd.DataFrame(fileio.load(Z_filename)))
            Z_dfs = pd.concat(Z_dfs, ignore_index=True, axis=1)
            fileio.save(Z_dfs, processing_dir + 'Z' + outputsuffix +
                        file_extentions)
            del Z_dfs
            
        yhat_filenames = glob.glob(processing_dir + 'batch_*/' + 'yhat' + 
                                   outputsuffix + '*')
        if yhat_filenames:
            yhat_filenames = fileio.sort_nicely(yhat_filenames)
            yhat_dfs = []
            for yhat_filename in yhat_filenames:
                yhat_dfs.append(pd.DataFrame(fileio.load(yhat_filename)))
            yhat_dfs = pd.concat(yhat_dfs, ignore_index=True, axis=1)
            fileio.save(yhat_dfs, processing_dir + 'yhat' + outputsuffix +
                        file_extentions)
            del yhat_dfs

        ys2_filenames = glob.glob(processing_dir + 'batch_*/' + 'ys2' + 
                                   outputsuffix + '*')
        if ys2_filenames:
            ys2_filenames = fileio.sort_nicely(ys2_filenames)
            ys2_dfs = []
            for ys2_filename in ys2_filenames:
                ys2_dfs.append(pd.DataFrame(fileio.load(ys2_filename)))
            ys2_dfs = pd.concat(ys2_dfs, ignore_index=True, axis=1)
            fileio.save(ys2_dfs, processing_dir + 'ys2' + outputsuffix +
                        file_extentions)
            del ys2_dfs

        rmse_filenames = glob.glob(processing_dir + 'batch_*/' + 'RMSE' + 
                                   outputsuffix + '*')
        if rmse_filenames:
            rmse_filenames = fileio.sort_nicely(rmse_filenames)
            rmse_dfs = []
            for rmse_filename in rmse_filenames:
                rmse_dfs.append(pd.DataFrame(fileio.load(rmse_filename)))
            rmse_dfs = pd.concat(rmse_dfs, ignore_index=True, axis=0)
            fileio.save(rmse_dfs, processing_dir + 'RMSE' + outputsuffix +
                        file_extentions)
            del rmse_dfs

        smse_filenames = glob.glob(processing_dir + 'batch_*/' + 'SMSE' + 
                                   outputsuffix + '*')
        if smse_filenames:
            smse_filenames = fileio.sort_nicely(smse_filenames)
            smse_dfs = []
            for smse_filename in smse_filenames:
                smse_dfs.append(pd.DataFrame(fileio.load(smse_filename)))
            smse_dfs = pd.concat(smse_dfs, ignore_index=True, axis=0)
            fileio.save(smse_dfs, processing_dir + 'SMSE' + outputsuffix +
                        file_extentions)
            del smse_dfs
            
        expv_filenames = glob.glob(processing_dir + 'batch_*/' + 'EXPV' + 
                                   outputsuffix + '*')
        if expv_filenames:
            expv_filenames = fileio.sort_nicely(expv_filenames)
            expv_dfs = []
            for expv_filename in expv_filenames:
                expv_dfs.append(pd.DataFrame(fileio.load(expv_filename)))
            expv_dfs = pd.concat(expv_dfs, ignore_index=True, axis=0)
            fileio.save(expv_dfs, processing_dir + 'EXPV' + outputsuffix +
                        file_extentions)
            del expv_dfs
            
        msll_filenames = glob.glob(processing_dir + 'batch_*/' + 'MSLL' + 
                                   outputsuffix + '*')
        if msll_filenames:
            msll_filenames = fileio.sort_nicely(msll_filenames)
            msll_dfs = []
            for msll_filename in msll_filenames:
                msll_dfs.append(pd.DataFrame(fileio.load(msll_filename)))
            msll_dfs = pd.concat(msll_dfs, ignore_index=True, axis=0)
            fileio.save(msll_dfs, processing_dir + 'MSLL' + outputsuffix +
                        file_extentions)
            del msll_dfs
            
        nll_filenames = glob.glob(processing_dir + 'batch_*/' + 'NLL' +
                                  outputsuffix + '*')
        if nll_filenames:
            nll_filenames = fileio.sort_nicely(nll_filenames)
            nll_dfs = []
            for nll_filename in nll_filenames:
                nll_dfs.append(pd.DataFrame(fileio.load(nll_filename)))
            nll_dfs = pd.concat(nll_dfs, ignore_index=True, axis=0)
            fileio.save(nll_dfs, processing_dir + 'NLL' + outputsuffix +
                        file_extentions)
            del nll_dfs

        bic_filenames = glob.glob(processing_dir + 'batch_*/' + 'BIC' +
                                  outputsuffix + '*')
        if bic_filenames:
            bic_filenames = fileio.sort_nicely(bic_filenames)
            bic_dfs = []
            for bic_filename in bic_filenames:
                bic_dfs.append(pd.DataFrame(fileio.load(bic_filename)))
            bic_dfs = pd.concat(bic_dfs, ignore_index=True, axis=0)
            fileio.save(bic_dfs, processing_dir + 'BIC' + outputsuffix +
                        file_extentions)
            del bic_dfs
        
        if func != 'predict' and func != 'extend':
            if not os.path.isdir(processing_dir + 'Models') and \
               os.path.exists(os.path.join(batches[0], 'Models')):
                os.mkdir(processing_dir + 'Models')
                
            meta_filenames = glob.glob(processing_dir + 'batch_*/Models/' + 
                                       'meta_data.md')
            mY = []
            sY = []
            X_scalers = []
            Y_scalers = []
            if meta_filenames:
                meta_filenames = fileio.sort_nicely(meta_filenames)
                with open(meta_filenames[0], 'rb') as file:
                    meta_data = pickle.load(file)
                
                for meta_filename in meta_filenames:
                    with open(meta_filename, 'rb') as file:
                        meta_data = pickle.load(file)
                    mY.append(meta_data['mean_resp'])
                    sY.append(meta_data['std_resp'])
                    if meta_data['inscaler'] in ['standardize', 'minmax', 
                                'robminmax']:
                        X_scalers.append(meta_data['scaler_cov'])
                    if meta_data['outscaler'] in ['standardize', 'minmax', 
                                'robminmax']:
                        Y_scalers.append(meta_data['scaler_resp'])
                meta_data['mean_resp'] = np.squeeze(np.stack(mY)) 
                meta_data['std_resp'] = np.squeeze(np.stack(sY))
                meta_data['scaler_cov'] = X_scalers 
                meta_data['scaler_resp'] = Y_scalers
                
                with open(os.path.join(processing_dir, 'Models', 
                                       'meta_data.md'), 'wb') as file:
                    pickle.dump(meta_data, file, protocol=PICKLE_PROTOCOL)
            
            batch_dirs = glob.glob(processing_dir + 'batch_*/')
            if batch_dirs:
                batch_dirs = fileio.sort_nicely(batch_dirs)
                for b, batch_dir in enumerate(batch_dirs):
                    src_files = glob.glob(batch_dir + 'Models/NM*' + 
                                          outputsuffix + '.pkl')
                    if src_files:
                        src_files = fileio.sort_nicely(src_files)
                        for f, full_file_name in enumerate(src_files):
                            if os.path.isfile(full_file_name):
                                file_name = full_file_name.split('/')[-1]
                                n = file_name.split('_')
                                n[-2] = str(b * batch_size + f)
                                n = '_'.join(n)
                                shutil.copy(full_file_name, processing_dir + 
                                            'Models/' + n)
                    elif func=='fit':
                        count = count+1
                        batch1 = glob.glob(batch_dir + '/' + job_name + '*.sh')
                        print('Failed batch: ' + batch1[0])
                        batch_fail.append(batch1)
                        
    # list batches that were not executed
    print('Number of batches that failed:' + str(count))
    batch_fail_df = pd.DataFrame(batch_fail)
    if file_extentions == '.txt':
        fileio.save_pd(batch_fail_df, processing_dir + 'failed_batches'+
                file_extentions)
    else:
        fileio.save(batch_fail_df, processing_dir +
            'failed_batches' +
            file_extentions)

    if not batch_fail:
        return 1
    else:
        return 0

def delete_nm(processing_dir,
              binary=False):
    """This function deletes all processing for normative modelling and just
    keeps the combined output.

    :Parameters:
        * processing_dir        -> Full path to the processing directory
        * binary                -> Results in pkl format? 

    written by (primarily) T Wolfers, (adapted) SM Kia
    """
    
    if binary:
        file_extentions = '.pkl'
    else:
        file_extentions = '.txt'
    for file in glob.glob(processing_dir + 'batch_*/'):
        shutil.rmtree(file)
    if os.path.exists(processing_dir + 'failed_batches' + file_extentions):
        os.remove(processing_dir + 'failed_batches' + file_extentions)


# all routines below are envronment dependent and require adaptation in novel
# environments -> copy those routines and adapt them in accrodance with your
# environment

def bashwrap_nm(processing_dir,
                python_path,
                normative_path,
                job_name,
                covfile_path,
                respfile_path,
                func='estimate',
                **kwargs):

    """ This function wraps normative modelling into a bash script to run it
    on a torque cluster system.

    :Parameters:
        * processing_dir     -> Full path to the processing dir
        * python_path        -> Full path to the python distribution
        * normative_path     -> Full path to the normative.py
        * job_name           -> Name for the bash script that is the output of
                                this function
        * covfile_path       -> Full path to a .txt file that contains all
                                covariats (subjects x covariates) for the
                                responsefile
        * respfile_path      -> Full path to a .txt that contains all features
                                (subjects x features)
        * cv_folds           -> Number of cross validations
        * testcovfile_path   -> Full path to a .txt file that contains all
                                covariats (subjects x covariates) for the
                                testresponse file
        * testrespfile_path  -> Full path to a .txt file that contains all
                                test features
        * alg                -> which algorithm to use
        * configparam        -> configuration parameters for this algorithm

    :outputs:
        * A bash.sh file containing the commands for normative modelling saved
          to the processing directory (written to disk)

    written by (primarily) T Wolfers
    """
    
    # here we use pop not get to remove the arguments as they used 
    cv_folds = kwargs.pop('cv_folds',None)
    testcovfile_path = kwargs.pop('testcovfile_path', None)
    testrespfile_path = kwargs.pop('testrespfile_path', None)
    alg = kwargs.pop('alg', None)
    configparam = kwargs.pop('configparam', None)
    standardize = kwargs.pop('standardize', True)
    
    # change to processing dir
    os.chdir(processing_dir)
    output_changedir = ['cd ' + processing_dir + '\n']

    bash_lines = '#!/bin/bash\n'
    bash_cores = 'export OMP_NUM_THREADS=1\n'
    bash_environment = [bash_lines + bash_cores]

    # creates call of function for normative modelling
    if (testrespfile_path is not None) and (testcovfile_path is not None):
        job_call = [python_path + ' ' + normative_path + ' -c ' +
                    covfile_path + ' -t ' + testcovfile_path + ' -r ' +
                    testrespfile_path + ' -f ' + func]
    elif (testrespfile_path is None) and (testcovfile_path is not None):
        job_call = [python_path + ' ' + normative_path + ' -c ' +
                    covfile_path + ' -t ' + testcovfile_path + ' -f ' + func]
    elif cv_folds is not None:
        job_call = [python_path + ' ' + normative_path + ' -c ' +
                    covfile_path + ' -k ' + str(cv_folds) +  ' -f ' + func]
    elif func != 'estimate':
        job_call = [python_path + ' ' + normative_path + ' -c ' +
                    covfile_path +  ' -f ' + func]
    else:
        raise(ValueError, """For 'estimate' function either testcov or cvfold
              must be specified.""")
        
    # add algorithm-specific parameters
    if alg is not None:
        job_call = [job_call[0] + ' -a ' + alg]
        if configparam is not None:
            job_call = [job_call[0] + ' -x ' + str(configparam)]
    
    # add standardization flag if it is false
    # if not standardize:
    #     job_call = [job_call[0] + ' -s']
    
    # add responses file
    job_call = [job_call[0] + ' ' + respfile_path]
    
    # add in optional arguments. 
    for k in kwargs:
        job_call = [job_call[0] + ' ' + k + '=' + kwargs[k]]

    # writes bash file into processing dir
    with open(processing_dir+job_name, 'w') as bash_file:
        bash_file.writelines(bash_environment + output_changedir + \
                             job_call + ["\n"])

    # changes permissoins for bash.sh file
    os.chmod(processing_dir + job_name, 0o700)


def qsub_nm(job_path,
            log_path,
            memory,
            duration):
    """
    This function submits a job.sh scipt to the torque custer using the qsub
    command.

    ** Input:
        * job_path      -> Full path to the job.sh file
        * memory        -> Memory requirements written as string for example
                           4gb or 500mb
        * duation       -> The approximate duration of the job, a string with
                           HH:MM:SS for example 01:01:01

    ** Output:
        * Submission of the job to the (torque) cluster

    witten by (primarily) T Wolfers, (adapted) SM Kia
    """

    # created qsub command
    if log_path is None:
        qsub_call = ['echo ' + job_path + ' | qsub -N ' + job_path + ' -l ' +
                     'procs=1' + ',mem=' + memory + ',walltime=' + duration]
    else:
        qsub_call = ['echo ' + job_path + ' | qsub -N ' + job_path +
                     ' -l ' + 'procs=1' + ',mem=' + memory + ',walltime=' + 
                     duration + ' -o ' + log_path + ' -e ' + log_path]

    # submits job to cluster
    call(qsub_call, shell=True)


def rerun_nm(processing_dir,
             log_path,
             memory,
             duration,
             binary=False):
    """
    This function reruns all failed batched in processing_dir after collect_nm
    has identified he failed batches

    * Input:
        * processing_dir        -> Full path to the processing directory
        * memory                -> Memory requirements written as string
                                   for example 4gb or 500mb
        * duration               -> The approximate duration of the job, a
                                   string with HH:MM:SS for example 01:01:01

    written by (primarily) T Wolfers, (adapted) SM Kia
    """

    if binary:
        file_extentions = '.pkl'
        failed_batches = fileio.load(processing_dir +
                                       'failed_batches' + file_extentions)
        shape = failed_batches.shape
        for n in range(0, shape[0]):
            jobpath = failed_batches[n, 0]
            print(jobpath)
            qsub_nm(job_path=jobpath,
                    log_path=log_path,
                    memory=memory,
                    duration=duration)
    else:
        file_extentions = '.txt'
        failed_batches = fileio.load_pd(processing_dir +
                                       'failed_batches' + file_extentions)
        shape = failed_batches.shape
        for n in range(0, shape[0]):
            jobpath = failed_batches.iloc[n, 0]
            print(jobpath)
            qsub_nm(job_path=jobpath,
                    log_path=log_path,
                    memory=memory,
                    duration=duration)

# COPY the rotines above here and aadapt those to your cluster
# bashwarp_nm; qsub_nm; rerun_nm

def sbatchwrap_nm(processing_dir,
                  python_path,
                  normative_path,
                  job_name,
                  covfile_path,
                  respfile_path,
                  memory,
                  duration,
                  func='estimate',
                  **kwargs):

    """ This function wraps normative modelling into a bash script to run it
    on a torque cluster system.

    :Parameters:
        * processing_dir     -> Full path to the processing dir
        * python_path        -> Full path to the python distribution
        * normative_path     -> Full path to the normative.py
        * job_name           -> Name for the bash script that is the output of
                                this function
        * covfile_path       -> Full path to a .txt file that contains all
                                covariats (subjects x covariates) for the
                                responsefile
        * respfile_path      -> Full path to a .txt that contains all features
                                (subjects x features)
        * cv_folds           -> Number of cross validations
        * testcovfile_path   -> Full path to a .txt file that contains all
                                covariats (subjects x covariates) for the
                                testresponse file
        * testrespfile_path  -> Full path to a .txt file that contains all
                                test features
        * alg                -> which algorithm to use
        * configparam        -> configuration parameters for this algorithm

    :outputs:
        * A bash.sh file containing the commands for normative modelling saved
          to the processing directory (written to disk)

    written by (primarily) T Wolfers
    """
    
    # here we use pop not get to remove the arguments as they used 
    cv_folds = kwargs.pop('cv_folds',None)
    testcovfile_path = kwargs.pop('testcovfile_path', None)
    testrespfile_path = kwargs.pop('testrespfile_path', None)
    alg = kwargs.pop('alg', None)
    configparam = kwargs.pop('configparam', None)
    standardize = kwargs.pop('standardize', True)
    
    # change to processing dir
    os.chdir(processing_dir)
    output_changedir = ['cd ' + processing_dir + '\n']

    sbatch_init='#!/bin/bash\n'
    sbatch_jobname='#SBATCH --job-name=' + processing_dir + '\n'
    sbatch_account='#SBATCH --account=p33_norment\n'
    sbatch_nodes='#SBATCH --nodes=1\n'
    sbatch_tasks='#SBATCH --ntasks=1\n'
    sbatch_time='#SBATCH --time=' + str(duration) + '\n'
    sbatch_memory='#SBATCH --mem-per-cpu=' + str(memory) + '\n'
    sbatch_module='module purge\n'
    sbatch_anaconda='module load anaconda3\n'
    sbatch_exit='set -o errexit\n'

    #echo -n "This script is running on "
    #hostname
    
    bash_environment = [sbatch_init + 
                        sbatch_jobname +
                        sbatch_account +
                        sbatch_nodes +
                        sbatch_tasks +
                        sbatch_time +
                        sbatch_module +
                        sbatch_anaconda]

    # creates call of function for normative modelling
    if (testrespfile_path is not None) and (testcovfile_path is not None):
        job_call = [python_path + ' ' + normative_path + ' -c ' +
                    covfile_path + ' -t ' + testcovfile_path + ' -r ' +
                    testrespfile_path + ' -f ' + func]
    elif (testrespfile_path is None) and (testcovfile_path is not None):
        job_call = [python_path + ' ' + normative_path + ' -c ' +
                    covfile_path + ' -t ' + testcovfile_path + ' -f ' + func]
    elif cv_folds is not None:
        job_call = [python_path + ' ' + normative_path + ' -c ' +
                    covfile_path + ' -k ' + str(cv_folds) +  ' -f ' + func]
    elif func != 'estimate':
        job_call = [python_path + ' ' + normative_path + ' -c ' +
                    covfile_path +  ' -f ' + func]
    else:
        raise(ValueError, """For 'estimate' function either testcov or cvfold
              must be specified.""")
        
    # add algorithm-specific parameters
    if alg is not None:
        job_call = [job_call[0] + ' -a ' + alg]
        if configparam is not None:
            job_call = [job_call[0] + ' -x ' + str(configparam)]
    
    # add standardization flag if it is false
    # if not standardize:
    #     job_call = [job_call[0] + ' -s']
    
    # add responses file
    job_call = [job_call[0] + ' ' + respfile_path]
    
    # add in optional arguments. 
    for k in kwargs:
        job_call = [job_call[0] + ' ' + k + '=' + kwargs[k]]

    # writes bash file into processing dir
    with open(processing_dir+job_name, 'w') as bash_file:
        bash_file.writelines(bash_environment + output_changedir + \
                             job_call + ["\n"] + [sbatch_exit])

    # changes permissoins for bash.sh file
    os.chmod(processing_dir + job_name, 0o700)

def sbatch_nm(job_path,
              log_path):
    """
    This function submits a job.sh scipt to the torque custer using the qsub
    command.

    ** Input:
        * job_path      -> Full path to the job.sh file
        * log_path      -> The logs are currently stored in the working dir

    ** Output:
        * Submission of the job to the (torque) cluster

    witten by (primarily) T Wolfers
    """

    # created qsub command
    sbatch_call = ['sbatch ' + job_path]

    # submits job to cluster
    call(sbatch_call, shell=True)
    
    def rerun_nm(processing_dir,
                 memory,
                 duration,
                 new_memory=False,
                 new_duration=False,
                 binary=False,
                 **kwargs):
        """
        This function reruns all failed batched in processing_dir after 
        collect_nm has identified he failed batches
    
        * Input:
            * processing_dir        -> Full path to the processing directory
            * memory                -> Memory requirements written as string
                                       for example 4gb or 500mb
            * duration              -> The approximate duration of the job, a
                                       string with HH:MM:SS for example 01:01:01
            * new_memory            -> If you want to change the memory 
                                        you have to indicate it here.
            * new_duration          -> If you want to change the duration 
                                        you have to indicate it here.
        * Outputs:
            * Reruns failed batches. 
    
        written by (primarily) T Wolfers
        """
        log_path = kwargs.pop('log_path', None)
    
        if binary:
            file_extentions = '.pkl'
            failed_batches = fileio.load(processing_dir +
                                         'failed_batches' + 
                                         file_extentions)
            shape = failed_batches.shape
            for n in range(0, shape[0]):
                jobpath = failed_batches[n, 0]
                print(jobpath)
                if new_duration != False:
                    with fileinput.FileInput(jobpath, inplace=True) as file:
                        for line in file:
                            print(line.replace(duration, new_duration), end='')
                if new_memory != False:
                    with fileinput.FileInput(jobpath, inplace=True) as file:
                        for line in file:
                            print(line.replace(memory, new_memory), end='')
                sbatch_nm(jobpath,
                          log_path)
        else:
            file_extentions = '.txt'
            failed_batches = fileio.load_pd(processing_dir +
                                           'failed_batches' + file_extentions)
            shape = failed_batches.shape
            for n in range(0, shape[0]):
                jobpath = failed_batches.iloc[n, 0]
                print(jobpath)
                if new_duration != False:
                    with fileinput.FileInput(jobpath, inplace=True) as file:
                        for line in file:
                            print(line.replace(duration, new_duration), end='')
                if new_memory != False:
                    with fileinput.FileInput(jobpath, inplace=True) as file:
                        for line in file:
                            print(line.replace(memory, new_memory), end='')
                sbatch_nm(jobpath,
                          log_path)
