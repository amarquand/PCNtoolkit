#!/.../anaconda/bin/python/

# -----------------------------------------------------------------------------
# Run parallel normantive modelling.
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


def execute_nm(processing_dir,
               python_path,
               normative_path,
               job_name,
               covfile_path,
               respfile_path,
               batch_size,
               memory,
               duration,
               cv_folds=None,
               testcovfile_path=None,
               testrespfile_path=None,
               alg='gpr',
               configparam=None,
               cluster_spec='torque',
               binary=False,
               log_path=None):

    """
    This function is a motherfunction that executes all parallel normative
    modelling routines. Different specifications are possible using the sub-
    functions.

    ** Input:
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

    # import of necessary modules
    import glob

    split_nm(processing_dir,
             respfile_path,
             batch_size,
             binary,
             testrespfile_path)

    batch_dir = glob.glob(processing_dir + 'batch_*')
    # print(batch_dir)
    number_of_batches = len(batch_dir)
    # print(number_of_batches)

    if binary:
        file_extentions = '.pkl'
    else:
        file_extentions = '.txt'

    for n in range(1, number_of_batches+1):
        print(n)
        if testrespfile_path is not None:
            if cv_folds is not None:
                raise(ValueError, """If the response file is specified
                                     cv_folds must be equal to None""")
            else:
                # specified train/test split
                batch_processing_dir = processing_dir + 'batch_' + str(n) + '/'
                batch_job_name = job_name + str(n) + '.sh'
                batch_respfile_path = (batch_processing_dir + 'resp_batch_' +
                                       str(n) + file_extentions)
                batch_testrespfile_path = (batch_processing_dir +
                                           'testresp_batch_' +
                                           str(n) + file_extentions)
                batch_job_path = batch_processing_dir + batch_job_name
                if cluster_spec is 'torque':
                    bashwrap_nm(processing_dir=batch_processing_dir,
                                python_path=python_path,
                                normative_path=normative_path,
                                job_name=batch_job_name,
                                covfile_path=covfile_path,
                                cv_folds=cv_folds,
                                respfile_path=batch_respfile_path,
                                testcovfile_path=testcovfile_path,
                                testrespfile_path=batch_testrespfile_path,
                                alg=alg,
                                configparam=configparam)
                    qsub_nm(job_path=batch_job_path,
                            log_path=log_path,
                            memory=memory,
                            duration=duration)
                elif cluster_spec is 'new':
                    # this part requires addition in different envioronment [
                    bashwrap_nm(processing_dir=batch_processing_dir)
                    qsub_nm(processing_dir=batch_processing_dir)
                    # ]
        if testrespfile_path is None:
            if testcovfile_path is not None:
                # forward model
                batch_processing_dir = processing_dir + 'batch_' + str(n) + '/'
                batch_job_name = job_name + str(n) + '.sh'
                batch_respfile_path = (batch_processing_dir + 'resp_batch_' +
                                       str(n) + file_extentions)
                batch_job_path = batch_processing_dir + batch_job_name
                if cluster_spec is 'torque':
                    bashwrap_nm(processing_dir=batch_processing_dir,
                                python_path=python_path,
                                normative_path=normative_path,
                                job_name=batch_job_name,
                                covfile_path=covfile_path,
                                cv_folds=cv_folds,
                                respfile_path=batch_respfile_path,
                                testcovfile_path=testcovfile_path,
                                alg=alg,
                                configparam=configparam)
                    qsub_nm(job_path=batch_job_path,
                            log_path=log_path,
                            memory=memory,
                            duration=duration)
                elif cluster_spec is 'new':
                    # this part requires addition in different envioronment [
                    bashwrap_nm(processing_dir=batch_processing_dir)
                    qsub_nm(processing_dir=batch_processing_dir)
                    # ]
            else:
                # cross-validation
                batch_processing_dir = (processing_dir + 'batch_' +
                                        str(n) + '/')
                batch_job_name = job_name + str(n) + '.sh'
                batch_respfile_path = (batch_processing_dir +
                                       'resp_batch_' + str(n) +
                                       file_extentions)
                batch_job_path = batch_processing_dir + batch_job_name
                if cluster_spec is 'torque':
                    bashwrap_nm(processing_dir=batch_processing_dir,
                                python_path=python_path,
                                normative_path=normative_path,
                                job_name=batch_job_name,
                                covfile_path=covfile_path,
                                cv_folds=cv_folds,
                                respfile_path=batch_respfile_path,
                                testcovfile_path=testcovfile_path,
                                testrespfile_path=testrespfile_path,
                                alg=alg,
                                configparam=configparam)
                    qsub_nm(job_path=batch_job_path,
                            log_path=log_path,
                            memory=memory,
                            duration=duration)
                elif cluster_spec is 'new':
                    # this part requires addition in different envioronment [
                    bashwrap_nm(processing_dir=batch_processing_dir)
                    qsub_nm(processing_dir=batch_processing_dir)
                    # ]


"""routines that are environment independent"""


def split_nm(processing_dir,
             respfile_path,
             batch_size,
             binary,
             testrespfile_path=None):

    """ This function prepares the input files for normative_parallel.

    ** Input:
        * processing_dir    -> Full path to the folder of processing
        * respfile_path     -> Full path to the responsefile.txt
                               (subjects x features)
        * batch_size        -> Number of features in each batch
        * testrespfile_path -> Full path to the test responsefile.txt
                               (subjects x features)
        * binary            -> If True binary file

    ** Output:
        * The creation of a folder struture for batch-wise processing

    witten by (primarily) T Wolfers (adapted) SM Kia
    """

    # import of necessary modules
    import os
    import sys
    import numpy as np
    import pandas as pd

    try:
        import nispat.fileio as fileio
    except ImportError:
        pass
        path = os.path.abspath(os.path.dirname(__file__))
        if path not in sys.path:
            sys.path.append(path)
            del path
        import fileio

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

        numsub = len(respfile.ix[0, :])
        batch_vec = np.arange(0,
                              numsub,
                              batch_size)
        batch_vec = np.append(batch_vec,
                              numsub)
        batch_vec = batch_vec-1
        for n in range(0, (len(batch_vec) - 1)):
            resp_batch = respfile.ix[:,
                                     (batch_vec[n] + 1): batch_vec[n + 1]]
            os.chdir(processing_dir)
            resp = str('resp_batch_' + str(n+1))
            batch = str('batch_' + str(n+1))
            if not os.path.exists(processing_dir + batch):
                os.makedirs(processing_dir + batch)
                if (binary==False):
                    fileio.save_pd(resp_batch,
                                   processing_dir + batch + '/' +
                                   resp + '.txt')
                else:
                    resp_batch.to_pickle(processing_dir + batch + '/' +
                                         resp + '.pkl')

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
            respfile = pd.read_pickle(testrespfile_path)

        respfile = pd.DataFrame(respfile)
        testrespfile = pd.DataFrame(testrespfile)

        numsub = len(respfile.ix[0, :])
        batch_vec = np.arange(0, numsub,
                              batch_size)
        batch_vec = np.append(batch_vec,
                              numsub)
        batch_vec = batch_vec-1
        for n in range(0, (len(batch_vec) - 1)):
            resp_batch = respfile.ix[:, (batch_vec[n] + 1): batch_vec[n + 1]]
            testresp_batch = testrespfile.ix[:, (batch_vec[n]+1): batch_vec[n +
                                             1]]
            os.chdir(processing_dir)
            resp = str('resp_batch_' + str(n+1))
            testresp = str('testresp_batch_' + str(n+1))
            batch = str('batch_' + str(n+1))
            if not os.path.exists(processing_dir + batch):
                os.makedirs(processing_dir + batch)
                if (binary==False):
                    fileio.save_pd(resp_batch,
                                   processing_dir + batch + '/' +
                                   resp + '.txt')
                    fileio.save_pd(testresp_batch,
                                   processing_dir + batch + '/' + testresp +
                                   '.txt')
                else:
                    resp_batch.to_pickle(processing_dir + batch + '/' +
                                         resp + '.pkl')
                    testresp_batch.to_pickle(processing_dir + batch + '/' +
                                             testresp + '.pkl')


def collect_nm(processing_dir,
               collect=False,
               binary=False):
    """This function checks and collects all batches.

    ** Input:
        * processing_dir        -> Full path to the processing directory
        * collect               -> If True data is checked for failed batches
                                and collected; if False data is just checked

    ** Output:
        * Text files containing all results accross all batches the combined
          output

    written by (primarily) T Wolfers, (adapted) SM Kia
    """
    # import of necessary modules
    import os
    import sys
    import glob
    import numpy as np
    import pandas as pd
    try:
        import nispat.fileio as fileio
    except ImportError:
        pass
        path = os.path.abspath(os.path.dirname(__file__))
        if path not in sys.path:
            sys.path.append(path)
            del path
        import fileio

    if binary:
        file_extentions = '.pkl'
    else:
        file_extentions = '.txt'

    # detect number of subjects, batches, hyperparameters and CV
    file_example = glob.glob(processing_dir + 'batch_1/' + 'resp*' +
                             file_extentions)
    if binary is False:
        file_example = fileio.load(file_example[0])
    else:
        file_example = pd.read_pickle(file_example[0])
    numsubjects = file_example.shape[0]
    batch_size = file_example.shape[1]

    all_Hyptxt = glob.glob(processing_dir + 'batch_*/' + 'Hyp*')
    if all_Hyptxt != []:
        first_Hyptxt = fileio.load(all_Hyptxt[0])
        first_Hyptxt = first_Hyptxt.transpose()
        nHyp = len(first_Hyptxt)
        dir_first_Hyptxt = os.path.dirname(all_Hyptxt[0])
        all_crossval = glob.glob(dir_first_Hyptxt + '/'+'Hyp*')
        n_crossval = len(all_crossval)

    # artificially creates files for batches that were not executed
    count = 0
    batch_fail = []
    for batch in glob.glob(processing_dir + 'batch_*/'):
        filepath = glob.glob(batch + 'yhat*')
        per_measures = glob.glob(batch + 'pRho*')
        if filepath == []:
            count = count+1
            batch1 = glob.glob(batch + '/*.sh')
            print(batch1)
            batch_fail.append(batch1)
            if collect is True:
                if per_measures:
                    pRho = np.ones(batch_size)
                    pRho = pRho.transpose()
                    pRho = pd.Series(pRho)
                    # fileio.save_pd(pRho, batch + 'pRho.txt')
                    fileio.save(pRho, batch + 'pRho' + file_extentions)
                if per_measures:
                    Rho = np.zeros(batch_size)
                    Rho = Rho.transpose()
                    Rho = pd.Series(Rho)
                    # fileio.save_pd(Rho, batch + 'Rho.txt')
                    fileio.save(Rho, batch + 'Rho' + file_extentions)
                if per_measures:
                    rmse = np.zeros(batch_size)
                    rmse = rmse.transpose()
                    rmse = pd.Series(rmse)
                    # fileio.save_pd(rmse, batch + 'rmse.txt')
                    fileio.save(rmse, batch + 'rmse' + file_extentions)
                if per_measures:
                    smse = np.zeros(batch_size)
                    smse = smse.transpose()
                    smse = pd.Series(smse)
                    # fileio.save_pd(smse, batch + 'smse.txt')
                    fileio.save(smse, batch + 'smse' + file_extentions)
                if per_measures:
                    Z = np.zeros([batch_size,
                                  numsubjects])
                    Z = pd.DataFrame(Z)
                    # fileio.save_pd(Z, batch + 'Z.txt')
                    fileio.save(Z, batch + 'Z' + file_extentions)
                pRho = np.zeros([batch_size,
                                 numsubjects])
                pRho = pd.DataFrame(pRho)
                # fileio.save_pd(pRho, batch + 'pRho.txt')
                fileio.save(pRho, batch + 'pRho' + file_extentions)

                Rho = np.zeros([batch_size,
                                numsubjects])
                Rho = pd.DataFrame(Rho)
                # fileio.save_pd(Rho, batch + 'Rho.txt')
                fileio.save(Rho, batch + 'Rho' + file_extentions)

                rmse = np.zeros([batch_size,
                                numsubjects])
                rmse = pd.DataFrame(rmse)
                # fileio.save_pd(rmse, batch + 'rmse.txt')
                fileio.save(rmse, batch + 'rmse' + file_extentions)

                smse = np.zeros([batch_size,
                                numsubjects])
                smse = pd.DataFrame(smse)
                # fileio.save_pd(smse, batch + 'smse.txt')
                fileio.save(smse, batch + 'smse' + file_extentions)

                yhat = np.zeros([batch_size,
                                 numsubjects])
                yhat = pd.DataFrame(yhat)
                # fileio.save_pd(yhat, batch + 'yhat.txt')
                fileio.save(yhat, batch + 'yhat' + file_extentions)

                ys2 = np.zeros([batch_size,
                                numsubjects])
                ys2 = pd.DataFrame(ys2)
                # fileio.save_pd(ys2, batch + 'ys2.txt')
                fileio.save(ys2, batch + 'ys2' + file_extentions)

                Z = np.zeros([batch_size,
                              numsubjects])
                Z = pd.DataFrame(Z)
                # fileio.save_pd(Z, batch + 'Z.txt')
                fileio.save(Z, batch + 'Z' + file_extentions)

                for n in range(1, n_crossval+1):
                    hyp = np.zeros([batch_size,
                                    nHyp])
                    hyp = pd.DataFrame(hyp)
                    # fileio.save_pd(hyp, batch + 'Hyp_' + str(n) + '.txt')
                    fileio.save(hyp, batch + 'hyp' + file_extentions)

    # list batches that were not executed
    print('Number of batches that failed:' + str(count))
    batch_fail_df = pd.DataFrame(batch_fail)
    # fileio.save_pd(batch_fail_df, processing_dir + 'failed_batches.txt')
    fileio.save(batch_fail_df, processing_dir +
                'failed_batches' +
                file_extentions)

    # combines all output files across batches
    if collect is True:
        pRho_filenames = glob.glob(processing_dir + 'batch_*/' + 'pRho*')
        if pRho_filenames:
            pRho_filenames = fileio.sort_nicely(pRho_filenames)
            pRho_dfs = []
            for pRho_filename in pRho_filenames:
                pRho_dfs.append(pd.DataFrame(fileio.load(pRho_filename)))
            pRho_combined = pd.concat(pRho_dfs, ignore_index=True)
            # fileio.save_pd(pRho_combined, processing_dir + 'pRho.txt')
            fileio.save(pRho_combined, processing_dir + 'pRho' +
                        file_extentions)

        Rho_filenames = glob.glob(processing_dir + 'batch_*/' + 'Rho*')
        if pRho_filenames:
            Rho_filenames = fileio.sort_nicely(Rho_filenames)
            Rho_dfs = []
            for Rho_filename in Rho_filenames:
                Rho_dfs.append(pd.DataFrame(fileio.load(Rho_filename)))
            Rho_combined = pd.concat(Rho_dfs, ignore_index=True)
            # fileio.save_pd(Rho_combined, processing_dir + 'Rho.txt')
            fileio.save(Rho_combined, processing_dir + 'Rho' + file_extentions)

        Z_filenames = glob.glob(processing_dir + 'batch_*/' + 'Z*')
        if Z_filenames:
            Z_filenames = fileio.sort_nicely(Z_filenames)
            Z_dfs = []
            for Z_filename in Z_filenames:
                Z_dfs.append(pd.DataFrame(fileio.load(Z_filename)))
            Z_combined = pd.concat(Z_dfs, ignore_index=True)
            # fileio.save_pd(Z_combined, processing_dir + 'Z.txt')
            fileio.save(Z_combined, processing_dir + 'Z' + file_extentions)

        yhat_filenames = glob.glob(processing_dir + 'batch_*/' + 'yhat*')
        if yhat_filenames:
            yhat_filenames = fileio.sort_nicely(yhat_filenames)
            yhat_dfs = []
            for yhat_filename in yhat_filenames:
                yhat_dfs.append(pd.DataFrame(fileio.load(yhat_filename)))
            yhat_combined = pd.concat(yhat_dfs, ignore_index=True)
            # fileio.save_pd(yhat_combined, processing_dir + 'yhat.txt')
            fileio.save(yhat_combined, processing_dir +
                        'yhat' +
                        file_extentions)

        ys2_filenames = glob.glob(processing_dir + 'batch_*/' + 'ys2*')
        if ys2_filenames:
            ys2_filenames = fileio.sort_nicely(ys2_filenames)
            ys2_dfs = []
            for ys2_filename in ys2_filenames:
                ys2_dfs.append(pd.DataFrame(fileio.load(ys2_filename)))
            ys2_combined = pd.concat(ys2_dfs, ignore_index=True)
            # fileio.save_pd(ys2_combined, processing_dir + 'ys2.txt')
            fileio.save(ys2_combined, processing_dir + 'ys2' + file_extentions)

        rmse_filenames = glob.glob(processing_dir + 'batch_*/' + 'rmse*')
        if rmse_filenames:
            rmse_filenames = fileio.sort_nicely(rmse_filenames)
            rmse_dfs = []
            for rmse_filename in rmse_filenames:
                rmse_dfs.append(pd.DataFrame(fileio.load(rmse_filename)))
            rmse_combined = pd.concat(rmse_dfs, ignore_index=True)
            # fileio.save_pd(rmse_combined, processing_dir + 'rmse.txt')
            fileio.save(rmse_combined, processing_dir +
                        'rmse' +
                        file_extentions)

        smse_filenames = glob.glob(processing_dir + 'batch_*/' + 'smse*')
        if rmse_filenames:
            smse_filenames = fileio.sort_nicely(smse_filenames)
            smse_dfs = []
            for smse_filename in smse_filenames:
                smse_dfs.append(pd.DataFrame(fileio.load(smse_filename)))
            smse_combined = pd.concat(smse_dfs, ignore_index=True)
            # fileio.save_pd(smse_combined, processing_dir + 'smse.txt')
            fileio.save(smse_combined, processing_dir + 'smse' +
                        file_extentions)

        for n in range(1, n_crossval+1):
            Hyp_filenames = glob.glob(processing_dir + 'batch_*/' + 'Hyp_' +
                                      str(n) + '.*')
            if Hyp_filenames:
                Hyp_filenames = fileio.sort_nicely(Hyp_filenames)
                Hyp_dfs = []
                for Hyp_filename in Hyp_filenames:
                    Hyp_dfs.append(pd.DataFrame(fileio.load(Hyp_filename)))
                Hyp_combined = pd.concat(Hyp_dfs, ignore_index=True)
                # fileio.save_pd(Hyp_combined, processing_dir + 'Hyp_' +
                #                str(n) + '.txt')
                fileio.save(Hyp_combined, processing_dir + 'Hyp_' + str(n) +
                            file_extentions)


def delete_nm(processing_dir,
              binary=False):
    """This function deletes all processing for normative modelling and just
    keeps the combined output.

    * Input:
        * processing_dir        -> Full path to the processing directory

    written by (primarily) T Wolfers, (adapted) SM Kia
    """
    import shutil
    import glob
    import os
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
                cv_folds=None,
                testcovfile_path=None,
                testrespfile_path=None,
                alg=None,
                configparam=None):

    """ This function wraps normative modelling into a bash script to run it
    on a torque cluster system.

    ** Input:
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

    ** Output:
        * A bash.sh file containing the commands for normative modelling saved
        to the processing directory

    witten by (primarily) T Wolfers
    """

    # import of necessary modules
    import os

    # change to processing dir
    os.chdir(processing_dir)
    output_changedir = ['cd ' + processing_dir + '\n']

    bash_lines = '#!/bin/bash\n'
    bash_cores = 'export OMP_NUM_THREADS=1\n'
    bash_environment = [bash_lines + bash_cores]

    # creates call of function for normative modelling
    if testrespfile_path is not None:
        if testcovfile_path is not None:
            if cv_folds is not None:
                raise(ValueError, """If the testrespfile_path and
                                  testcovfile_path are not specified
                                  cv_folds must be equal to None""")
            else:
                job_call = [python_path + ' ' + normative_path + ' -c ' +
                            covfile_path + ' -t ' + testcovfile_path + ' -r ' +
                            testrespfile_path]

    if testrespfile_path is None:
        if testcovfile_path is None:
            if cv_folds is not None:
                job_call = [python_path + ' ' + normative_path + ' -c ' +
                            covfile_path + ' -k ' + str(cv_folds)]
            else:
                raise(ValueError, """If the testresponsefile_path and
                                  testcovfile_path are specified cv_folds
                                  must be larger than or equal to two(2)""")

    if testrespfile_path is None:
        if testcovfile_path is not None:
            if cv_folds is None:
                job_call = [python_path + ' ' + normative_path + ' -c ' +
                            covfile_path + ' -t ' + testcovfile_path]
            else:
                raise(ValueError, """If the test response file is and
                                  testcovfile is not specified cv_folds
                                  must be NONE""")

    # add algorithm-specific parameters
    if alg is not None:
        job_call = [job_call[0] + ' -a ' + alg]
        if configparam is not None:
            job_call = [job_call[0] + ' -x ' + str(configparam)]

    # add responses file
    job_call = [job_call[0] + ' ' + respfile_path]

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

    # import of necessary modules
    from subprocess import call

    # created qsub command
    if log_path is None:
        qsub_call = ['echo ' + job_path + ' | qsub -N ' + job_path + ' -l ' +
                     'mem=' + memory + ',walltime=' + duration]
    else:
        qsub_call = ['echo ' + job_path + ' | qsub -N ' + job_path +
                     ' -l ' + 'mem=' + memory + ',walltime=' + duration +
                     ' -o ' + log_path + ' -e ' + log_path]

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
    import nispat
    if binary:
        file_extentions = '.pkl'
    else:
        file_extentions = '.txt'
    failed_batches = nispat.fileio.load(processing_dir +
                                        'failed_batches' + file_extentions)
    shape = failed_batches.shape
    for n in range(0, shape[0]):
        jobpath = failed_batches[n, 0]
        print(jobpath)
        nispat.normative_parallel.qsub_nm(job_path=jobpath,
                                          log_path=log_path,
                                          memory=memory,
                                          duration=duration)

# COPY the rotines above here and aadapt those to your cluster
# bashwarp_nm; qsub_nm; rerun_nm
