#!/.../anaconda/bin/python/

# -----------------------------------------------------------------------------
# Run parallel normantive modelling.
#    All processing takes place in the processing directory (processing_dir)
#    All inputs should be text files and space seperated
#
#    It is possible to run these functions using...
#
#    * k-fold cross-validation
#    * estimating a training dataset then applying to a second test dataset
#
#    First,the data is split for parallel processing.
#    Second, the splits are submitted to the cluster.
#    Third, the output is collected and combined.
#
#    ** Main functions of normative parallel:
#        * execute_nm     -> executes split, bashwrap and qsub
#        *    split_nm     -> splits matrix (subjects x features) into
#                             vectors (subjects x batches of features)
#        *    bashwrap_nm  -> wraps python functions into a bash script
#        *    qsub_nm      -> executes bashwraped python script
#        * collect_nm      -> checkes, collects and combines output
#        * rerun_batches_nm -> reruns failed batches
#        * delete_nm       -> deletes unnessecary folders and files
#
# witten by Thomas Wolfers
#
# gratitude to Andre Marquand, Maarten Mennes, Matthias Ekman and Hong Lee
# for useful discussions and help around cluster computing.
#
# -----------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division

""" Run parallel normantive modelling.
    All processing takes place in the processing directory (processing_dir)
    All inputs should be text files and space seperated

    It is possible to run these functions using...

    * k-fold cross-validation
    * estimating a training dataset then applying to a second test dataset

    First,the data is split for parallel processing.
    Second, the splits are submitted to the cluster.
    Third, the output is collected and combined.

    ** Main functions of normative parallel:
        * execute_nm     -> executes split, bashwrap and qsub
        *    split_nm     -> splits matrix (subjects x features) into
                             vectors (subjects x batches of features)
        *    bashwrap_nm  -> wraps python functions into a bash script
        *    qsub_nm      -> executes bashwraped python script
        * collect_nm      -> checkes, collects and combines output
        * delete_nm       -> deletes unnessecary folders and files

* witten by Thomas Wolfers

"""


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
               bash_environment=None):

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
        * bash_environment   -> A txt file containing the necessary commands
                                for your bash environment to work

    written by Thomas Wolfers
    """

    # import of necessary modules
    import glob

    split_nm(processing_dir, respfile_path, batch_size, testrespfile_path)

    batch_dir = glob.glob(processing_dir + 'batch_*')
    print(batch_dir)
    number_of_batches = len(batch_dir)
    print(number_of_batches)

    for n in range(1, number_of_batches+1):
        print(n)

        if testrespfile_path is not None:
            if cv_folds is not None:
                raise(ValueError, """If the response file is specified
                                     cv_folds must be equal to None""")
            else:
                batch_processing_dir = processing_dir + 'batch_' + str(n) + '/'
                batch_job_name = job_name + str(n) + '.sh'
                batch_respfile_path = (batch_processing_dir + 'resp_batch_' +
                                       str(n) + '.txt')
                batch_testrespfile_path = (batch_processing_dir +
                                           'testresp_batch_' + str(n) + '.txt')
                batch_job_path = batch_processing_dir + batch_job_name

                bashwrap_nm(processing_dir=batch_processing_dir,
                            python_path=python_path,
                            normative_path=normative_path,
                            job_name=batch_job_name,
                            covfile_path=covfile_path,
                            cv_folds=cv_folds,
                            respfile_path=batch_respfile_path,
                            testcovfile_path=testcovfile_path,
                            testrespfile_path=batch_testrespfile_path,
                            bash_environment=bash_environment)
                qsub_nm(job_path=batch_job_path,
                        memory=memory,
                        duration=duration)

        if testrespfile_path is None:
            if testcovfile_path is not None:
                batch_processing_dir = processing_dir + 'batch_' + str(n) + '/'
                batch_job_name = job_name + str(n) + '.sh'
                batch_respfile_path = (batch_processing_dir + 'resp_batch_' +
                                       str(n) + '.txt')
                batch_job_path = batch_processing_dir + batch_job_name

                bashwrap_nm(processing_dir=batch_processing_dir,
                            python_path=python_path,
                            normative_path=normative_path,
                            job_name=batch_job_name,
                            covfile_path=covfile_path,
                            cv_folds=cv_folds,
                            respfile_path=batch_respfile_path,
                            testcovfile_path=testcovfile_path,
                            bash_environment=bash_environment)
                qsub_nm(job_path=batch_job_path,
                        memory=memory,
                        duration=duration)
            else:
                batch_processing_dir = (processing_dir + 'batch_' +
                                        str(n) + '/')
                batch_job_name = job_name + str(n) + '.sh'
                batch_respfile_path = (batch_processing_dir +
                                       'resp_batch_' + str(n) + '.txt')
                batch_job_path = batch_processing_dir + batch_job_name

                bashwrap_nm(processing_dir=batch_processing_dir,
                            python_path=python_path,
                            normative_path=normative_path,
                            job_name=batch_job_name,
                            covfile_path=covfile_path,
                            cv_folds=cv_folds,
                            respfile_path=batch_respfile_path,
                            testcovfile_path=testcovfile_path,
                            testrespfile_path=testrespfile_path,
                            bash_environment=bash_environment)
                qsub_nm(job_path=batch_job_path,
                        memory=memory,
                        duration=duration)


def split_nm(processing_dir, respfile_path, batch_size,
             testrespfile_path=None):

    """ This function prepares the input files for parallel normative modelling.

    ** Input:
        * processing_dir    -> Full path to the folder of processing
        * respfile_path     -> Full path to the responsefile.txt
                               (subjects x features)
        * batch_size        -> Number of features in each batch
        * testrespfile_path -> Full path to the test responsefile.txt
                               (subjects x features)

    ** Output:
        * The creation of a folder struture for batch-wise processing

    witten by Thomas Wolfers
    """

    # import of necessary modules
    import numpy as np
    import os
    import nispat.fileio as fileio
    import pandas as pd

    # splits response into batches
    if testrespfile_path is None:
        respfile = fileio.load_ascii(respfile_path)
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
                fileio.save_pd(resp_batch,
                               processing_dir + batch + '/' + resp + '.txt')

    # splits response and test responsefile into batches
    else:
        respfile = fileio.load_ascii(respfile_path)
        respfile = pd.DataFrame(respfile)
        testrespfile = fileio.load_ascii(testrespfile_path)
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
                fileio.save_pd(resp_batch,
                               processing_dir + batch + '/' + resp + '.txt')
                fileio.save_pd(testresp_batch,
                               processing_dir + batch + '/' + testresp +
                               '.txt')


def bashwrap_nm(processing_dir, python_path, normative_path, job_name,
                covfile_path, respfile_path,
                cv_folds=None, testcovfile_path=None,
                testrespfile_path=None, bash_environment=None):

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
        * bash_environment   -> A file containing the necessary commands
                                for your bash environment to work

    ** Output:
        * A bash.sh file containing the commands for normative modelling saved
        to the processing directory

    witten by Thomas Wolfers
    """

    # import of necessary modules
    import os

    # change to processing dir
    os.chdir(processing_dir)
    output_changedir = ['cd ' + processing_dir + '\n']

    # sets bash environment if necessary
    if bash_environment is not None:
        bash_environment = [bash_environment]
        print("""Your own environment requires in any case:
              #!/bin/bash\n export and optionally OMP_NUM_THREADS=1\n""")
    else:
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
                            testrespfile_path + ' ' + respfile_path]

    if testrespfile_path is None:
        if testcovfile_path is None:
            if cv_folds is not None:
                job_call = [python_path + ' ' + normative_path + ' -c ' +
                            covfile_path + ' -k ' + str(cv_folds) + ' ' +
                            respfile_path]
            else:
                raise(ValueError, """If the testresponsefile_path and
                                  testcovfile_path are specified cv_folds
                                  must be larger than or equal to two(2)""")

    if testrespfile_path is None:
        if testcovfile_path is not None:
            if cv_folds is None:
                job_call = [python_path + ' ' + normative_path + ' -c ' +
                            covfile_path + ' -t ' + testcovfile_path + ' ' +
                            respfile_path]
            else:
                raise(ValueError, """If the test response file is and
                                  testcovfile is not specified cv_folds
                                  must be NONE""")

    # writes bash file into processing dir
    with open(processing_dir+job_name, 'w') as bash_file:
        bash_file.writelines(bash_environment + output_changedir + job_call)

    # changes permissoins for bash.sh file
    os.chmod(processing_dir + job_name, 0o700)


def qsub_nm(job_path, memory, duration):
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

    witten by Thomas Wolfers
    """

    # import of necessary modules
    from subprocess import call

    # created qsub command
    qsub_call = ['echo ' + job_path + ' | qsub -N ' + job_path + ' -l ' +
                 'mem=' + memory + ',walltime=' + duration]

    # submits job to cluster
    call(qsub_call, shell=True)


def collect_nm(processing_dir, collect=False):
    """This function checks and collects all batches.

    ** Input:
        * processing_dir        -> Full path to the processing directory
        * collect               -> If True data is checked for failed batches
                                and collected; if False data is just checked

    ** Output:
        * Text files containing all results accross all batches the combined
          output

    written by Thomas Wolfers
    """
    # import of necessary modules
    import glob
    import numpy as np
    import os
    import pandas as pd
    import nispat.fileio as fileio

    # detect number of subjects, batches, hyperparameters and CV
    file_example = glob.glob(processing_dir + 'batch_1/' + 'resp*.txt')
    file_example = fileio.load_pd(file_example[0])
    numsubjects = file_example.shape[0]
    batch_size = file_example.shape[1]

    all_Hyptxt = glob.glob(processing_dir + 'batch_*/' + 'Hyp*')
    if all_Hyptxt != []:
        first_Hyptxt = fileio.load_pd(all_Hyptxt[0])
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
                    fileio.save_pd(pRho,
                                   batch + 'pRho.txt')

                if per_measures:
                    Rho = np.zeros(batch_size)
                    Rho = Rho.transpose()
                    Rho = pd.Series(Rho)
                    fileio.save_pd(Rho,
                                   batch + 'Rho.txt')

                if per_measures:
                    rmse = np.zeros(batch_size)
                    rmse = rmse.transpose()
                    rmse = pd.Series(rmse)
                    fileio.save_pd(rmse,
                                   batch + 'rmse.txt')

                if per_measures:
                    smse = np.zeros(batch_size)
                    smse = smse.transpose()
                    smse = pd.Series(smse)
                    fileio.save_pd(smse,
                                   batch + 'smse.txt')

                if per_measures:
                    Z = np.zeros([batch_size,
                                  numsubjects])
                    Z = pd.DataFrame(Z)
                    fileio.save_pd(Z,
                                   batch + 'Z.txt')

                pRho = np.zeros([batch_size,
                                 numsubjects])
                pRho = pd.DataFrame(pRho)
                fileio.save_pd(pRho,
                               batch + 'pRho.txt')

                Rho = np.zeros([batch_size,
                                numsubjects])
                Rho = pd.DataFrame(Rho)
                fileio.save_pd(Rho,
                               batch + 'Rho.txt')

                rmse = np.zeros([batch_size,
                                numsubjects])
                rmse = pd.DataFrame(rmse)
                fileio.save_pd(rmse,
                               batch + 'rmse.txt')

                smse = np.zeros([batch_size,
                                numsubjects])
                smse = pd.DataFrame(smse)
                fileio.save_pd(smse,
                               batch + 'smse.txt')

                yhat = np.zeros([batch_size,
                                 numsubjects])
                yhat = pd.DataFrame(yhat)
                fileio.save_pd(yhat,
                               batch + 'yhat.txt')

                ys2 = np.zeros([batch_size,
                                numsubjects])
                ys2 = pd.DataFrame(ys2)
                fileio.save_pd(ys2,
                               batch + 'ys2.txt')

                Z = np.zeros([batch_size,
                              numsubjects])
                Z = pd.DataFrame(Z)
                fileio.save_pd(Z,
                               batch + 'Z.txt')

                for n in range(1, n_crossval+1):
                    hyp = np.zeros([batch_size,
                                    nHyp])
                    hyp = pd.DataFrame(hyp)
                    fileio.save_pd(hyp,
                                   batch + 'Hyp_' + str(n) + '.txt')

    # list batches that were not executed
    print('Number of batches that failed:' + str(count))
    batch_fail_df = pd.DataFrame(batch_fail)
    fileio.save_pd(batch_fail_df, processing_dir + 'failed_batches.txt')

    # combines all output files across batches
    if collect is True:
        pRho_filenames = glob.glob(processing_dir + 'batch_*/' + 'pRho*')
        if pRho_filenames:
            pRho_filenames = fileio.sort_nicely(pRho_filenames)
            pRho_dfs = []
            for pRho_filename in pRho_filenames:
                pRho_dfs.append(fileio.load_pd(pRho_filename))
                pRho_combined = pd.concat(pRho_dfs,
                                          ignore_index=True)
            fileio.save_pd(pRho_combined, processing_dir + 'pRho.txt')

        Rho_filenames = glob.glob(processing_dir + 'batch_*/' + 'Rho*')
        if pRho_filenames:
            Rho_filenames = fileio.sort_nicely(Rho_filenames)
            Rho_dfs = []
            for Rho_filename in Rho_filenames:
                Rho_dfs.append(fileio.load_pd(Rho_filename))
                Rho_combined = pd.concat(Rho_dfs,
                                         ignore_index=True)
            fileio.save_pd(Rho_combined, processing_dir + 'Rho.txt')

        Z_filenames = glob.glob(processing_dir + 'batch_*/' + 'Z*')
        if Z_filenames:
            Z_filenames = fileio.sort_nicely(Z_filenames)
            Z_dfs = []
            for Z_filename in Z_filenames:
                Z_dfs.append(fileio.load_pd(Z_filename))
                Z_combined = pd.concat(Z_dfs,
                                       ignore_index=True)
            fileio.save_pd(Z_combined, processing_dir + 'Z.txt')

        yhat_filenames = glob.glob(processing_dir + 'batch_*/' + 'yhat*')
        if yhat_filenames:
            yhat_filenames = fileio.sort_nicely(yhat_filenames)
            yhat_dfs = []
            for yhat_filename in yhat_filenames:
                yhat_dfs.append(fileio.load_pd(yhat_filename))
                yhat_combined = pd.concat(yhat_dfs,
                                          ignore_index=True)
            fileio.save_pd(yhat_combined, processing_dir + 'yhat.txt')

        ys2_filenames = glob.glob(processing_dir + 'batch_*/' + 'ys2*')
        if ys2_filenames:
            ys2_filenames = fileio.sort_nicely(ys2_filenames)
            ys2_dfs = []
            for ys2_filename in ys2_filenames:
                ys2_dfs.append(fileio.load_pd(ys2_filename))
                ys2_combined = pd.concat(ys2_dfs,
                                         ignore_index=True)
            fileio.save_pd(ys2_combined, processing_dir + 'ys2.txt')

        rmse_filenames = glob.glob(processing_dir + 'batch_*/' + 'rmse*')
        if rmse_filenames:
            rmse_filenames = fileio.sort_nicely(rmse_filenames)
            rmse_dfs = []
            for rmse_filename in rmse_filenames:
                rmse_dfs.append(fileio.load_pd(rmse_filename))
                rmse_combined = pd.concat(rmse_dfs,
                                          ignore_index=True)
            fileio.save_pd(rmse_combined, processing_dir + 'rmse.txt')

        smse_filenames = glob.glob(processing_dir + 'batch_*/' + 'smse*')
        if rmse_filenames:
            smse_filenames = fileio.sort_nicely(smse_filenames)
            smse_dfs = []
            for smse_filename in smse_filenames:
                smse_dfs.append(fileio.load_pd(smse_filename))
                smse_combined = pd.concat(smse_dfs,
                                          ignore_index=True)
            fileio.save_pd(smse_combined, processing_dir + 'smse.txt')

        for n in range(1, n_crossval+1):
            Hyp_filenames = glob.glob(processing_dir + 'batch_*/' + 'Hyp_' +
                                      str(n) + '.*')
            if Hyp_filenames:
                Hyp_filenames = fileio.sort_nicely(Hyp_filenames)
                Hyp_dfs = []
                for Hyp_filename in Hyp_filenames:
                    Hyp_dfs.append(fileio.load_pd(Hyp_filename))
                    Hyp_combined = pd.concat(Hyp_dfs,
                                             ignore_index=True)
                fileio.save_pd(Hyp_combined, processing_dir + 'Hyp_' +
                               str(n) + '.txt')


def rerun_nm(processing_dir, memory, duration):
    """This function reruns all failed batched in processing_dir after collect_nm
    has identified he failed batches

    * Input:
        * processing_dir        -> Full path to the processing directory
        * memory        -> Memory requirements written as string for example
                           4gb or 500mb
        * duation       -> The approximate duration of the job, a string with
                           HH:MM:SS for example 01:01:01

    written by Thomas Wolfers
    """
    import nispat

    failed_batches = nispat.fileio.load_pd(processing_dir +
                                           'failed_batches.txt')
    shape = failed_batches.shape
    for n in range(0, shape[0]):
        jobpath = failed_batches.iloc[n, 0]
        print(jobpath)
        nispat.normative_parallel.qsub_nm(job_path=jobpath,
                                          memory=memory,
                                          duration=duration)


def delete_nm(processing_dir):
    """This function deletes all processing for normative modelling and just
    keeps the combined output.

    * Input:
        * processing_dir        -> Full path to the processing directory

    written by Thomas Wolfers
    """
    import shutil
    import glob
    import os

    for file in glob.glob(processing_dir + 'batch_*/'):
        shutil.rmtree(file)
    if os.path.exists(processing_dir + 'failed_batches.txt'):
        os.remove(processing_dir + 'failed_batches.txt')
