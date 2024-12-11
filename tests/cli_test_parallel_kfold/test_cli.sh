#! /bin/bash
set -x

# Assign the current directory to a variable
export testdir=$(pwd)
export tempdir="$testdir/temp"
mkdir $tempdir
chmod -R 766 $tempdir
export data_name="fcon1000"
echo "Downloading the data..."
curl -o $tempdir/$data_name https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/$data_name.csv
echo "Splitting the data into train and test covariates, responses and batch effects..."
python split_data.py --input_file $tempdir/$data_name --output_dir $tempdir

echo "Fitting the model..."
python submit_jobs.py func=fit covfile_path=$tempdir/X_tr_$data_name.pkl respfile_path=$tempdir/Y_tr_$data_name.pkl

echo "Predicting the test set..."
python submit_jobs.py func=predict covfile_path=$tempdir/X_te_$data_name.pkl respfile_path=$tempdir/Y_te_$data_name.pkl

echo "Also doing estimate..."
python submit_jobs.py func=estimate covfile_path=$tempdir/X_tr_$data_name.pkl respfile_path=$tempdir/Y_tr_$data_name.pkl testcovfile_path=$tempdir/X_te_$data_name.pkl testrespfile_path=$tempdir/Y_te_$data_name.pkl

echo "Done!"
rm -R $tempdir