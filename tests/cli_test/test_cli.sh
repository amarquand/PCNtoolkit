#! /bin/bash
set -x

# Assign the current directory to a variable
export testdir=$(pwd)
export tempdir="$testdir/temp"
mkdir $tempdir
chmod -R 766 $tempdir
export data_name="fcon1000"
export model_config="-a blr warp=WarpSinArcsinh optimizer=l-bfgs-b warp_reparam=True inscaler=standardize"
echo "Downloading the data..."
curl -o $tempdir/$data_name https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/$data_name.csv
echo "Splitting the data into train and test covariates, responses and batch effects..."
python split_data.py --input_file $tempdir/$data_name --output_dir $tempdir
echo "Fitting the model..."
normative $tempdir/Y_tr_$data_name.pkl -c $tempdir/X_tr_$data_name.pkl -f fit $model_config
echo "Predicting the test set..."
normative $tempdir/Y_te_$data_name.pkl -c $tempdir/X_te_$data_name.pkl -f predict $model_config inputsuffix=fit outputsuffix=predict
echo "Also doing estimate..."
normative $tempdir/Y_tr_$data_name.pkl -c $tempdir/X_tr_$data_name.pkl -f estimate $model_config -t $tempdir/X_te_$data_name.pkl -r $tempdir/Y_te_$data_name.pkl outputsuffix=estimate
echo "Done!"
rm -R $tempdir