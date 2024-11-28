#!/bin/bash
set -x

script_dir=$(realpath $(dirname $(realpath $0)))
test_dir=$(realpath $script_dir/..)
root_dir=$(realpath $test_dir/..)

export DOCKER_DIR=$root_dir/docker


export LOCAL_DATA_PATH="$script_dir/temp/pcntoolkit/data"
export DOCKER_DATA_PATH="/mnt/data"
mkdir -p $LOCAL_DATA_PATH
curl -o $LOCAL_DATA_PATH/fcon1000 https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv

cd $DOCKER_DIR
cd ..

echo "Splitting the data into train and test covariates, responses and batch effects..."
python tests/cli_test/split_data.py --input_file $LOCAL_DATA_PATH/fcon1000 --output_dir $LOCAL_DATA_PATH

cd $DOCKER_DIR

docker run -v $LOCAL_DATA_PATH:$DOCKER_DATA_PATH pcntoolkit:v0.31.0_dev normative $DOCKER_DATA_PATH/Y_tr_fcon1000.pkl -c $DOCKER_DATA_PATH/X_tr_fcon1000.pkl -f fit -a hbr warp=WarpSinArcsinh optimizer=l-bfgs-b warp_reparam=True
