# Predictive Clinical Neuroscience Toolkit
[![Gitter](https://badges.gitter.im/predictive-clinical-neuroscience/community.svg)](https://gitter.im/predictive-clinical-neuroscience/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Documentation Status](https://readthedocs.org/projects/pcntoolkit/badge/?version=latest)](https://pcntoolkit.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5207839.svg)](https://doi.org/10.5281/zenodo.5207839)


Predictive Clinical Neuroscience software toolkit (formerly nispat). 

Methods for normative modelling, spatial statistics and pattern recognition. Documentation, including tutorials can be found on [readthedocs](https://pcntoolkit.readthedocs.io/en/latest/). Click on the docs button above to visit the site. 

## Basic installation (on a local machine)

#### Install anaconda3 

using the download here: https://www.anaconda.com/download

#### Create environment 
```
conda create <env_name> python==3.12
```

#### Activate environment

```
source activate <env_name>
```

#### Install torch 

Use the command that you get from the command builder here: https://pytorch.org/get-started/locally/. This will ensure you do not install the CUDA version of torch if your pc does not have a GPU. We also recommend that you use the `conda` option. 

### Install nutpie using conda 

```
conda install nutpie numba -c conda-forge
```

#### Install PCNtoolkit

Using pip:
```
pip install pcntoolkit
```

Using a local clone of the repo:
```
python -m pip install .
```

## Alternative installation (on a shared resource)

#### Make sure conda is available on the system.
Otherwise install it first from https://www.anaconda.com/ 

```
conda --version
```

#### Create a conda environment in a shared location

```
conda create -y python==3.12 numpy mkl blas --prefix=/shared/conda/<env_name>
```

#### Activate the conda environment 

```
conda activate /shared/conda/<env_name>
```
#### install torch 

Using the command that you get from the command builder here:

```
https://pytorch.org/get-started/locally/
```

If your shared resource has no GPU, make sure you select the 'CPU' field in the 'Compute Platform' row. Here we also prefer conda over pip.


### Install nutpie using conda 

```
conda install nutpie numba -c conda-forge
```

#### Clone the repo

```
git clone https://github.com/amarquand/PCNtoolkit.git
```

### Install in the conda environment

```
cd PCNtoolkit/
python -m pip install .
```
### Test 
```
python -c "import pcntoolkit as pk;print(pk.__file__)"
```

## Quickstart usage

For normative modelling, functionality is handled by the normative.py script, which can be run from the command line, e.g.

```
# python normative.py -c /path/to/training/covariates -t /path/to/test/covariates -r /path/to/test/response/variables /path/to/my/training/response/variables
```

For more information, please see the following resources:

* [documentation](https://github.com/amarquand/PCNtoolkit/wiki/Home)
* [developer documentation](https://amarquand.github.io/PCNtoolkit/doc/build/html/)
* a tutorial and worked through example on a [real-world dataset](https://github.com/predictive-clinical-neuroscience/PCNtoolkit-demo)
