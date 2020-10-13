# Predictive Clinial Neuroscience Toolkit
Predictive Clinical Neuroscience software toolkit (formerly nispat). Methods for normative modelling, spatial statistics and pattern recognition 

## Installation (on a local machine)

i) install anaconda3 ii) create enviornment with "conda create --name env_name" iii) activate environment by "source activate env_name" iv) install the required packages below with "conda install "

```
conda install pip spyder pandas scipy
```

v) install PCNtoolkit

```
pip install pcntoolkit
```

Alternatively, download/clone from https://github.com/amarquand/PCNtoolkit/, unzip into download folder and execute "pip install PCNtoolkit-master/ "


## Alternative method to install on a shared resource
Make sure conda is available on the system.
Otherwise install it first from https://www.anaconda.com/ 

```
conda --version
```

Create a conda environment in a shared location

```
conda create -y python==3.7.7 numpy mkl blas --prefix=/shared/conda/normative_modeling/1.2.2
```

Activate the conda environment 

```
conda activate /shared/conda/normative_modeling/1.2.2
```

Install other dependencies

```
conda install -y spyder pandas scipy 
```

Install pip dependencies

```
pip --no-cache-dir install nibabel sklearn torch glob3 
```

Clone the repo

```
git clone https://github.com/amarquand/PCNtoolkit.git
```

install in the conda environment

```
cd PCNtoolkit/
python3 setup.py install
```

Test 
```
python -c "import pcntoolkit as pk;print(pk.__file__)"
```

# Quickstart usage
