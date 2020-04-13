## Install on your local machine


i) install Anaconda3 ii) create enviornment with " conda create --name normative_modeling " iii) activate environment by " source activate normative_modeling " iv) install the required packages below with " conda install " or " pip install "

```
conda install pip " " conda install spyder " " conda install pandas " " conda install scipy " " pip install nibabel " " pip install sklearn " " pip install torch " " pip install glob3 
```

v) download/clone nispat from https://github.com/amarquand/nispat/ vi) unzip nispat into download folder vii) change dir to download folder using the terminal and execute " pip install nispat-master/ "


## Install on a shared resource
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
conda activate /shared/onda/cnormative_modeling/1.2.2
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
git clone https://github.com/amarquand/nispat.git
```

install nispat in the conda environment

```
cd nispat/
python3 setup.py install
```

Test 
```
python -c "import nispat as pk;print(pk.__file__)"
```

