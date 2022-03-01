.. title:: install

Installation
==================

Basic installation (on a local machine)
-----------------------------------------------------

1. Install anaconda3 

2. Create enviornment 

.. code-block:: bash

	conda create --name <env_name>

3. Activate environment 

.. code-block:: bash

	source activate <env_name>

4. Install required conda packages
	
.. code-block:: bash

	conda install pip pandas scipy
	
5. Install PCNtoolkit (plus dependencies)
	
.. code-block:: bash

	pip install pcntoolkit
	
Alternative installation (on a shared resource)
-----------------------------------------------------

1. Make sure conda is available on the system. Otherwise install it first from https://www.anaconda.com/ 
	
.. code-block:: bash

	conda --version

	
2. Create a conda environment in a shared location
	
.. code-block:: bash

	conda create -y python==3.7.7 numpy mkl blas --prefix=/shared/conda/<env_name>

	
3. Activate the conda environment 
	
.. code-block:: bash

	conda activate /shared/conda/<env_name>


4. Install other dependencies
	
.. code-block:: bash

	conda install -y pandas scipy 

	
5. Install pip dependencies
	
.. code-block:: bash

	pip --no-cache-dir install nibabel sklearn torch glob3 

	
6. Clone the repo
	
.. code-block:: bash

	git clone https://github.com/amarquand/PCNtoolkit.git

	
7. Install in the conda environment
	
.. code-block:: bash

	cd PCNtoolkit/
	python3 setup.py install

	
8. Test 

.. code-block:: bash

	python -c "import pcntoolkit as pk;print(pk.__file__)"

	
Quickstart usage
-----------------------------------------------------

For normative modelling, functionality is handled by the ``normative.py`` script, which can be run from the command line, e.g.
	
.. code-block:: bash

	python normative.py -c /path/to/training/covariates -t /path/to/test/covariates -r /path/to/test/response/variables /path/to/my/training/response/variables
