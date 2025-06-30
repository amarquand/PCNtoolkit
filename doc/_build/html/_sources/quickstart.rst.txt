Quickstart Guide
================

Installation Options
--------------------

Basic Installation (Local Machine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Install anaconda3
2. Create and activate environment:

   .. code-block:: bash

       conda create --name <env_name>
       source activate <env_name>

3. Install required conda packages:

   .. code-block:: bash

       conda install pip pandas scipy

4. Install PCNtoolkit:

   .. code-block:: bash

       pip install pcntoolkit

Alternative Installation (Shared Resource)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Ensure conda is available:

   .. code-block:: bash

       conda --version

2. Create conda environment in shared location:

   .. code-block:: bash

       conda create -y python==3.8.3 numpy mkl blas --prefix=/shared/conda/<env_name>

3. Activate the environment:

   .. code-block:: bash

       conda activate /shared/conda/<env_name>

4. Install dependencies:

   .. code-block:: bash

       conda install -y pandas scipy
       pip --no-cache-dir install nibabel scikit-learn torch glob3

5. Clone and install:

   .. code-block:: bash

       git clone https://github.com/amarquand/PCNtoolkit.git
       cd PCNtoolkit/
       python3 setup.py install

6. Test installation:

   .. code-block:: bash

       python -c "import pcntoolkit as pk;print(pk.__file__)"

Basic Usage
-----------

For normative modeling, use the normative.py script from command line:

.. code-block:: bash

    python normative.py -c /path/to/training/covariates -t /path/to/test/covariates -r /path/to/test/response/variables /path/to/my/training/response/variables

Additional Resources
--------------------

* `Documentation <https://github.com/amarquand/PCNtoolkit/wiki/Home>`_
* `Developer Documentation <https://amarquand.github.io/PCNtoolkit/doc/build/html/>`_
* `Tutorial and Examples <https://github.com/predictive-clinical-neuroscience/PCNtoolkit-demo>`_ 