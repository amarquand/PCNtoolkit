from setuptools import setup, find_packages

setup(name='pcntoolkit',
      version='0.28',
      description='Predictive Clinical Neuroscience toolkit',
      url='http://github.com/amarquand/PCNtoolkit',
      author='Andre Marquand',
      author_email='andre.marquand@donders.ru.nl',
      license='GNU GPLv3',
      packages=find_packages(),
      install_requires=[
          'argparse',
          'nibabel>=2.5.1',
          'six',
          'scikit-learn', 
          'bspline',
          'matplotlib',
          'numpy',
          'scipy>=1.3.2',
          'pandas>=0.25.3',
          'torch>=1.1.0', 
          'sphinx-tabs',
          'pymc>=5.1.0',
          'arviz==0.13.0'
      ],
      zip_safe=False)
