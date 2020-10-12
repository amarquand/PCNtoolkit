from setuptools import setup, find_packages

setup(name='pcntoolkit',
      version='0.16',
      description='Predictive Clinical Neuroscience toolkit',
      url='http://github.com/amarquand/nispat',
      author='Andre Marquand',
      author_email='a.marquand@donders.ru.nl',
      license='GNU GPLv3',
      packages=find_packages(),
      install_requires=[
          'argparse',
          'nibabel',
          'six',
          'sklearn', 
          'bspline',
          'matplotlib',
          'pandas>=0.25.3',
          'torch>=1.1.0', 
          'pymc3>=3.7', 
          'Theano==1.0.5'           
      ],
      zip_safe=False)
