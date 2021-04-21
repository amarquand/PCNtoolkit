from setuptools import setup, find_packages

setup(name='pcntoolkit',
      version='0.20',
      description='Predictive Clinical Neuroscience toolkit',
      url='http://github.com/amarquand/nispat',
      author='Andre Marquand',
      author_email='a.marquand@donders.ru.nl',
      license='GNU GPLv3',
      packages=find_packages(),
      install_requires=[
          'argparse',
          'nibabel>=2.5.1',
          'six',
          'sklearn', 
          'bspline',
          'matplotlib',
          'numpy>=1.19.5',
          'scipy>=1.3.2',
          'pandas>=0.25.3',
          'torch>=1.1.0', 
          'pymc3>=3.11.2',
      ],
      zip_safe=False)
