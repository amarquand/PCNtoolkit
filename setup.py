from setuptools import setup, find_packages

setup(name='nispat',
      version='0.13',
      description='Spatial methods for neuroimaging data',
      url='http://github.com/amarquand/nispat',
      author='Andre Marquand',
      author_email='a.f.marquand@fcdonders.ru.nl',
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
          'Theano==1.0.4'           
      ],
      zip_safe=False)
