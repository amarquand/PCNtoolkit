from setuptools import setup, find_packages

setup(name='nispat',
      version='0.12',
      description='Spatial methods for neuroimaging data',
      url='http://github.com/amarquand/nispat',
      author='Andre Marquand',
      author_email='a.f.marquand@fcdonders.ru.nl',
      license='GNU GPLv3',
      packages=find_packages(),
      install_requires=['nibabel', 'glob3', 'torch', 'pandas', 'scipy', 'nibabel', 'sklearn'],
      zip_safe=False)
