from setuptools import setup, find_packages

setup(name='nispat',
      version='0.12',
      description='Spatial methods for neuroimaging data',
      url='http://github.com/amarquand/nispat',
      author='Andre Marquand',
      author_email='a.f.marquand@fcdonders.ru.nl',
      license='GNU GPLv3',
      packages=find_packages(),
      install_requires=[
          'sklearn', 'torch>=1.1.0'
      ],
      zip_safe=False)
