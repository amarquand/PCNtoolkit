from setuptools import setup, find_packages


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as f:
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith("#")]

requirements = parse_requirements('requirements.txt')

setup(name='pcntoolkit',
      version='0.29',
      description='Predictive Clinical Neuroscience toolkit',
      url='http://github.com/amarquand/PCNtoolkit',
      author='Andre Marquand',
      author_email='andre.marquand@donders.ru.nl',
      license='GNU GPLv3',
      packages=find_packages(),
      install_requires=requirements,
      zip_safe=False)
