# Predictive Clinical Neuroscience Toolkit
Predictive Clinical Neuroscience software toolkit (formerly nispat). 

A Python package for normative modelling, spatial statistics and pattern recognition.

# IMPORTANT 
## Deprecation warning

This is PCNtoolkit version 1.X.X, released originally in June 2025. Any scripts, models, and results created with version 0.X.X are **not compatible** with this and future versions of the toolkit. 

To use the models created with versions 0.35 and earlier, please install the appropriate version using `pip install pcntoolkit==0.35`, or replace 0.35 with your desired version. The old version of the toolbox is also still available on [GitHub](https://github.com/amarquand/PCNtoolkit/tree/v0.35).

## Installation

```bash
pip install pcntoolkit
```

## Documentation

See the [documentation](https://pcntoolkit.readthedocs.io/en/latest/) for more details.

Documentation for the earlier version of the toolbox is available [here](https://pcntoolkit.readthedocs.io/en/v0.35/)

## Example usage

```python
from pcntoolkit import {load_fcon, BLR, NormativeModel}

fcon1000 = load_fcon()

train, test = fcon1000.train_test_split()

# Create a BLR model with heteroskedastic noise
model = NormativeModel(BLR(heteroskedastic=True), 
                       inscaler='standardize', 
                       outscaler='standardize')

model.fit_predict(train, test)
```

