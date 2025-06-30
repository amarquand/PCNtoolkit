# Predictive Clinical Neuroscience Toolkit
Predictive Clinical Neuroscience software toolkit (formerly nispat). 

A Python package for normative modelling, spatial statistics and pattern recognition.

## Deprecation warning

This is PCNtoolkit version 1.X.X, released originally in June 2025. Any scripts, models, and results created with version 0.X.X are incompatible with this and future versions of the toolkit. 

To use the models created with versions 0.35 and earlier, please install the appropriate version using `pip install pcntoolkit==0.35`, or replace 0.35 with your desired version. 


## Installation

```bash
pip install pcntoolkit
```

## Documentation

See the [documentation](https://pcntoolkit.readthedocs.io/en/latest/) for more details.


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

