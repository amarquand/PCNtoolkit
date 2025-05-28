# Predictive Clinical Neuroscience Toolkit
Predictive Clinical Neuroscience software toolkit (formerly nispat). 

A Python package for normative modelling, spatial statistics and pattern recognition.

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

## Citation 

If you use this software in your research, please cite:



