import os
import pytest
from tempfile import gettempdir

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
"""This file contains pytest fixtures for file paths in the PCNtoolkit.

The fixtures defined here include:
1. Log directory
2. Save directory
3. Response file
4. Mask file
5. Covariate file
6. Test covariate file
7. Test response file
"""


@pytest.fixture
def log_dir():
    return os.path.join(gettempdir(), "pcntoolkit_tests", "log_test")


@pytest.fixture
def save_dir():
    return os.path.join(gettempdir(), "pcntoolkit_tests", "save_load_test")


@pytest.fixture
def responsefile():
    return os.path.join(
        PROJECT_ROOT, "pytest_tests", "resources", "data", "responses.csv"
    )


@pytest.fixture
def maskfile():
    return None


@pytest.fixture
def covfile():
    return os.path.join(
        PROJECT_ROOT, "pytest_tests", "resources", "data", "covariates.csv"
    )


@pytest.fixture
def testcov():
    return os.path.join(
        PROJECT_ROOT, "pytest_tests", "resources", "data", "covariates_test.csv"
    )


@pytest.fixture
def testresp():
    return os.path.join(
        PROJECT_ROOT, "pytest_tests", "resources", "data", "responses_test.csv"
    )


@pytest.fixture
def trbefile():
    return os.path.join(
        PROJECT_ROOT, "pytest_tests", "resources", "data", "batch_effects.csv"
    )


@pytest.fixture
def tsbefile():
    return os.path.join(
        PROJECT_ROOT, "pytest_tests", "resources", "data", "batch_effects_test.csv"
    )


@pytest.fixture
def resource_dir():
    return os.path.join(PROJECT_ROOT, "pytest_tests", "resources")
