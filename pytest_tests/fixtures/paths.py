import pytest


@pytest.fixture
def responsefile():
    return "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/responses.csv"


@pytest.fixture
def maskfile():
    return None


@pytest.fixture
def covfile():
    return "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/covariates.csv"


@pytest.fixture
def testcov():
    return "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/covariates_test.csv"


@pytest.fixture
def testresp():
    return (
        "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/responses_test.csv"
    )


@pytest.fixture
def trbefile():
    return (
        "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/batch_effects.csv"
    )


@pytest.fixture
def tsbefile():
    return "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/batch_effects_test.csv"


@pytest.fixture
def resource_dir():
    return "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources"
