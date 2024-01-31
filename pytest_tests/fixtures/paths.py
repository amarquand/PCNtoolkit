import pytest


@pytest.fixture(scope="session")
def responsefile():
    return "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/responses.csv"


@pytest.fixture(scope="session")
def maskfile():
    return None


@pytest.fixture(scope="session")
def covfile():
    return "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/covariates.csv"


@pytest.fixture(scope="session")
def testcov():
    return "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/covariates_test.csv"


@pytest.fixture(scope="session")
def testresp():
    return (
        "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/responses_test.csv"
    )


@pytest.fixture(scope="session")
def trbefile():
    return (
        "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/batch_effects.csv"
    )


@pytest.fixture(scope="session")
def tsbefile():
    return "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/batch_effects_test.csv"


@pytest.fixture(scope="session")
def resource_dir():
    return "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources"
