import pytest

from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_factory import (
    create_normative_model,
    load_normative_model,
)
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.blr.blr import BLR
from pcntoolkit.regression_model.blr.blr_conf import BLRConf
from pcntoolkit.regression_model.gpr.gpr import GPR
from pcntoolkit.regression_model.gpr.gpr_conf import GPRConf
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.reg_conf import RegConf
from pytest_tests.fixtures.path_fixtures import *


@pytest.mark.parametrize(
    "norm_subclass, reg_conf, reg_model", [(NormHBR, HBRConf, HBR)]
)
def test_create_normative_model(
    norm_subclass: NormBase, generic_norm_conf: NormConf, reg_conf: RegConf, reg_model
):
    norm_model: NormBase = create_normative_model(generic_norm_conf, reg_conf())
    assert isinstance(norm_model, norm_subclass)
    assert norm_model.regression_model_type == reg_model
