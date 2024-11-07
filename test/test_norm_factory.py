from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_factory import (
    create_normative_model,
    create_normative_model_from_args,
    load_normative_model,
)
from pcntoolkit.regression_model.blr.blr_conf import BLRConf
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf


def test_create_normative_model_hbr():
    norm_conf = NormConf()
    reg_conf = HBRConf()
    model = create_normative_model(norm_conf, reg_conf)
    assert isinstance(model, NormHBR)

def test_create_normative_model_blr():
    norm_conf = NormConf()
    reg_conf = BLRConf()
    model = create_normative_model(norm_conf, reg_conf)
    assert isinstance(model, NormBLR)

def test_load_normative_model(tmp_path):
    # Setup a temporary directory with a mock normative model
    model_path = tmp_path / "normative_model_dict.json"
    model_path.write_text('{"norm_conf": {"normative_model_name": "NormHBR"}}')
    
    model = load_normative_model(tmp_path)
    assert isinstance(model, NormHBR)

def test_create_normative_model_from_args():
    args = {"alg": "hbr"}
    model = create_normative_model_from_args(args)
    assert isinstance(model, NormHBR) 