import os

from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_factory import create_normative_model
from pcntoolkit.regression_model.blr.blr_conf import BLRConf
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf



def main():
    log_dir = "/home/stijn/temp/pcntoolkit/log"
    save_dir = "/home/stijn/temp/pcntoolkit/save"
    normconf = NormConf(perform_cv = False, cv_folds=0, log_dir=log_dir, save_dir=save_dir)
    normconf.save_as_json("/home/stijn/temp/pcntoolkit/normconf.json")
    regconf = HBRConf(example_parameter=-1)

    normative_model = create_normative_model(normconf, regconf)
    normative_model.fit_predict(None)

if __name__ == "__main__":
    main()