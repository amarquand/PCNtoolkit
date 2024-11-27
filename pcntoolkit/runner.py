import os
import warnings
from copy import deepcopy

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase

# First CV folds

class Runner:
    def __init__(
        self,
        normative_model: NormBase,
        cross_validate: bool = False,
        cv_folds: int = 5,
        parallelize: bool = False,
        cluster_type: str = "local",
        n_jobs: int = 1,
    ):
        self.normative_model = normative_model
        self.cross_validate = cross_validate
        self.cv_folds = cv_folds
        self.parallelize = parallelize
        self.cluster_type = cluster_type
        self.n_jobs = n_jobs

        if self.cross_validate and self.cv_folds <= 1:
            raise ValueError("If cross-validation is enabled, cv_folds must be greater than 1")
        if not self.normative_model.norm_conf.savemodel:
            warnings.warn("Model saving is disabled. Results will not be saved.")

    def fit(self, data: NormData) -> None:
        model_conf = self.normative_model.norm_conf

        if self.cross_validate:
            for i_fold, (train_data, test_data) in enumerate(data.kfold_split(self.cv_folds)):
                fold_model_conf = deepcopy(model_conf)

                fold_model_conf.set_log_dir(os.path.join(model_conf.log_dir, f"fold_{i_fold}"))
                fold_model_conf.set_save_dir(os.path.join(model_conf.save_dir, f"fold_{i_fold}"))

                fold_model = self.normative_model.__class__(fold_model_conf, self.default_reg_conf)
                self.normative_model.fit(train_data)

                # self.normative_model.predict(test_data)

        

    def predict(self, data: NormData) -> NormData:
        pass

    def fit_predict(self, data: NormData) -> NormData:
        pass
