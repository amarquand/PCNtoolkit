from pcntoolkit.normative_model import NormativeModel
import arviz as az
import pymc as pm


def compare_hbr_models(models: dict[str, str]):
    """Compares HBR models

    Args:
        models (dict[str, str]): dictionary of [model name, path]

    Returns:
        dictionary of (responsevar, comparison): [str, dataframe]
    """

    loaded_models: dict[str, NormativeModel] = {}
    for k, v in models.items():
        try:
            m = NormativeModel.load(v)
            m.predict(m.synthesize())
            loaded_models[k] = m
        except Exception as e:
            print("Cannot load model at location:", v, e)

    comparisons = {}
    for respvar in loaded_models[k].response_vars:
        traces = {}
        for name, model in loaded_models.items():
            with model[respvar].pymc_model:
                pm.compute_log_likelihood(model[respvar].idata)
            traces[name] = model[respvar].idata
        comparisons[respvar] = az.compare(traces)
    return comparisons
