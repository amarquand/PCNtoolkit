from dataclasses import dataclass


@dataclass(frozen=True)
class ParamConf:
    # Specify the type of distribution
    linear: bool = False
    random: bool = False
    centered: bool = False
    random_slope: bool = False
    centered_slope: bool = False
    random_intercept: bool = False
    centered_intercept: bool = False

    # Simple, fixed configuration
    dist: str = "Normal"
    dist_params: tuple = (0, 1)

    # Random configuration
    mu_dist: str = "Normal"
    mu_dist_params: tuple = (0, 1)
    sigma_dist: str = "HalfNormal"
    sigma_dist_params: tuple = (1,)
    offset_dist: str = "Normal"
    offset_dist_params: tuple = (0, 1)

    # Simple linear configurations
    slope_dist: str = "Normal"
    slope_dist_params: tuple = (1,)

    intercept_dist: str = "Normal"
    intercept_dist_params: tuple = (0, 1)

    # Linear random configurations
    mu_slope_dist: str = "Normal"
    mu_slope_dist_params: tuple = (0, 1)
    sigma_slope_dist: str = "HalfNormal"
    sigma_slope_dist_params: tuple = (1,)
    offset_slope_dist: str = "Normal"
    offset_slope_dist_params: tuple = (0, 1)

    # intercept
    mu_intercept_dist: str = "Normal"
    mu_intercept_dist_params: tuple = (0, 1)
    sigma_intercept_dist: str = "HalfNormal"
    sigma_intercept_dist_params: tuple = (1,)
    offset_intercept_dist: str = "Normal"
    offset_intercept_dist_params: tuple = (0, 1)


def default_mu_conf():
    return ParamConf(name='mu',linear=True,
                    random_intercept=True,
                    random_slope=False)

def default_sigma_conf():
    return ParamConf(name='sigma',dist="HalfNormal", dist_params=(1,))