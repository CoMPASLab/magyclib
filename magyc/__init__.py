# import methods from the corresponding modules
from .methods import magyc_bfg, magyc_ifg, magyc_ls, magyc_nls
from .benchmark_methods import ellipsoid_fit, ellipsoid_fit_fang
from .benchmark_methods import sphere_fit
from .benchmark_methods import twostep_hi, twostep_hsi
from .benchmark_methods import sar_aid, sar_kf, sar_ls
from .benchmark_methods import magfactor3
from .sim_data import create_synthetic_dataset
from .utils import hsi_calibration_validation, pds_geodesic_distance
from .plots import joe_hdg_error_std, joe_pos_error, joe_hdg_error_violin, ellipsoid_plot, magfield_data_plot

# define __all__ for the module
__all__ = ['magyc_bfg', 'magyc_ifg', 'magyc_ls', 'magyc_nls']
__all__ += ['ellipsoid_fit', 'ellipsoid_fit_fang']
__all__ += ['sphere_fit']
__all__ += ['twostep_hi', 'twostep_hsi']
__all__ += ['sar_aid', 'sar_kf', 'sar_ls']
__all__ += ['magfactor3']
__all__ += ['create_synthetic_dataset']
__all__ += ['hsi_calibration_validation', 'pds_geodesic_distance']
__all__ += ['joe_hdg_error_std', 'joe_pos_error', 'joe_hdg_error_violin', 'ellipsoid_plot', 'magfield_data_plot']
