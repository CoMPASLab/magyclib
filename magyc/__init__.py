# import methods from the corresponding modules
from .methods import magyc_bfg, magyc_ifg, magyc_ls, magyc_nls
from .benchmark_methods import ellipsoid_fit, ellipsoid_fit_fang
from .benchmark_methods import sphere_fit
from .benchmark_methods import twostep_hi, twostep_hsi
from .benchmark_methods import sar_aid, sar_kf, sar_ls
from .benchmark_methods import magfactor3

# define __all__ for the module
__all__ = ['magyc_bfg', 'magyc_ifg', 'magyc_ls', 'magyc_nls']
__all__ += ['ellipsoid_fit', 'ellipsoid_fit_fang']
__all__ += ['sphere_fit']
__all__ += ['twostep_hi', 'twostep_hsi']
__all__ += ['sar_aid', 'sar_kf', 'sar_ls']
__all__ += ['magfactor3']
