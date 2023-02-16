import numpy as np
import logging

from pymusic.big_array import BigArray
from pymusic.spec import WedgeHarmonicsTransform1D, WedgeHarm1DArray, WedgeBCs, SphericalHarmonicsTransform1D
from pymusic.big_array import SphHarm1DArray
from pymusic_igw.analysis_task import AnalysisTask


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def autoHarm1DArray(task: AnalysisTask, arr: BigArray, max_ell: float, wedge_bc: WedgeBCs):
	'''
	Creates either a `SphHarm1DArray` or a `WedgeHarm1DArray`, as appropriate.
	Transparent implementation, returns identically structured `BigArray` either way.
	'''
	if task.is_wedge:
		logger.info("Transforming to wedge harmonics...")
		wh_xform = WedgeHarmonicsTransform1D(task.sim.grid.grids[1], bc=wedge_bc)
		trans = WedgeHarm1DArray(
			arr,
			wh_xform,
			theta_axis="x2",
			ell_eff_axis="ell",
		)
		ells = np.array(trans.labels_along_axis("ell"))
		ells = ells[ells <= max_ell]
		return trans.take(ells, "ell")
	else:
		logger.info("Transforming to spherical harmonics...")
		sh_xform = SphericalHarmonicsTransform1D(task.quad, int(max_ell))
		return SphHarm1DArray(
			arr,
			sh_xform,
			theta_axis="x2",
			ell_axis="ell",
			ells=range(int(max_ell))
		)
