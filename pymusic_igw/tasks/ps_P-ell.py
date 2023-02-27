#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging

from pymusic.spec import NuFFT1D, BlackmanWindow, WedgeBCs
from pymusic.big_array import TimedArray
from pymusic.big_array import FFTPowerSpectrumArray

from pymusic_igw import AnalysisTask, autoHarm1DArray


# Set up logging
logger = logging.getLogger(__name__)


def nearest_to(arr, values):
	return np.array([arr[np.abs(arr - val).argmin()] for val in values])


class PowerSpecEll(AnalysisTask):

	def __init__(self):
		super().__init__("ps_P-ell")

	def compute(self):
		logger.info("Setting up power spectrum pipeline")
		field = "vel_1"
		dt = np.mean(np.diff(np.array(self.sim_data.labels_along_axis("time"))))
		fft = NuFFT1D(window=BlackmanWindow(), sampling_period=dt, spacing_tol=0.07)
		spec = TimedArray(FFTPowerSpectrumArray(
			autoHarm1DArray(
				self,
				self.sim_data.xs(field, axis="var"),
				max_ell=200,
				wedge_bc=WedgeBCs.ZERO_DERIVATIVE # Change to PERIODIC if necessary
			),
			fft,
			"time",
			"freq",
		))


		freqs = spec.labels_along_axis("freq")

		# Find the radii in the grid that are closest to the requested radii
		radii = np.array(spec.labels_along_axis("x1"))
		wanted_radii = np.arange(self.params.boundary_conv, 0.38 * self.params.radius, 0.02 * self.params.radius)
		logger.info(f"Selecting radii {wanted_radii}")
		radii = nearest_to(radii, wanted_radii)
		logger.info(f"Found radii {radii}")

		# Do the same for the requested frequencies
		freqs = np.array(spec.labels_along_axis("freq"))
		wanted_freqs = np.array([1e-6, 2e-6, 4e-6, 10e-6, 20e-6, 50e-6])
		logger.info(f"Selecting freqs {wanted_freqs}")
		freqs = nearest_to(freqs, wanted_freqs)
		logger.info(f"Found freqs {freqs}")

		# spec_selected is of shape (freqs, len(radii), len(ells))
		logger.info("Computing selected power spectra")
		spec_selected = spec.take(radii, "x1").take(freqs, "freq").array()
		ells = spec.labels_along_axis("ell")

		return (field, radii, ells, freqs, spec_selected)

	def plot(self, result):
		field, radii, ells, freqs, spec = result

		spec = spec.swapaxes(0, 1)
		# now spec.shape ~ (radii, freqs, ells)
		col = plt.get_cmap("coolwarm")
		norm = matplotlib.colors.LogNorm(vmin=freqs[0], vmax=freqs[-1])
		fig, axes = plt.subplots(len(radii), 1, figsize=(5, 4 * len(radii)))
		for ax, r, spec_r in zip(axes, radii, spec):
			for freq, spec_r_omega in zip(freqs, spec_r):
				ax.plot(ells, spec_r_omega, c=col(norm(freq)), label=r"$\omega = {:.1f}\mu$Hz".format(freq * 1e6))
			ax.set_title(r"$r = {:.3f}R_*$".format(r/self.params.radius))
			ax.set_ylim(top=1e11, bottom=1e-5)
			ax.set_xlabel(r"$\ell$")
			ax.set_ylabel(r"$P[v_r]$")
			# ax.set_xscale("log")
			ax.set_yscale("log")
			# ax.axvline(x=1 / self.params.tau_conv * 1e6, c="blue", lw=1)
			# ax.axvline(x=0.5 / 9e3 * 1e6, c="black", lw=1, ls='--')
			ax.grid(axis='y')
			ax.legend(ncols=2, fontsize='small')

		fig.tight_layout(rect=[0, 0.02, 1, 0.98])
		return fig

if __name__ == "__main__":
	PowerSpecEll().run()
