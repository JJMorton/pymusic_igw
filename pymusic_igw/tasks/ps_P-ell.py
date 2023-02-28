#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
from typing import Sequence

from pymusic.spec import NuFFT1D, BlackmanWindow, WedgeBCs
from pymusic.big_array import TimedArray, BigArray, FFTPowerSpectrumArray

from pymusic_igw import AnalysisTask, autoHarm1DArray


# Set up logging
logger = logging.getLogger(__name__)


def nearest_to(arr: BigArray, field: str, values: Sequence):
	'''
	Find the grid points in the scale of `field` that are closest to the requested values
	'''
	logger.info(f"Looking for values of {values} in {field}")
	axis = np.array(arr.labels_along_axis(field))
	axis_nearest = np.array([axis[np.abs(axis - val).argmin()] for val in values])
	logger.info(f"Found values of {axis_nearest} in {field}")
	return axis


class PowerSpecEll(AnalysisTask):
	'''
	A line plot of the power spectrum of the radial velocity at fixed radii and frequencies.
	Plotted against ell.
	Makes multiple plots, one for each selected radius, and each with the range of selected frequencies.
	'''

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
		wanted_radii = np.arange(self.params.boundary_conv, 0.38 * self.params.radius, 0.02 * self.params.radius)
		radii = nearest_to(spec, "x1", wanted_radii)

		# Do the same for the requested frequencies
		wanted_freqs = np.array([1e-6, 2e-6, 4e-6, 10e-6, 20e-6, 50e-6])
		freqs = nearest_to(spec, "freq", wanted_freqs)

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
				ax.plot(ells[1:], spec_r_omega[1:], c=col(norm(freq)), label=r"$\omega = {:.1f}\mu$Hz".format(freq * 1e6))
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
