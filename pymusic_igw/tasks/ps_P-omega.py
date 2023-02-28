#!/usr/bin/env python3

from os import path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import logging

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


class PowerSpec(AnalysisTask):
    '''
    A line plot of the power spectrum of the radial velocity at fixed radii and ells.
    Plotted against frequency.
    Makes multiple plots, one for each selected ell, and each with the range of selected radii.
    '''

    def __init__(self):
        super().__init__("ps_P-omega")

    def compute(self):
        logger.info("Setting up power spectrum pipeline")
        ells = [2, 3, 4, 7]
        field = "vel_1"
        dt = np.mean(np.diff(np.array(self.sim_data.labels_along_axis("time"))))
        fft = NuFFT1D(window=BlackmanWindow(), sampling_period=dt, spacing_tol=0.07)
        spec = TimedArray(FFTPowerSpectrumArray(
            autoHarm1DArray(
                self,
                self.sim_data.xs(field, axis="var"),
                max_ell=max(ells) + 1,
                wedge_bc=WedgeBCs.ZERO_DERIVATIVE
            ),
            fft,
            "time",
            "freq",
        ))


        freqs = spec.labels_along_axis("freq")

        # Find the radii in the grid that are closest to the requested radii
        wanted_radii = np.arange(self.params.boundary_conv, 0.38 * self.params.radius, 0.02 * self.params.radius)
        radii = nearest_to(spec, "x1", wanted_radii)

        # Do the same for ell
        ells = nearest_to(spec, "ell", ells)

        # spec_selected is of shape (omega, len(radii), len(ells))
        logger.info("Computing selected power spectra")
        spec_selected = spec.take(radii, "x1").take(ells, "ell").array()

        return (field, radii, ells, freqs, spec_selected)

    def plot(self, result):
        field, radii, ells, freqs, spec = result

        # GYRE predictions
        gyre_file = path.join(self.base_dir, "summary.txt")
        logger.info(f"Reading from {gyre_file}")
        gyre = np.genfromtxt(gyre_file, skip_header=5, names=True, unpack=True)
        gyre_ells = gyre[0]
        gyre_freqs = gyre[2]

        # Make the plots
        numcols = 2
        numrows = int(np.ceil(len(ells) / numcols))
        fig = plt.figure(figsize=(numcols * 4 * 1.5, numrows * 3 * 1.5))
        fig.suptitle(f"Power Spectrum for Field '{field}'")
        for i in range(len(ells)):
            ell = ells[i]
            spec_ell = spec[:, :, i] # The spectra with ell=ells[i]
            ax = fig.add_subplot(numrows, numcols, i + 1)
            # Plot dashed lines for the GYRE modes
            for gyre_l, gyre_freq in zip(gyre_ells, gyre_freqs):
                if gyre_l != ell: continue
                plt.axvline(gyre_freq, c='lightgrey', lw=1, ls='--')
            # Plot solid lines for the power spectra
            for j in range(len(radii)):
                r = radii[j]
                spec_r_ell = spec_ell[:, j]
                ax.plot(np.array(freqs) * 1e6, spec_r_ell, label=r"$r={:.3f}R_\ast$".format(r / self.params.radius))
            ax.set_title(r"$\ell = {}$".format(ell))
            ax.set_xlabel(r"$\omega$ ($\mu$Hz)")
            ax.set_ylabel(f"P[{field}]")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()

        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        return fig

if __name__ == "__main__":
    PowerSpec().run()
