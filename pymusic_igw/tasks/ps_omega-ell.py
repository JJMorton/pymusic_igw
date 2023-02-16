#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging

from pymusic.spec.wedge_harmonics import WedgeBCs
from pymusic.spec import NuFFT1D
from pymusic.spec import BlackmanWindow
from pymusic.big_array import TimedArray
from pymusic.big_array import FFTPowerSpectrumArray

from pymusic_igw import AnalysisTask, autoHarm1DArray


# Set up logging
logger = logging.getLogger(__name__)


class PowerSpecOmegaEll(AnalysisTask):

    def __init__(self):
        super().__init__("ps_omega-ell")

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
                wedge_bc=WedgeBCs.ZERO_DERIVATIVE
            ),
            fft,
            "time",
            "freq",
        ))

        ells = np.array(spec.labels_along_axis("ell"))
        ells = ells[ells <= 200]
        freqs = np.array(spec.labels_along_axis("freq"))

        # Take equally spaced radii through radiative zone
        all_radii = np.array(self.sim_data.labels_along_axis("x1"))
        radii = np.arange(0.1, 1.0, 0.1) * (all_radii[-1] - self.params.boundary_conv) + self.params.boundary_conv
        logger.info(f"Selecting radii {radii}")
        radii = np.array([all_radii[np.abs(all_radii - r).argmin()] for r in radii])

        # spec_selected is of shape (omega, len(radii), len(ells))
        logger.info("Computing selected power spectra")
        spec_selected = spec.take(radii, "x1").take(ells, "ell").array()

        return (field, radii, ells, freqs, spec_selected)

    def plot(self, result):
        field, radii, ells, freqs, spec = result

        logger.info(f"radii: {radii.shape}")
        logger.info(f"ells: {ells.shape}")
        logger.info(f"freqs: {freqs.shape}")

        # Make the plots
        numcols = 1
        numrows = int(np.ceil(len(radii) / numcols))
        fig = plt.figure(figsize=(numcols * 4 * 1.5, numrows * 3 * 1.5))
        fig.suptitle(f"Power Spectrum for Field '{field}'")
        for i, radius in enumerate(radii):
            spec_r = spec[:, i, :]
            ax = fig.add_subplot(numrows, numcols, i + 1)
            mesh = ax.pcolormesh(ells, freqs*1e6, spec_r, cmap='inferno', norm=matplotlib.colors.LogNorm())
            fig.colorbar(mesh)
            ax.set_xlabel(r"$\ell$")
            ax.set_ylabel(r"$\omega$")
            ax.set_title(r"$r = {:.2f}$".format(radius / self.params.radius))

        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        return fig

if __name__ == "__main__":
    PowerSpecOmegaEll().run()
