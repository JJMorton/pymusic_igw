#!/usr/bin/env python3

from os import path
import matplotlib.pyplot as plt
import numpy as np
import logging

from pymusic.spec import SphericalHarmonicsTransform1D
from pymusic.spec import NuFFT1D
from pymusic.spec import BlackmanWindow
from pymusic.big_array import TimedArray
from pymusic.big_array import SphHarm1DArray
from pymusic.big_array import FFTPowerSpectrumArray

from pymusic_igw import AnalysisTask


# Set up logging
logger = logging.getLogger(__name__)


class PowerSpecAvgTheta(AnalysisTask):

    def __init__(self):
        super().__init__("power_spec_avg_theta")

    def compute(self):
        field = "vel_1"
        times = np.array(self.sim_data.labels_along_axis("time"))
        logger.info(f"Times from t={times[0]:E} to t={times[-1]:E}")
        dt = np.mean(np.diff(times))
        fft = NuFFT1D(window=BlackmanWindow(), sampling_period=dt, spacing_tol=0.07)

        logger.info("Setting up power spectrum pipeline")
        # spec.shape ~ (freq, r)
        spec = TimedArray(
            FFTPowerSpectrumArray(
                self.sim_data.xs(field, axis="var"),
                fft,
                "time",
                "freq",
            )
            .collapse(self.quad.average, "x2")
            .slabbed("x1", 256)
            .scaled(1.0 / (4 * np.pi))
        )

        freqs = np.array(spec.labels_along_axis("freq"))
        radii = np.array(spec.labels_along_axis("x1"))

        logger.info("Computing selected power spectra")
        return (field, radii, freqs, spec.array())

    def plot(self, result):
        field, radii, freqs, spec = result

        # Make the plots
        fig = plt.figure()
        fig.suptitle(f"Power Spectrum for Field '{field}'")
        ax = fig.add_subplot(1, 1, 1)
        ax.pcolormesh(radii, freqs*1e6, spec, cmap='inferno')
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$\omega$")

        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        return fig

if __name__ == "__main__":
    PowerSpecAvgTheta().run()
