#!/usr/bin/env python3

import numpy as np
import logging

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pymusic.particles as pmp
from pymusic_igw import AnalysisTask


# Set up logging
logger = logging.getLogger(__name__)


class ParticleMeanVariance(AnalysisTask):

    def __init__(self):
        super().__init__(f"particle_mean_variance")

    def compute(self):
        # Get all tracer files in dump directory
        dumps_dir = Path(self.base_dir, "particle_dumps/")
        if not dumps_dir.exists():
            logger.error("Particle dumps not found at " + str(dumps_dir))
            return
        files = sorted(dumps_dir.glob("*.tracers.h5"))
        dumps = [pmp.ParticleDumpFromFile(f) for f in files]
        num_dumps = len(dumps)
        num_particles = len(dumps[0].dataframe())
        logger.info(f"{num_dumps} particle dumps from t={dumps[0].time:E} to t={dumps[-1].time:E}")

        # Reduce memory usage by increasing number of chunks to use
        num_chunks = 4
        chunk_size = int(np.ceil(num_particles / num_chunks))
        logger.info(f"Calculating means in {num_chunks} chunks, {chunk_size} particles in each")

        # These arrays are 1D of length num_particles
        init_radii_tot = np.array([])
        mean_radii_tot = np.array([])
        var_radii_tot = np.array([])

        # Calculate each chunk
        for chunk in range(num_chunks):
            # Calculate which particles to use
            p_start = chunk * chunk_size
            p_end = (chunk + 1) * chunk_size
            if p_end >= num_particles:
                p_end = num_particles - 1
            logger.info(f"Chunk {chunk + 1}, {p_end - p_start} particles")
            radii = np.zeros((num_dumps, p_end - p_start))

            # Read the dumps
            logger.info("Reading particle dumps...")
            for i, dump in enumerate(dumps):
                df = dump.dataframe()
                # logger.info(f"Analysing particle dump {i} ({len(df)} particles)")
                radii[i] = np.array(df.sort_index()["x1"][p_start:p_end])

            # Calculate the means and variances
            init_radii = radii[0]
            logger.info("Calculating means...")
            mean_radii = np.mean(radii, axis=0)
            logger.info("Calculating variances...")
            var_radii = np.mean(np.power(radii - mean_radii[np.newaxis, :], 2), axis=0) # sum[(r - mean)**2] / N

            # Append to complete arrays
            init_radii_tot = np.concatenate((init_radii_tot, init_radii))
            mean_radii_tot = np.concatenate((mean_radii_tot, mean_radii))
            var_radii_tot = np.concatenate((var_radii_tot, var_radii))

        return init_radii, mean_radii, var_radii

    def plot(self, result):
        init_radii, mean_radii, var_radii = result
        init_radii /= self.params.radius
        mean_radii /= self.params.radius
        var_radii /= self.params.radius**2
        fig = plt.figure(figsize=(5 * 2, 3.5))
        ax_mean = fig.add_subplot(1, 2, 1)
        ax_var = fig.add_subplot(1, 2, 2)
        ax_mean.scatter(init_radii, mean_radii, marker=',', alpha=0.05, lw=0, s=1)
        ax_var.scatter(init_radii, np.sqrt(var_radii), marker=',', alpha=0.05, lw=0, s=1)
        for ax in (ax_mean, ax_var):
            ax.axvline(self.params.boundary_conv / self.params.radius,
                ls="-", label=r"$r_{\rm conv}$", lw=1, c="black")
            ax.axvline((self.params.l_max_heatflux + self.params.boundary_conv) / self.params.radius,
                ls="--", label=r"$l_{\rm max}(\mathbf{f}_{\delta T})$", lw=1, c="black")
            ax.axvline((self.params.l_max_ekinflux + self.params.boundary_conv) / self.params.radius,
                ls="-.", label=r"$l_{\rm max}(\mathbf{f}_k)$", lw=1, c="black")
            ax.legend()
        ax_mean.set_xlabel("Initial radius")
        ax_mean.set_ylabel("Mean radius")
        ax_var.set_xlabel("Initial radius")
        ax_var.set_ylabel("Std. dev. of radius")
        fig.tight_layout()
        return fig


if __name__ == "__main__":
    ParticleMeanVariance().run()

