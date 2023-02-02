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
        super().__init__(f"particle_penetration")

    def compute(self):
        # Get all tracer files in dump directory
        dumps_dir = Path(self.base_dir, "particle_dumps/")
        if not dumps_dir.exists():
            logger.error("Particle dumps not found at " + str(dumps_dir))
            return
        files = sorted(dumps_dir.glob("*.tracers.h5"))
        dumps = [pmp.ParticleDumpFromFile(f) for f in files]
        logger.info(f"{len(dumps)} particle dumps from t={dumps[0].time:E} to t={dumps[-1].time:E}")

        particles = {}
        for gid in dumps[0].dataframe().index:
            particles[gid] = []
        for i, dump in enumerate(dumps):
            df = dump.dataframe()
            logger.info(f"Analysing particle dump {i} ({len(df)} particles)")
            for gid, series in df.iterrows():
                particles[gid].append(series["x1"])

        init_radii = np.array([r[0] for r in particles.values()])
        mean_radii = np.array([np.mean(r) for r in particles.values()])
        var_radii = np.array([np.var(r) for r in particles.values()]) # sum[(r - mean)**2] / N

        return init_radii, mean_radii, var_radii

    def plot(self, result):
        init_radii, mean_radii, var_radii = result
        init_radii /= self.params.radius
        mean_radii /= self.params.radius
        var_radii /= self.params.radius**2
        fig = plt.figure(figsize=(4 * 2, 3))
        ax_mean = fig.add_subplot(1, 2, 1)
        ax_var = fig.add_subplot(1, 2, 2)
        ax_mean.scatter(init_radii, mean_radii, marker=',')
        ax_var.scatter(init_radii, var_radii, marker=',')
        ax_mean.set_xlabel("Initial radius (stellar radii)")
        ax_mean.set_ylabel("Mean radius (stellar radii)")
        ax_var.set_xlabel("Initial radius (stellar radii)")
        ax_var.set_ylabel("Variance of radial position (stellar radii sq)")
        fig.tight_layout()
        return fig

if __name__ == "__main__":
    ParticleMeanVariance().run()

