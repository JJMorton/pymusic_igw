#!/usr/bin/env python3

import numpy as np
import logging

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pymusic.particles as pmp
import pymusic.particles.rm2017 as rm17
import pymusic.utils as pmu
from pymusic_igw import AnalysisTask


# Set up logging
logger = logging.getLogger(__name__)


class ParticleDiffusion(AnalysisTask):

    def __init__(self):
        super().__init__(f"particle_diffusion")

    def compute(self):
        # Get all tracer files in dump directory
        dumps_dir = Path(self.base_dir, "particle_dumps/")
        if not dumps_dir.exists():
            logger.error("Particle dumps not found at " + str(dumps_dir))
            return
        NUM_DUMPS = 200
        files = sorted(dumps_dir.glob("*.tracers.h5"))[-NUM_DUMPS:]
        dumps = [pmp.ParticleDumpFromFile(f) for f in files]
        num_dumps = len(dumps)
        num_particles = len(dumps[0].dataframe())
        logger.info(f"{num_dumps} particle dumps from t={dumps[0].time:E} to t={dumps[-1].time:E}")

        res_r = 500
        taus = np.array([1, 2, 5, 10, 20])
        g = rm17.RGrid(self.domain_r[0], self.domain_r[1], res_r)
        Ps = np.zeros((len(taus), res_r))
        Qs = np.zeros((len(taus), res_r))
        for i, tau in enumerate(taus):
            d = rm17.SummedDisplProfile(
                [
                    rm17.DisplProfileFromDumpPair(di, df, g)
                    for (di, df) in pmu.LaggedPairs(dumps, lag=tau, shift=1)
                ],
                show_progress=True,
            )
            df = d.dataframe()
            Qs[i] = df["Q"] / df["n"]
            Ps[i] = df["P"] / df["n"]

        return g.faces(), taus, Ps, Qs

    def plot(self, result):
        rgrid, taus, Ps, Qs = result
        rgrid /= self.params.radius
        fig, (ax1, ax2) = plt.subplots(2, 1)
        for tau, P, Q in zip(taus, Ps, Qs):
            ax1.semilogy(rgrid, P, label=f"tau={tau}")
            ax2.semilogy(rgrid, Q, label=f"tau={tau}")
        ax1.set_xlabel(r"r/R")
        ax2.set_xlabel(r"r/R")
        ax1.set_ylabel(r"P(r)")
        ax2.set_ylabel(r"Q(r)")
        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        return fig

if __name__ == "__main__":
    ParticleDiffusion().run()

