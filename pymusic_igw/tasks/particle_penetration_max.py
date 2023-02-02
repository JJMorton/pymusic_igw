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


class ParticlePenetration(AnalysisTask):

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

        # Filter dumps to contain only particles that started in the convective region
        logger.info("Filtering dumps...")
        first_df = dumps[0].dataframe()
        gids = first_df[self.is_in_convective_zone(first_df["x1"])].index.astype(int)
        dumps = [pmp.DumpFilteredByGids(dump, gids) for dump in dumps]
        # Now, only iterate particles that are in the radiative zone (i.e. have penetrated)
        dumps = [pmp.DumpFilteredByFunc(dump, lambda df: ~self.is_in_convective_zone(df["x1"])) for dump in dumps]

        logger.info("Finding the maximum penetration depth for each particle")
        maxp = {}
        for i, dump in enumerate(dumps):
            df = dump.dataframe()
            logger.info(f"Analysing particle dump {i} ({len(df)} particles)")
            for gid, series in df.iterrows():
                pen = np.abs(series["x1"] - self.params.boundary_conv)
                maxp[gid] = max(maxp.get(gid) or 0, pen)

        return np.array(list(maxp.values())),

    def plot(self, result):
        # Render the plots and create the video
        pens, = result
        pens /= self.params.radius
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(pens, bins=40)
        ax.axvline(self.params.l_max_heatflux / self.params.radius, ls="--", label=r"$l_{\rm max}(\mathbf{f}_{\delta T})$")
        ax.axvline(self.params.l_max_ekinflux / self.params.radius, ls="-.", label=r"$l_{\rm max}(\mathbf{f}_k)$")
        ax.legend()
        ax.set_xlabel("Maximum penetration (stellar radii)")
        ax.set_ylabel("Frequency")
        ax.set_yticks([])
        ax.set_yticklabels([])
        fig.tight_layout()
        return fig

if __name__ == "__main__":
    ParticlePenetration().run()

