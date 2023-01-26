#!/usr/bin/env python3

import numpy as np
import logging
import itertools

import numpy as np
from pathlib import Path
from pymusic.big_array import CachedArray
import pymusic.particles as pmp
from pymusic.plotting import PlotOnSameAxes, SinglePlotFigure, FfmpegMp4Movie
from pymusic.particles.plotting import Spherical2DParticlesPlot, Spherical2DDomainBounds

from pymusic_igw import AnalysisTask, Spherical2DArrayPlot, SymmetricFixedBounds


# Set up logging
logger = logging.getLogger(__name__)

FIELD = "vel_1" # The hydro field to plot
NUM_PARTICLE_DUMPS = 120 # The number of particle dumps to animate
SATURATE_CONV = True # If True, set the colorbar range to that of only the radiative zone
SATURATE_OVERSHOOT_EXCLUDE = 0.05 # If SATURATE_CONV, exclude this many stellar radii outside the convective zone when calculating the colour scale
THETA_EPSILON = np.radians(0.1) # Allowed variation in theta (radians) for particles along the spokes
SPOKE_SPACING = np.pi / 16 # Angular spacing between spokes of highlighted particles
NUM_PARTICLES_IN_SPOKE = 100 # Number of particles in each spoke

class ParticleMovie(AnalysisTask):

    def __init__(self):
        super().__init__(f"particle_movie_{FIELD}", plot_ext="mp4")

    def compute(self):
        # Get the hydro field to be plotted in the background
        if FIELD == "vel":
            vels = self.sim_data.take(["vel_1", "vel_2"], "var").sum_abs2("var").sqrt()
        else:
            vels = self.sim_data.xs(FIELD, "var")
        vels = CachedArray(vels)
        logger.info(f"Read velocities: {vels}")
        radii = np.array(self.sim_data.labels_along_axis("x1"))
        thetas = np.array(self.sim_data.labels_along_axis("x2"))

        # Get all tracer files in dump directory
        dumps_dir = Path(self.base_dir, "particle_dumps/")
        if not dumps_dir.exists():
            logger.error("Particle dumps not found at " + str(dumps_dir))
            return
        files = sorted(dumps_dir.glob("*.tracers.h5"))
        dumps = [pmp.ParticleDumpFromFile(f) for f in files]
        r_bounds = (radii[0], radii[-1])
        theta_bounds = (thetas[0], thetas[-1])

        # Collect together MUSIC and particle dumps
        hydro_and_particle_data = pmp.SynchedHydroAndParticleData(hydro_data=vels, particle_seq=dumps) 
        synched_times = hydro_and_particle_data.times
        if len(synched_times) < NUM_PARTICLE_DUMPS:
            logger.error(f"Not enough overlap between MUSIC and tracer dumps to make {NUM_PARTICLE_DUMPS} frames")
            return
        index_start = len(synched_times) - NUM_PARTICLE_DUMPS
        logger.info(f"First frame at t={synched_times[index_start]:E}s")

        # Get the initial positions of all the particles
        first_df = hydro_and_particle_data[index_start][2].dataframe()
        init_radii = np.array(first_df["x1"])
        init_thetas = np.array(first_df["x2"])

        # Get the GIDs of the particles along the spokes
        theta_range = theta_bounds[1] - theta_bounds[0]
        filter_spokes = np.fmod(init_thetas, SPOKE_SPACING) < THETA_EPSILON # bool array to filter particles along spokes
        filter_conv = self.is_in_convective_zone(init_radii) # bool array to filter particles in the convective zone
        gids_conv = first_df[np.logical_and(filter_spokes, filter_conv)].index # GIDs of the particles in the convective zone
        gids_rad = first_df[np.logical_and(filter_spokes, ~filter_conv)].index # GIDs of the particles in the radiative zone

        # Limit the number of tracer particles to highlight
        num_spokes = int(theta_range / SPOKE_SPACING)
        # gids_conv = gids_conv[:min(NUM_PARTICLES_IN_SPOKE * num_spokes, len(gids_conv))]
        logger.info(f"Chosen {len(gids_conv)} particles in convective zone")
        # gids_rad = gids_rad[:min(NUM_PARTICLES_IN_SPOKE * num_spokes, len(gids_rad))]
        logger.info(f"Chosen {len(gids_rad)} particles in radiative zone")

        # Compute the colorbar range
        if SATURATE_CONV:
            # Fit the colorbar range to only the radiative zone
            overshoot_outer = self.params.boundary_conv + SATURATE_OVERSHOOT_EXCLUDE * self.params.radius
            overshoot_inner = self.params.boundary_conv - SATURATE_OVERSHOOT_EXCLUDE * self.params.radius
            r_filter = self.is_in_convective_zone(radii) | ((radii > overshoot_inner) & (radii < overshoot_outer))
            r_filter = ~r_filter
            arr = (hydro_and_particle_data.hydro_data
                .take(synched_times, "time")
                .take(radii[r_filter], "x1")
                .array())
        else:
            # Fit the colorbar range to the entire domain
            arr = (hydro_and_particle_data.hydro_data
                .take(synched_times, "time")
                .array())
        color_bounds = SymmetricFixedBounds(vrange=np.abs(arr).max())

        # We have to make the plots while we still have all the dump files available
        logger.info(f"Creating {NUM_PARTICLE_DUMPS} animation frames")
        plots=[
            PlotOnSameAxes(
                [
                    Spherical2DDomainBounds( # the bounds of the domain
                        r_bounds=r_bounds, theta_bounds=theta_bounds
                    ),
                    Spherical2DDomainBounds( # The convective-radiative boundary
                        r_bounds=(r_bounds[0], self.params.boundary_conv), theta_bounds=theta_bounds
                    ),
                    Spherical2DParticlesPlot( # particles in the convective region
                        pmp.DumpFilteredByGids(
                            particles, gids=gids_conv
                        ),
                        color=lambda _: "black",
                        scale=lambda _: 2.0,
                    ),
                    Spherical2DParticlesPlot( # particles in the radiative region
                        pmp.DumpFilteredByGids(
                            particles, gids=gids_rad
                        ),
                        color=lambda _: "black",
                        scale=lambda _: 2.0,
                    ),
                    Spherical2DArrayPlot( # The hydro field
                        hydro,
                        cmap="bwr",
                        with_colorbar=True,
                        color_bounds=color_bounds,
                    ),
                ]
            )
            for time, hydro, particles in itertools.islice(hydro_and_particle_data, index_start, None)
        ]

        figs = [SinglePlotFigure(plot=plot, figsize=(4, 8)).figure() for plot in plots]
        return (figs,)

    def plot(self, result):
        # Render the plots and create the video
        figs, = result
        save_dir = Path(self._get_plot_path().parent, "particle_movie/")
        save_dir.mkdir(exist_ok=True)
        for f in save_dir.glob("*.png"):
            f.unlink()
        FfmpegMp4Movie(figs, save_dir, framerate=20).render_to(self._get_plot_path())

if __name__ == "__main__":
    ParticleMovie().run()
