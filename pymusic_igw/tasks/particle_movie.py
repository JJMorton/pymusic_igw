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
NUM_PARTICLE_DUMPS = 40 # The number of particle dumps to animate, set to zero to use all dumps
DOMAIN_R = (0.23, 0.33) # The limits of the domain in r (stellar radii), use (None, None) for entire domain
DOMAIN_THETA = (np.pi*0.4, np.pi*0.6) # The limits of the domain in theta, use (None, None) for entire domain
SATURATE_CONV = True # If True, set the colorbar range to that of only the radiative zone
SATURATE_OVERSHOOT_EXCLUDE = 0.05 # If SATURATE_CONV, exclude this many stellar radii outside the convective zone when calculating the colour scale
THETA_EPSILON = np.radians(0.1) # Allowed variation in theta (radians) for particles along the spokes
SPOKE_SPACING = np.pi / 20 # Angular spacing between spokes of highlighted particles

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
        logger.info(f"Read MUSIC dumps: {vels}")
        radii = np.array(self.sim_data.labels_along_axis("x1"))
        thetas = np.array(self.sim_data.labels_along_axis("x2"))
        r_bounds = ((DOMAIN_R[0] * self.params.radius) or radii[0], (DOMAIN_R[1] * self.params.radius) or radii[-1])
        theta_bounds = (DOMAIN_THETA[0] or thetas[0], DOMAIN_THETA[1] or thetas[-1])

        def get_visible_gids(part_dump):
            df = part_dump.dataframe()
            part_radii = np.array(df["x1"])
            part_thetas = np.array(df["x2"])
            filter_domain = np.logical_and.reduce((
                part_radii > r_bounds[0],
                part_radii < r_bounds[1],
                part_thetas > theta_bounds[0],
                part_thetas < theta_bounds[1]
            ))
            gids = df[filter_domain].index # GIDs of the particles within the domain
            return gids

        # Get all tracer files in dump directory
        dumps_dir = Path(self.base_dir, "particle_dumps/")
        if not dumps_dir.exists():
            logger.error("Particle dumps not found at " + str(dumps_dir))
            return
        files = sorted(dumps_dir.glob("*.tracers.h5"))
        dumps = [pmp.ParticleDumpFromFile(f) for f in files]
        NUM_PARTICLE_DUMPS = NUM_PARTICLE_DUMPS or len(dumps)

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
        gids_conv = first_df[np.logical_and.reduce((filter_spokes, filter_conv))].index # GIDs of the particles in the convective zone
        gids_rad = first_df[np.logical_and.reduce((filter_spokes, ~filter_conv))].index # GIDs of the particles in the radiative zone
        logger.info(f"Chosen {len(gids_conv)} particles in convective zone")
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
                            particles, gids=np.intersect1d(gids_conv, get_visible_gids(particles))
                        ),
                        color=lambda _: "black",
                        scale=lambda _: 2.0,
                    ),
                    Spherical2DParticlesPlot( # particles in the radiative region
                        pmp.DumpFilteredByGids(
                            particles, gids=np.intersect1d(gids_rad, get_visible_gids(particles))
                        ),
                        color=lambda _: "lime",
                        scale=lambda _: 3.0,
                    ),
                    Spherical2DArrayPlot( # The hydro field
                        # Limit to specified domain
                        (hydro
                            .take(radii[np.logical_and(radii > r_bounds[0], radii < r_bounds[1])], "x1")
                            .take(thetas[np.logical_and(thetas > theta_bounds[0], thetas < theta_bounds[1])], "x2")),
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
