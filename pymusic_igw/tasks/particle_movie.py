#!/usr/bin/env python3

import numpy as np
import logging

import numpy as np
from pathlib import Path
from pymusic.big_array import CachedArray
import pymusic.particles as pmp
from pymusic.plotting import FixedBounds, PlotOnSameAxes, SinglePlotFigure, FfmpegMp4Movie
from pymusic.particles.plotting import Spherical2DParticlesPlot, Spherical2DDomainBounds

from pymusic_igw import AnalysisTask, Spherical2DArrayPlot


# Set up logging
logger = logging.getLogger(__name__)

FIELD = "vel" # The hydro field to plot
NUM_PARTICLE_DUMPS = 120 # The number of particle dumps to animate

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
        print(f"Read velocities: {vels}")
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
        hydro_and_particle_data = pmp.SynchedHydroAndParticleData(hydro_data=vels, particle_seq=dumps[:NUM_PARTICLE_DUMPS]) 

        # Find the GIDs of tracer particles at specific radii
        first_df = dumps[0].dataframe()
        init_radii = np.array(first_df["x1"])
        gids_boundary = first_df[np.abs(init_radii - self.params.boundary_conv) < 0.001 * self.params.radius].index # Convective-radiative boundary
        gids_conv = first_df[np.abs(init_radii - 0.5*(self.params.boundary_conv + r_bounds[0])) < 0.001 * self.params.radius].index # Convective region
        gids_rad = first_df[np.abs(init_radii - 0.5*(self.params.boundary_conv + r_bounds[1])) < 0.001 * self.params.radius].index # Radiative region

        # Limit the number of tracer particles to highlight
        gids_boundary = gids_boundary[:min(1000, len(gids_boundary))]
        print(f"Chosen {len(gids_boundary)} particles near the convective boundary")
        gids_conv = gids_conv[:min(1000, len(gids_conv))]
        print(f"Chosen {len(gids_conv)} particles in the convective zone")
        gids_rad = gids_rad[:min(30, len(gids_rad))] # Few particles in radiative zone, to make longitudinal motion evident
        print(f"Chosen {len(gids_rad)} particles in the radiative zone")

        arr = hydro_and_particle_data.hydro_data.take(hydro_and_particle_data.times, "time").array()
        color_bounds = FixedBounds(vmin=arr.min(), vmax=arr.max())

        # We have to make the plots while we still have all the dump files available
        plots=[
            PlotOnSameAxes(
                [
                    # ... at the very top layer:
                    Spherical2DDomainBounds(  # the bounds of the domain
                        r_bounds=r_bounds, theta_bounds=theta_bounds
                    ),
                    # ... then:
                    Spherical2DParticlesPlot(  # particles at the boundary
                        pmp.DumpFilteredByGids(
                            particles, gids=gids_boundary
                        ),
                        color=lambda _: "lightgreen",
                        scale=lambda _: 5.0,
                    ),
                    Spherical2DParticlesPlot(  # particles in the convective region
                        pmp.DumpFilteredByGids(
                            particles, gids=gids_conv
                        ),
                        color=lambda _: "white",
                        scale=lambda _: 3.0,
                    ),
                    Spherical2DParticlesPlot(  # particles in the radiative region
                        pmp.DumpFilteredByGids(
                            particles, gids=gids_rad
                        ),
                        color=lambda _: "orange",
                        scale=lambda _: 5.0,
                    ),
                    Spherical2DArrayPlot(
                        hydro,
                        cmap="magma",
                        with_colorbar=False,
                        color_bounds=color_bounds,
                    ),
                    # Spherical2DParticlesPlot(  # all of the particles
                    #     dump,
                    #     color=lambda df: df["attr:maxp_x1"],  # ...colored by maxp_x1...
                    #     cmap="magma",
                    #     scale=lambda df: 1.0,  # ...using a small marker size...
                    # ),
                ]
            )
            for time, hydro, particles in hydro_and_particle_data
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
