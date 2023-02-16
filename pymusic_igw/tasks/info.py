#!/usr/bin/env python3

import logging
import numpy as np
from pymusic_igw import AnalysisTask
from pymusic_igw.v_rms_methods import e_kin_density_array
from pymusic.big_array import BigArray
from pymusic.math import SphericalMidpointQuad1D


logger = logging.getLogger(__name__)


def v_rms_theta(sim_data: BigArray, quad: SphericalMidpointQuad1D, theta_axis: str = "x2") -> np.ndarray:
    """
    Calculate v_rms(t, r). Only averaged over theta
    """
    Ek_avg_theta = (
        e_kin_density_array(sim_data)
        .collapse(quad.average, theta_axis)
        # .slabbed("time", 100)
        .sqrt()
    )
    rho_avg_theta = (
        sim_data.xs("density", axis="var")
        .collapse(quad.average, theta_axis)
        # .slabbed("time", 100)
        .sqrt()
    )
    return np.sqrt(2) * Ek_avg_theta.array() / rho_avg_theta.array()


def calc_tau_conv(sim_data: BigArray, quad: SphericalMidpointQuad1D, boundary_conv: float, core_conv: bool) -> float:
    '''
    Calculate convective turnover time (Le Saux et al. 2022, Eq. 5)
    '''
    logger.info("Calculating convective timescale...")
    times = np.array(sim_data.labels_along_axis("time"))
    average_period = 1e7 # The time period to integrate over

    # First filter the data to only the convective region
    filter_radius = (lambda r: r < boundary_conv) if core_conv else (lambda r: r > boundary_conv)
    filter_time = lambda t: t > (float(times[-1]) - average_period)
    conv_data = sim_data.take_filter(filter_radius, axis="x1").take_filter(filter_time, axis="time")
    times = np.array(conv_data.labels_along_axis("time"))
    logger.info(f"Using {len(conv_data.labels_along_axis('time'))} timesteps (covering {times[-1] - times[0]:E}s)")

    # Calculate v_rms(t, r)
    v_rms = v_rms_theta(conv_data, quad)

    # Calculate the timescale
    radii_conv = np.array(conv_data.labels_along_axis("x1"))
    dr = (radii_conv[-1] - radii_conv[0]) / len(radii_conv)
    tau = np.mean(np.sum(1.0 / v_rms, axis=1) * dr)
    logger.info("Done.")

    return tau


class BlankTask(AnalysisTask):
    
    def __init__(self):
        super().__init__("blank")

    def compute(self):
        times = np.array(self.sim_data.labels_along_axis("time"))
        logger.info(f"Start time: {times[0]:.2E}")
        logger.info(f"End time: {times[-1]:.2E}")

        logger.info(f"Radius bounds: ({self.domain_r[0]:.2E}, {self.domain_r[1]:.2E})")
        logger.info(f"Theta bounds: ({self.domain_theta[0]/np.pi:.2f}pi, {self.domain_theta[1]/np.pi:.2f}pi)")

        tau_conv = calc_tau_conv(self.sim_data, self.quad, self.params.boundary_conv, self.params.core_conv)
        logger.info(f"tau_conv = {tau_conv:.3E}")

        return None

    def plot(self):
        pass

if __name__ == "__main__":
    BlankTask().run()

