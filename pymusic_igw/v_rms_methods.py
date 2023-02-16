r"""
Written by A. Le Saux and T. Guillet.

Methods for time- and angle-averaged RMS velocities
===================================================

This script compares different implementations of RMS velocity averages.

Defining the notations:

 * :math:`v_p = [v_r, v_\theta], p = [1, 2]`
 * :math:`[ \cdot ]_t = {N_t}^{-1} \sum_t ( \cdot )`
 * :math:`[ \cdot ]_\nu = {N_\nu}^{-1} \sum_\nu ( \cdot )` (average in frequency domain, :math:`N_\nu = N_t`)
 * :math:`[ \cdot ]_\theta = {N_\theta}^{-1} \sum_\theta( \cdot )` (naive averaging on :math:`\theta` values)
 * :math:`[ \cdot ]_S = (4\pi)^{-1} \int_S ( \cdot ) \mathrm{d}\Omega`
   (spherical integral average, approximated using quadrature rule)

the methods are given by:

``Isa``
    .. autofunction:: v_rms_isa

``Isa_sphere``
    .. autofunction:: v_rms_isa_sphere

``Kinetic_energy_velocity``
    .. autofunction:: v_rms_ekin

``Samadi``
    .. autofunction:: v_rms_samadi

``Samadi_spherical_harmonics``
    .. autofunction:: v_rms_samadi_sph_harm

``Samadi_spectral``
    .. autofunction:: v_rms_samadi_spectral

``Time_average_first``
    .. autofunction:: v_rms_time_av_first

``Belkacem_time``
    .. autofunction:: v_rms_belkacem_time

``Belkacem_freq``
    .. autofunction:: v_rms_belkacem_freq

``Raw_second_moment``
    .. autofunction:: v_rms_raw_second_moment

``Belkacem_freq_no_sub_mean``
    .. autofunction:: v_rms_belkacem_freq_no_sub_mean
"""

import numpy as np

from pymusic.big_array import BigArray
from pymusic.math import SphericalMidpointQuad1D
from pymusic.spec import NuFFT1D, NoWindow

from pymusic.big_array import ItemsIndex1d
from pymusic.big_array import CachedArray, MultiApplyArray, SphHarm1DArray, FFTArray
from pymusic.big_array import e_kin_density_array

ONE_OVER_FOUR_PI = 1.0 / (4.0 * np.pi)


def v_rms_isa(v):
    r"""
    :math:`v_\mathrm{rms} = \left[ \sqrt{[\sum_p v_p^2]_\theta} \right]_t`
    """
    return (
        v.abs2()
        .sum("var")  # sum on p components
        .mean("x2")  # theta average
        .sqrt()
        .mean("time")
    ).array()


def v_rms_isa_sphere(v, quad):
    r"""
    :math:`v_\mathrm{rms} = \left[ \sqrt{[\sum_p v_p^2]_S} \right]_t`
    """
    return (
        v.abs2()
        .sum("var")  # sum on p components
        .collapse(quad.average, "x2")  # spherical average
        .sqrt()
        .mean("time")
    ).array()


def v_rms_ekin(sim_data: BigArray, quad: SphericalMidpointQuad1D) -> np.ndarray:
    r"""Mass-weighted squared velocity
    :mat:`v_/mathrm{rms} = \sqrt{ \left[ \sum_p \rho v_p^2 \right]_{S,t}/[\rho]_{S,t}}`
    """
    Ek_avg = (
        e_kin_density_array(sim_data, "var")
        .slabbed("time", 100)
        .mean("time")
        .collapse(quad.average, "x2")
    )
    rho_avg = (
        sim_data.xs("rho", axis="var")
        .slabbed("time", 100)
        .mean("time")
        .collapse(quad.average, "x2")
    )
    return np.sqrt(2 * Ek_avg.array() / rho_avg.array())


def v_rms_spatial_av_first(v, quad):
    r"""After https://www.aanda.org/articles/aa/abs/2003/19/aa3271/aa3271.html:

    :math:`v_\mathrm{rms} = \sqrt{\left[ \sum_p([v_p^2]_S - [v_p]_S^2) \right]_t}`
    """
    v_mom_savg = CachedArray(
        moments(v).collapse(  # moments v, v^2
            quad.average, "x2"
        )  # spherical average of each moment
    )
    v2s = v_mom_savg.xs(2, "moments").sum("var").mean("time")
    vs2 = v_mom_savg.xs(1, "moments").abs2().sum("var").mean("time")
    return np.sqrt(v2s.array() - vs2.array())


# noinspection PyShadowingNames
def v_rms_spatial_av_first_sph_harm(v, ell_max, sh_xform):
    r"""Samadi method for RMS velocity, computed from SH decomposition:

    :math:`v_\mathrm{rms} = \sqrt{ (4\pi)^{-1} \sum_{p} \sum_{\ell \geq 1} [v_{\ell,p}^2]_t }`

    (Note that the sum on :math:`\ell` excludes the value :math:`\ell=0`).
    This is formally identical to the ``Samadi`` method in the limit :math:`\ell \rightarrow \infty`.
    """
    v_sh = SphHarm1DArray(
                v,
                sh_xform,
                theta_axis="x2",
                ell_axis="ell",
                ells=range(ell_max + 1),
           )
    v_sh_mom_tavg = CachedArray(moments(v_sh).slabbed("time", 100).mean("time"))
    return (
        v_sh_mom_tavg.xs(2, "moments")
        .take(
            range(1, ell_max + 1), "ell"
        )  # Mind start at ell=1 since ell=0 corresponds to subtracted angular average term
        .sum("ell")
        .scaled(ONE_OVER_FOUR_PI)
        .sum("var")
        .sqrt()
    ).array()


def v_rms_spatial_av_first_spectral(v, ell_max, sh_xform):
    r"""Samadi method for RMS velocity, computed from SH decomposition and Fourier transform:

    :math:`v_\mathrm{rms} = \sqrt{ (4\pi N_t)^{-1} \sum_{\nu,p,\ell \geq 1} |\hat v_{\ell,p}(\nu)|^2}`
    (Note that the sum on :math:`\ell` excludes the value :math:`\ell=0`).
    """
    v_sh = SphHarm1DArray(
                v,
                sh_xform,
                theta_axis="x2",
                ell_axis="ell",
                ells=range(ell_max + 1),
            )
    v_ell_freq = v_ell_freq_belkacem(v_sh)
    n_t = v_sh.size_along_axis("time")
    return (
        v_ell_freq.abs2()
        .sum("var")
        .take(range(1, ell_max + 1), "ell")
        .sum("ell")
        .scaled(ONE_OVER_FOUR_PI)
        .sum("freq")
        .scaled(1.0 / n_t)
        .sqrt()
        .array()
    )


def v_rms_time_av_first(v, quad):
    r"""Method for rms velocity in real space, where the time average is performed first and the angular average last:

    :math: `v_\mathrm{rms} = \sqrt{ \left[ \sum_{p} ([v_p^2]_t - [v_p]_t^2) \right]_S }`
    """
    v_mom_savg = CachedArray(moments(v).mean("time"))
    v2s = (
        v_mom_savg.xs(2, "moments")
        .sum("var")
        .collapse(quad.average, "x2")
    )
    vs2 = (
        v_mom_savg.xs(1, "moments")
        .abs2()
        .sum("var")
        .collapse(quad.average, "x2")
    )
    return np.sqrt(v2s.array() - vs2.array())


# noinspection PyShadowingNames
def v_rms_time_av_first_sph_harm(v, ell_max, sh_xform):
    r"""After https://www.aanda.org/articles/aa/abs/2009/04/aa10827-08/aa10827-08.html, eqs (B.5) and (B.6):

    :math:`v_\mathrm{rms} = \sqrt{ (4 \pi)^{-1} \sum_{\ell,p} ([v_{\ell,p}^2]_t - [v_{\ell,p}]_t^2) }`

    where :math:`v_{\ell,p}` is the spherical harmonics transform of the velocity components,
    following the same normalization conventions as Belkacem+2009.
    Note that the factor of :math:`(4 \pi)^{-1}` is missing from the paper but is needed.
    """
    v_sh = SphHarm1DArray(
                v,
                sh_xform,
                theta_axis="x2",
                ell_axis="ell",
                ells=range(ell_max + 1),
            )
    v_sh_mom_tavg = moments(v_sh).slabbed("time", 100).mean("time")
    v_sh_mom_tavg = CachedArray(v_sh_mom_tavg.take(range(ell_max + 1), "ell"))
    vsh_2s = (
        v_sh_mom_tavg.xs(2, "moments").sum("ell").scaled(ONE_OVER_FOUR_PI).sum("var")
    )
    vsh_s2 = (
        v_sh_mom_tavg.xs(1, "moments")
        .abs2()
        .sum("ell")
        .scaled(ONE_OVER_FOUR_PI)
        .sum("var")
    )
    return np.sqrt(vsh_2s.array() - vsh_s2.array())


def v_ell_freq_belkacem(v_sh):
    dt = np.mean(np.diff(v_sh.labels_along_axis("time")))
    fft = NuFFT1D(
        NoWindow(), sampling_period=dt, spacing_tol=0.1
    )  # NoWindow is required for Parseval-Plancherel to hold
    return FFTArray(
        v_sh,
        fft,
        "time",
        "freq",
    )


def v_rms_time_av_first_spectral(v, ell_max, sh_xform):
    r"""Belkacem+2009 in frequency domain,
    after https://www.aanda.org/articles/aa/abs/2009/04/aa10827-08/aa10827-08.html:

    :math:`v_\mathrm{rms} = \sqrt{ (4 \pi N_t)^{-1} \sum_{\ell,p,\nu \neq 0} |\hat v_{\ell,p}(\nu)|^2 }`
    """
    v_sh = SphHarm1DArray(
                v,
                sh_xform,
                theta_axis="x2",
                ell_axis="ell",
                ells=range(ell_max + 1),
            )
    v_ell_freq = v_ell_freq_belkacem(v_sh)
    n_t = v_sh.size_along_axis("time")
    return (
        v_ell_freq.abs2()
        .sum("var")
        .sum("ell")
        .scaled(ONE_OVER_FOUR_PI)
        .take_filter(
            lambda nu: nu != 0.0, "freq"
        )  # time average (nu == 0.0) is subtracted by excluding it before summation
        .sum("freq")
        .scaled(1.0 / n_t)  # cannot use .mean(): would divide by n_t-1 instead of n_t
        .sqrt()
        .array()
    )


def v_rms_raw_second_moment(v, quad):
    r"""Alternative to v_rms_time_average_first and v_rms_spatial_average_first with no substraction of the mean:

    :math: `v_\mathrm{rms} = \sqrt{ \sum_{p} [v_p^2]_{S,t}} = \sqrt{ \sum_{p} [v_p^2]_{t,S}}
    """
    return (
        v.abs2()
        .sum("var")
        .collapse(quad.average, "x2")
        .mean("time")
        .sqrt()
        .array()
    )


def v_rms_belkacem_freq_no_sub_mean(v_sh):
    r"""Belkacem+2009 in frequency domain, without subtracting time averages of first moments,
    after https://www.aanda.org/articles/aa/abs/2009/04/aa10827-08/aa10827-08.html:

    :math:`v_\mathrm{rms} = \sqrt{ (4 \pi N_t)^{-1} \sum_{\ell,p,\nu} |\hat v_{\ell,p}(\nu)|^2 }`

    Equivalent to v_rms_raw_second_moment in spectral domain
    """
    v_ell_freq = v_ell_freq_belkacem(v_sh)
    n_t = v_sh.size_along_axis("time")
    return (
        v_ell_freq.abs2()
        .sum("freq")
        .scaled(1.0 / n_t)
        .sum("var")
        .sum("ell")
        .scaled(ONE_OVER_FOUR_PI)
        .sqrt()
        .array()
    )


def moments(arr):
    """Construct an array with the first and second powers of `arr` along a new "moments" axis"""
    return MultiApplyArray(
        arr,
        [lambda x: x, lambda x: x ** 2],
        ItemsIndex1d("moments", [1, 2]),
        new_iaxis=0,
    )
