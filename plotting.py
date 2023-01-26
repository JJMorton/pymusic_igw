from typing import Tuple, Callable
from pymusic.plotting import Plot, BoundsFromMinMax
from pymusic.big_array import BigArray
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Spherical2DArrayPlot(Plot):
    
    array: BigArray
    cmap: str = "jet"
    with_colorbar: bool = True
    color_bounds: Callable[[NDArray], Tuple[float, float]] = BoundsFromMinMax()
    
    def draw_on(self, ax: plt.Axes) -> None:
        r = self.array.labels_along_axis("x1")
        theta = self.array.labels_along_axis("x2")
        R, Theta = np.meshgrid(r, theta)
        X = R*np.sin(Theta)
        Y = R*np.cos(Theta)
        arr = self.array.array()
        vmin, vmax = self.color_bounds(arr)
        mesh = ax.pcolormesh(X, Y, arr.T[:-1, :-1], cmap=self.cmap, shading="flat", vmin=vmin, vmax=vmax)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        if self.with_colorbar:
            ax.figure.colorbar(mesh, ax=ax)
