from pathlib import Path
import os.path
from typing import Any, Tuple, List
import pickle
import matplotlib.pyplot as plt
import argparse
import logging
from abc import ABC, abstractmethod
import json
from numpy import float64
from numpy.typing import NDArray
from dataclasses import dataclass

from pymusic.big_array import BigArray
from pymusic.io.music import ArrayBC, MusicSim
from pymusic.io.music_new_format import MusicDumpInfo
from pymusic.math import SphericalMidpointQuad1D
from pymusic.io.music import ReflectiveArrayBC, PeriodicArrayBC

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

@dataclass(frozen=True)
class Params():

	radius: float64 # Stellar radius
	mass: float64 # Stellar mass
	boundary_conv: float64 # Radius of the convective-radiative interface
	core_conv: bool # Is the core convective?
	boundary_conds: Tuple[ArrayBC, ArrayBC] # The boundary conditions for r and theta
	tau_conv: float64 # The convective timescale

	@classmethod
	def fromJSON(cls, filename: Path):
		parse_bc = lambda bc: ReflectiveArrayBC() if bc == "r" else PeriodicArrayBC()
		try:
			with open(filename, "r") as f:
				data = json.load(f)
				p = Params(
					radius = data["radius"],
					mass = data["mass"],
					boundary_conv = data["boundary_conv"] * data["radius"],
					core_conv = data["core_conv"],
					boundary_conds = tuple(parse_bc(bc) for bc in (data["boundary_r"], data["boundary_theta"])),
					tau_conv = data["tau_conv"],
				)
		except FileNotFoundError:
			logger.error("Missing " + filename.as_posix())
			return None
		except KeyError:
			logger.error("Missing key in " + filename.as_posix())
			return None

		Rsun = 6.957e8
		Msun = 1.98847e30
		print("PARAMS: M={:.2f}M_sun R={:.2f}R_sun boundary_conv={:.3f}R convective-{} tau_conv={:.2E}s BC=({})".format(
			p.mass / Msun,
			p.radius / Rsun,
			p.boundary_conv / p.radius,
			"core" if p.core_conv else "envelope",
			p.tau_conv,
			','.join(["reflective" if type(BC) is ReflectiveArrayBC else "periodic" for BC in p.boundary_conds])
		))
		return p


class AnalysisTask(ABC):
	'''
	A class to provide a wrapper for analysis scripts using `pymusic`.
	The analysis is split into the 'compute' part and the 'plot' part.
	This makes it easy to run the computationally intensive part remotely, and plot the
	results on your local machine as many times as you would like.
	'''

	name: str
	plot_ext: str
	verbose: bool = False
	dump_files: List[str] = []
	sim: MusicSim
	sim_data: BigArray
	quad: SphericalMidpointQuad1D
	params: Params
	base_dir: str = "./"


	def __init__(self, name: str, plot_ext: str = "png"):
		self.name = name
		self.plot_ext = plot_ext


	@abstractmethod
	def compute(self) -> Any:
		pass


	@abstractmethod
	def plot(self, result: Any) -> plt.Figure:
		pass


	def run(self):

		'''
		Parse command-line arguments and run `compute()` and/or `plot()` as requested
		'''
		parser = argparse.ArgumentParser()
		parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
		parser.add_argument("--clean", action="store_true", help="clean old .pkl files")
		parser.add_argument("-l", "--location", default="./", metavar="<dir>", help="the directory containing the 1D model, gyre output, etc.")
		parser.add_argument("-p", "--plot", action="store_true", help="plot the previously computed result")
		parser.add_argument("-c", "--compute", type=str, nargs='+', metavar="<dump files>", help="compute the analysis using the specified MUSIC dump files")
		args = parser.parse_args()

		self.verbose = args.verbose

		self.base_dir = args.location

		p = Params.fromJSON(Path(self.base_dir, "params.json"))
		if p is None:
			logger.info("Specify the directory containing the correct params.json with -l (see --help for info)")
			return
		else:
			self.params = p

		if args.compute:
			self._run_compute(args.compute)
		if args.plot:
			self._run_plot()

		if args.clean:
			logger.info("Removing old .pkl files...")
			for file in self._get_result_path().parent.glob(self._get_result_path().name + ".*"):
				if self.verbose: logger.info(f"Removing {file}")
				Path(file).unlink()

		if (not args.compute) and (not args.plot) and (not args.clean):
			parser.print_help()


	def is_in_convective_zone(self, radii) -> NDArray[bool]:
		if self.params.core_conv:
			return radii < self.params.boundary_conv
		else:
			return radii > self.params.boundary_conv


	def _get_result_path(self) -> Path:
		name = self.name
		dir_path = Path(self.base_dir, "output")

		# Make the output directory, if it doesn't already exist
		Path(dir_path).mkdir(parents=True, exist_ok=True)

		return Path(os.path.join(dir_path, f'{name}.pkl'))


	def _get_plot_path(self) -> Path:
		name = self.name
		dir_path = Path(self.base_dir, "plots")

		# Make the output directory, if it doesn't already exist
		Path(dir_path).mkdir(parents=True, exist_ok=True)

		return Path(os.path.join(dir_path, name + '.' + self.plot_ext))


	def _run_compute(self, dump_files: List[str]):
		'''
		Run the `compute` method, and save the result to a file using pickle
		'''
		logger.info("===== Running analysis task =====")

		# Setup simulation from dump files
		logger.info(f"Reading {len(dump_files)} dump files...")
		self.dump_info = MusicDumpInfo(num_scalars=0, num_space_dims=2, num_velocities=2)
		self.dump_files = sorted(dump_files)
		self.sim = MusicSim.from_dump_file_names(dump_files, self.dump_info, self.params.boundary_conds)
		self.sim_data = self.sim.big_array(verbose=self.verbose)
		self.quad = SphericalMidpointQuad1D(self.sim.point_grid(axis=1))
		logger.info("Done!")
		logger.info(f"sim_data: {self.sim_data}")
		logger.info(f"Fields available under the 'var' axis: {self.sim_data.labels_along_axis('var')}")

		file_path = self._get_result_path()
		logger.info(f"Will save analysis result to '{file_path}'")

		# If the output file already exists, rename it to have a number suffix so we don't overwrite it
		if os.path.exists(file_path):
			i = 1
			# while os.path.exists(os.path.join(dir_path, f'{name}_{i}.npy')):
			while os.path.exists(file_path.as_posix() + f".{i}"):
				i += 1
			backup_file_path = file_path.as_posix() + f".{i}"
			Path(file_path).rename(backup_file_path)
			logger.warning(f"Moved '{file_path}' to '{backup_file_path}' to avoid overwriting")

		# Perform the computation
		result = self.compute()
		if result is None:
			return

		# Save the result to the pickled file
		with open(file_path, "wb") as f:
			pickle.dump(result, f)
		logger.info("Saved!")


	def _run_plot(self):
		'''
		Read the saved result and save a plot
		'''
		logger.info("===== Plotting analysis task =====")
		file_path = self._get_result_path()
		if not os.path.exists(file_path):
			logger.error(f"Result not found at '{file_path}', run the analysis task before plotting")
			return
		logger.info(f"Reading result from '{file_path}'")
		with open(file_path, "rb") as f:
			result = pickle.load(f)
		figure = self.plot(result)
		if figure is not None:
			plot_file_path = self._get_plot_path()
			logger.info(f"Saving plot to '{plot_file_path}'")
			figure.savefig(plot_file_path)
			logger.info("Saved!")
		else:
			logger.warning("No plot returned from analysis task")
