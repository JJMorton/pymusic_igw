from pathlib import Path
import warnings
import numpy as np
from pymusic.big_array import BigArray, IndexNd, StackedArray, SummedArray, TakeArray, cached_property
import h5py as h5
import logging

from typing import Iterator, Sequence
from numpy.typing import NDArray


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class NeighbourMeanArray(BigArray):
    """
    Averages each element with its neighbours
    """

    def __init__(self, arr: BigArray, axis: str, neighbours: int):
        """
        :param arr: array to neighbour-average
        :param axis: the axis to average along
        :param neighbours: the number of neighbours to use from either side
        """
        self._array = arr
        self._axis = axis
        self._nn = neighbours

    def _required_labels_for(self, label: object) -> Iterator[object]:
        idx = self._array.index1d(self._axis).ordinal_index(label)
        all_labels = np.array(self._array.labels_along_axis(self._axis))
        num_labels = len(all_labels)
        idx = np.arange(idx-self._nn, idx+1+self._nn)
        idx = idx[np.where((idx >= 0) & (idx < num_labels))]
        for l in all_labels[idx]:
            yield(l)

    def array(self) -> NDArray:
        print(self.shape)
        all_labels = self.labels_along_axis(self._axis)
        axis_index = self.axes.index(self._axis)
        return StackedArray(
            [
                (
                    self._array
                    .take(list(self._required_labels_for(l)), self._axis)
                    .mean(self._axis)
                )
                for l in all_labels
            ],
            self._array.index1d(self._axis),
            axis_index
        ).array()

    def _index(self) -> IndexNd:
        return self._array.index

    def sum(self, axis: str) -> BigArray:
        if axis == self._axis:
            # We have to sum now
            warnings.warn("Summing over neighbour-averaged array elements, you probably don't want to do this")
            return SummedArray(self, axis)
        else:
            # Sum array and then average over neighbours
            return NeighbourMeanArray(self._array.sum(axis), self._axis, self._nn)

    def take(self, labels: Sequence[object], axis: str) -> BigArray:
        needed_labels = labels

        if axis == self._axis:
            def flatten(l):
                return [item for sublist in l for item in sublist]
            # needed_labels contains requested labels and all the neighbours to them
            needed_labels = flatten(self._required_labels_for(l) for l in labels)

        return NeighbourMeanArray(
            self._array.take(needed_labels, axis),
            self._axis,
            self._nn,
        )


class TakeClosestArray(BigArray):
    """
    Take the labels with a value closest to `wanted_labels`
    """

    def __init__(self, arr: BigArray, axis: str, wanted_labels: Sequence[np.float64]):
        """
        :param arr: array to take from
        :param axis: the axis to select labels from
        :param wanted_labels: the axis labels to get the closest to
        """
        self._array = arr
        self._axis = axis
        self._labels_want = np.array(wanted_labels)

    @cached_property
    def _labels(self) -> Sequence[str]:
        all_labels = np.array(self._array.labels_along_axis(self._axis))
        if np.any((self._labels_want < all_labels[0]) | (self._labels_want > all_labels[-1])):
            warnings.warn("Some requested labels outside of axis range")
        labels_idx = [np.abs(all_labels - l).argmin() for l in self._labels_want]
        labels = all_labels[labels_idx]
        return labels

    def array(self) -> NDArray:
        return self._array.take(self._labels, self._axis).array()

    def _index(self) -> IndexNd:
        idx = self._array.index
        return idx.take(self._labels, self._axis)

    def sum(self, axis: str) -> BigArray:
        if axis == self._axis:
            return SummedArray(self, axis)
        else:
            return TakeClosestArray(self._array.sum(axis), self._axis, self._labels_want)

    def take(self, labels: Sequence[object], axis: str) -> BigArray:
        if axis == self._axis:
            # Create new TakeClosestArray with only selected labels
            return TakeClosestArray(
                self,
                self._axis,
                labels,
            )
        else:
            # Forward take to array and then take closest
            return TakeClosestArray(
                self._array.take(labels, axis),
                self._axis,
                self._labels_want,
            )


class HDF5Array(BigArray):
    """
    Evaluates the array in its entirety, and saves to disk as a HDF5 file, complete with axes
    """

    def __init__(self, array: BigArray, path: Path):
        self._array = array
        self._path = Path(path)

        self._saved = False

        # If the file already exists, back it up with a number prefix
        if self._path.exists():
            p = self._path
            i = 0
            while p.exists():
                i += 1
                p = Path.joinpath(self._path.parent, self._path.name + f".{i}")
            self._path.rename(p)
            logger.info(f"Renamed '{self._path.as_posix()}' to '{p.as_posix()}' to avoid overwriting")

    def sum(self, axis: str) -> BigArray:
        return SummedArray(self, axis)
        # return self._array.sum(axis)

    def take(self, labels: Sequence[object], axis: str) -> BigArray:
        return TakeArray(self, labels, axis)
        # return self._array.take(labels, axis)

    def _index(self) -> IndexNd:
        return self._array.index

    def array(self) -> NDArray:
        logger.info(f"Will compute array and save to '{self._path.as_posix()}'")
        arr = self._array.array()
        print(arr.shape)
        logger.info(f"Opening '{self._path.as_posix()}' for writing")
        with h5.File(self._path, "w") as f:
            dset = f.create_dataset("array", data=arr)
            for i, axis in enumerate(self.axes):
                scale = f.create_dataset(axis, data=self.labels_along_axis(axis))
                scale.make_scale(axis)
                dset.dims[i].attach_scale(scale)
                print(i, axis, len(self.labels_along_axis(axis)), scale)
        self._saved = True
        logger.info(f"Written array to '{self._path.as_posix()}'")

        return arr

    @property
    def saved(self) -> bool:
        return self._saved

    @property
    def hdf(self) -> h5.File:
        return h5.File(self._path, "r+")
