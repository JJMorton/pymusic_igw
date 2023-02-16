#!/usr/bin/env python3

import logging
import numpy as np
from pymusic_igw import AnalysisTask

logger = logging.getLogger(__name__)

class BlankTask(AnalysisTask):
    
    def __init__(self):
        super().__init__("blank")

    def compute(self):
        times = np.array(self.sim_data.labels_along_axis("time"))
        logger.info(f"Start time: {times[0]:.2E}")
        logger.info(f"End time: {times[-1]:.2E}")

        logger.info(f"Radius bounds: ({self.domain_r[0]:.2E}, {self.domain_r[1]:.2E})")
        logger.info(f"Theta bounds: ({self.domain_theta[0]/np.pi:.2f}pi, {self.domain_theta[1]/np.pi:.2f}pi)")
        return None

    def plot(self):
        pass

if __name__ == "__main__":
    BlankTask().run()

