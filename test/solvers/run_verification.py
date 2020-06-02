#!/usr/bin/env python3

import sys
sys.path.append('../..')
from srlife import materials, thermal

from thermalsol import ManufacturedSolution

import numpy as np

import matplotlib.pyplot as plt

problems = [
    ManufacturedSolution(1, 
      lambda t, r: t, 
      lambda t, k, alpha, r: k/alpha * (r*0.0+1)),
    ManufacturedSolution(1, 
      lambda t, r: np.sin(t)*np.log(r), 
      lambda t, k, alpha, r: k/alpha * np.log(r) * np.cos(t)),
    ManufacturedSolution(1,
      lambda t, r: np.sin(r),
      lambda t, k, alpha, r: k * np.sin(r) - k/r*np.cos(r))
    ]


def run_with(solver, material):
  """
    Run the standard problems with the provided solver and material
  """
  for problem in problems:
    res = problem.solve(solver, material)
    problem.plot_comparison(res, material)
    plt.show()

if __name__ == "__main__":
  run_with(thermal.FiniteDifferenceImplicitThermalSolver(), 
      materials.ConstantThermalMaterial("Test", 10.0, 5.0))
