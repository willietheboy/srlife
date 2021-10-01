#!/usr/bin/env python3

from os import sep

import numpy as np
from math import ceil, floor

import sys
sys.path.append('../..')
from srlife import receiver, solverparams, library, thermal, \
  structural, system, damage, managers

# Functions in this directory:
#import preprocessors as pre
import postprocessors as post
from printers import *

def sample_parameters():
  params = solverparams.ParameterSet()

  # How to extrapolate damage forward in time based on the cycles provided
  # Options:
  #     "lump" = D_future = sum(D_simulated) / N * days
  #     "last" = D_future = sum(D_simulated[:-1]) + D_simulated[-1] * days
  #     "poly" = polynomial extrapolation with order given by the "order" param
  params["damage"]["extrapolate"] = "last"
  #params["damage"]["order"] = 2

  return params

def main(thepanel, dim, defomat, ndays):

  ## Load rheology models:
  mat = "740H"
  thermat = "base"
  damat = "base"
  thermal_mat, deformation_mat, damage_mat = library.load_material(
    mat,
    thermat,
    defomat,
    damat
  )

  headerprint(' POST-PROCESS PANEL {} '.format(thepanel))
  filename = "panel{}-resu{}-{}-N{}.hdf5".format(thepanel, dim, defomat, ndays)
  model = receiver.Receiver.load(filename)
  strprint('File', filename)
  strprint('Panels', model.npanels)
  strprint('Tubes', model.ntubes)
  strprint('Cycles', model.days)
  strprint('Material', mat)
  strprint('Rheology', defomat)

  ## Multiply the fatigue damage by the number of days/clouds per cycle
  ## -> 4 days per cycle
  ## -> 5 cloud transitions per day
  fmult = 4 * 5

  # Load some customized solution parameters
  # These are all optional, all the solvers have default values
  # for parameters not provided by the user
  params = sample_parameters()

  # Damage model to use in calculating life
  damage_model = damage.TimeFractionInteractionDamage(params["damage"])

  # Save the tube data for structural visualization and report tube lifetime
  headerprint(' LIFETIME ', ' ')
  life = {}; Dc = {}; Df = {}
  for pi, panel in model.panels.items():
    headerprint(' '+pi+' ', ' ')
    strprint('Tube stiffness', panel.stiffness)
    for ti, tube in panel.tubes.items():
      # creep and fatigue damage accumulated each cycle and estimated life:
      Dc[ti], Df[ti], life[ti] = post.creep_fatigue(
        damage_model, tube, damage_mat, model, fmult
      )
      strprint('Tube location/ID', ti)
      valeprint('First cycle creep damage', Dc[ti][0])
      valeprint('First cycle fatigue damage', Df[ti][0])
      valeprint('Last cycle creep damage', Dc[ti][-1])
      valeprint('Last cycle fatigue damage', Df[ti][-1])
      valprint('Est. cycles to fail', life[ti])

  return Dc, Df, life

if __name__ == "__main__":

  """
  Post-process analysis stored in HDF5 files

  Usage: post_panels.py <first_panel> <last_panel> <ncycles>

  Where:
      first_panel    Index of first panel in HDF5 file set [0:11]
      last_panel     Index of last panel in HDF5 file set [1:12]
      ncycles        Number of cycles simulated in HDF5 file
  """

  if len(sys.argv) < 4 or len(sys.argv) > 4:
    RuntimeError("Usage: post_panels.py <first_panel> <last_panel> <ncycles>")
  first_panel = int(sys.argv[1])
  last_panel = int(sys.argv[2])
  ncycles = int(sys.argv[3])
  dim = '2D'
  defo = 'elastic_creep'
  Dc = {}; Df = {}; life = {}; model = {}
  for i in range(first_panel, last_panel):
      pi = 'panel{}'.format(i)
      Dc[pi] = {}; Df[pi] = {}; life[pi] = {}
      Dc[pi], Df[pi], life[pi] = main(i, dim, defo, ncycles)

  ## Plot extrapolation of creep damage to 2750 cycles (11e3 days):
  headerprint(' EXTRAPOLATE LIFETIMES ')
  post.plot_cycle_cdamage(
    Dc, ncycles, tubeid='maxDc',
    filename='panels{}-{}-resu{}-{}-N{}_dDc.pdf'.format(
      first_panel, last_panel-1, dim, defo, ncycles
    ),
    verbose=True
  )

  ## Check equivalent strain range (through fatigue damage evolution):
  # aster = {
  #   'panel0': 5.236726095376204e-07,
  #   'panel1': 7.422008696634635e-07,
  #   'panel2': 8.669093048693842e-07,
  #   'panel3': 3.0121057902408216e-06,
  #   'panel4': 8.593519726043566e-07,
  #   'panel5': 2.890085074478858e-06
  # }
  post.plot_cycle_fdamage(
    Df, 'maxDc', 'panels{}-{}-resu{}-{}-N{}_dDf.pdf'.format(
      first_panel, last_panel-1, dim, defo, ncycles
    )
  )
