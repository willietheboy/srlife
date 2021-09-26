#!/usr/bin/env python3

from os import sep

import numpy as np
from math import ceil, floor

import sys
sys.path.append('../..')
from srlife import receiver, solverparams, library, thermal, \
  structural, system, damage, managers

from printers import *

def vmStress(tube):
  """
  Calculate von Mises effective stress

  Parameters:
    tube        single tube with full results
  """
  vm = np.sqrt((
    (tube.quadrature_results['stress_xx'] -
     tube.quadrature_results['stress_yy'])**2.0 +
    (tube.quadrature_results['stress_yy'] -
     tube.quadrature_results['stress_zz'])**2.0 +
    (tube.quadrature_results['stress_zz'] -
     tube.quadrature_results['stress_xx'])**2.0 +
    6.0 * (tube.quadrature_results['stress_xy']**2.0 +
           tube.quadrature_results['stress_yz']**2.0 +
           tube.quadrature_results['stress_xz']**2.0))/2.0
  )
  return vm

def effStrain(tube):
  """
  Calculate effective strain

  Parameters:
    tube        single tube with full results
  """
  ee = np.sqrt(
    2.0/3.0 * (
      tube.quadrature_results['mechanical_strain_xx']**2 +
      tube.quadrature_results['mechanical_strain_yy']**2 +
      tube.quadrature_results['mechanical_strain_zz']**2
    ) +
    4.0/3.0 * (
      tube.quadrature_results['mechanical_strain_xy']**2 +
      tube.quadrature_results['mechanical_strain_xy']**2 +
      tube.quadrature_results['mechanical_strain_yz']**2
    )
  )
  return ee

def creep_fatigue(dmodel, tube, material, receiver):
  """
    Calculate the single-tube number of repetitions to failure

    Parameters:
      tube        single tube with full results
      dmodel      damage material model
      receiver    receiver, for metadata
  """
  # Material point cycle creep damage
  Dc = dmodel.creep_damage(tube, material, receiver)

  # Material point cycle fatigue damage
  Df = dmodel.fatigue_damage(tube, material, receiver)

  nc = receiver.days

  # This is going to be expensive, but I don't see much way around it
  return np.max(Dc.reshape(nc,-1).T, axis=0), \
    np.max(Df.reshape(nc,-1).T, axis=0), \
    min(dmodel.calculate_max_cycles(
      dmodel.make_extrapolate(c),
      dmodel.make_extrapolate(f), material
    ) for c,f in zip(Dc.reshape(nc,-1).T, Df.reshape(nc,-1).T))

def cumulative_creep_damage(tube, material):
  """
  Calculate cumulative cycle creep damage (e.g. for paraview)

  Parameters:
    tube        single tube with full results
    material    damage material model
  """
  tR = material.time_to_rupture(
    "averageRupture",
    tube.quadrature_results['temperature'],
    tube.quadrature_results['vonmises']
  )
  dts = np.diff(tube.times)
  dts = np.insert(dts, 0, 0.0)
  time_dmg = dts[:,np.newaxis,np.newaxis]/tR[:]
  cum_dmg = np.cumsum(time_dmg, axis=0)
  return cum_dmg

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

def main(thepanel):

  ## Load rheology models:
  mat = "740H"
  thermat = "base"
  defomat = "elastic_creep" # elastic_model|elastic_creep|base
  damat = "base"
  thermal_mat, deformation_mat, damage_mat = library.load_material(
    mat,
    thermat,
    defomat,
    damat
  )

  headerprint(' POST-PROCESS PANEL {} '.format(thepanel))
  filename = "panel{}-resu-2D-{}.hdf5".format(thepanel, defomat)
  model = receiver.Receiver.load(filename)
  strprint('File', filename)
  strprint('Panels', model.npanels)
  strprint('Tubes', model.ntubes)
  strprint('Cycles', model.days)
  strprint('Material', mat)
  strprint('Rheology', defomat)

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
      Dc[ti], Df[ti], life[ti] = creep_fatigue(
        damage_model, tube, damage_mat, model
      )
      headerprint(' '+ti+' ', ' ')
      valeprint('First cycle creep damage', Dc[ti][0])
      valeprint('First cycle fatigue damage', Df[ti][0])
      valeprint('Last cycle creep damage', Dc[ti][-1])
      valeprint('Last cycle fatigue damage', Df[ti][-1])
      valprint('Est. cycles to fail', life[ti])
  headerprint('')

  return Dc, Df, life

if __name__ == "__main__":
  """
  Units: stress in MPa, strain in mm/mm, time in hours, temperature in K
  """
  thepanel = 1
  Dc, Df, life = main(thepanel)
