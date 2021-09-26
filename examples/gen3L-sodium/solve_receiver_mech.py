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

def lds_damage(dmodel, tube, material, receiver):
  """
    Calculate the single-tube number of repetitions to failure
    ---! Modified version to return additional info !---

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
  Calculate cumulative cycle creep damage (e.g. paraview)

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

def creep_damage(tube, material):
  """
  Calculate creep damage for all points

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
  return time_dmg

def sample_parameters():
  params = solverparams.ParameterSet()

  params["nthreads"] = 2
  params["progress_bars"] = True
  # If true store results on disk (slower, but less memory)
  # -> has little impact on short simulation memory usage
  params["page_results"] = False

  params["thermal"]["steady"] = False
  params["thermal"]["rtol"] = 1.0e-6
  params["thermal"]["atol"] = 1.0e-4
  params["thermal"]["miter"] = 20
  params["thermal"]["substep"] = 100

  params["structural"]["rtol"] = 1.0e-6
  params["structural"]["atol"] = 1.0e-8
  params["structural"]["miter"] = 50
  params["structural"]["verbose"] = False

  params["system"]["rtol"] = 1.0e-6
  params["system"]["atol"] = 1.0e-8
  params["system"]["miter"] = 20
  params["system"]["verbose"] = False

  # How to extrapolate damage forward in time based on the cycles provided
  # Options:
  #     "lump" = D_future = sum(D_simulated) / N * days
  #     "last" = D_future = sum(D_simulated[:-1]) + D_simulated[-1] * days
  #     "poly" = polynomial extrapolation with order given by the "order" param
  params["damage"]["extrapolate"] = "last"
  #params["damage"]["order"] = 2

  return params

def main(thepanel):

  # Load the receiver/panel previously saved
  # modname = 'receiver.hdf5'
  #thepanel = 1
  modname = 'panel{}.hdf5'.format(thepanel)
  model = receiver.Receiver.load(modname)
  headerprint(' LOADED MODEL: {} '.format(modname), '=')
  strprint('Panels', model.npanels)
  strprint('Tubes', model.ntubes)

  ## Liquid sodium in 60OD, 1.2WT -> "base": 3m/s, "lowflow": 1.5 m/s
  fluid_mat = library.load_fluid("sodium", "base")
  ## Load rheology models:
  thermat = "base"
  defomat = "elastic_creep" # elastic_model|elastic_creep|base
  damat = "base"
  thermal_mat, deformation_mat, damage_mat = library.load_material(
    "740H",
    thermat,
    defomat,
    damat
  )
  strprint('Rheology', defomat)

  # Load some customized solution parameters
  # These are all optional, all the solvers have default values
  # for parameters not provided by the user
  params = sample_parameters()

  # Define the thermal solver to use in solving the heat transfer problem
  thermal_solver = thermal.FiniteDifferenceImplicitThermalSolver(
      params["thermal"])

  # Define the structural solver to use in solving the individual tube problems
  structural_solver = structural.PythonTubeSolver(params["structural"])
  # Define the system solver to use in solving the coupled structural system
  system_solver = system.SpringSystemSolver(params["system"])
  # Damage model to use in calculating life
  damage_model = damage.TimeFractionInteractionDamage(params["damage"])

  # The solution manager
  solver = managers.SolutionManager(model, thermal_solver, thermal_mat, fluid_mat,
      structural_solver, deformation_mat, damage_mat,
      system_solver, damage_model, pset = params)

  ## Use axial points of maximum temperature from 3D thermal:
  zTmax = {
    'panel0': 6500.0,
    'panel1': 5500.0,
    'panel2': 9000.0,
    'panel3': 4500.0,
    'panel4': 9500.0,
    'panel5': 4500.0,
    'panel6': 4500.0,
    'panel7': 9500.0,
    'panel8': 4500.0,
    'panel9': 9000.0,
    'panel10': 5500.0,
    'panel11': 6500.0
  }

  ## Reduce problem to 2D-GPS:
  headerprint(" GENERALISED PLANE STRAIN ", ' ')
  for pi, panel in model.panels.items():
    for ti, tube in panel.tubes.items():
      valprint(ti, zTmax[pi], 'mm')
      tube.make_2D(zTmax[pi])

  # Heuristics (reset temperature to T0 each cycle)
  #solver.add_heuristic(managers.CycleResetHeuristic())

  ## 2D thermal and structural:
  solver.solve_heat_transfer()
  solver.solve_structural()

  # Save the tube data for structural visualization and report tube lifetime
  headerprint(' LIFETIME ')
  life = {}; Dc = {}; Df = {}
  for pi, panel in model.panels.items():
    headerprint(' '+pi+' ', ' ')
    for ti, tube in panel.tubes.items():
      tube.add_quadrature_results('vonmises', vmStress(tube))
      tube.add_quadrature_results('meeq', effStrain(tube))
      tube.add_quadrature_results(
        'cumDc', cumulative_creep_damage(tube, damage_mat)
      )
      tube.write_vtk("resu"+sep+"2D-%s-%s-%s" % (defomat, pi, ti))
      # creep and fatigue damage accumulated each cycle and estimated life:
      Dc[ti], Df[ti], life[ti] = lds_damage(
        damage_model, tube, damage_mat, model
      )
      headerprint(' '+ti+' ', ' ')
      valeprint('Last cycle creep damage', np.max(Dc[ti][:,-1]))
      valeprint('Last cycle fatigue damage', np.max(Df[ti][:,-1]))
      valprint('Est. cycles to fail', life[ti])

  ## Save complete model (including results) to HDF5 file:
  #model.save("3D-%s-results-model.hdf5" % tmode)
  return Dc, Df, life

if __name__ == "__main__":
  """
  Units: stress in MPa, strain in mm/mm, time in hours, temperature in K
  """
  thepanel = 1
  main(thepanel)
