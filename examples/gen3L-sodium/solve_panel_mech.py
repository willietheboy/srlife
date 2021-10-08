#!/usr/bin/env python3

from os import sep, path

import numpy as np
from math import ceil, floor

import sys
sys.path.append('../..')
from srlife import receiver, solverparams, library, thermal, \
  structural, system, damage, managers

# Functions in this directory:
import preprocessors as pre
import postprocessors as post
from printers import *

def sample_parameters():
  params = solverparams.ParameterSet()

  params["nthreads"] = 1
  params["progress_bars"] = True
  # If true store results on disk (slower, but less memory)
  # -> has little impact on short simulation (<10 cycles) memory usage
  params["page_results"] = False

  params["thermal"]["rtol"] = 1.0e-6
  params["thermal"]["atol"] = 1.0e-8
  params["thermal"]["miter"] = 50
  params["thermal"]["substep"] = 40
  params["thermal"]["steady"] = False

  params["structural"]["rtol"] = 1.0e-6
  params["structural"]["atol"] = 1.0e-8
  params["structural"]["miter"] = 50
  params["structural"]["verbose"] = False

  params["system"]["rtol"] = 1.0e-3
  params["system"]["atol"] = 1.0e-4
  params["system"]["miter"] = 300
  params["system"]["verbose"] = False

  # How to extrapolate damage forward in time based on the cycles provided
  # Options:
  #     "lump" = D_future = sum(D_simulated) / N * days
  #     "last" = D_future = sum(D_simulated[:-1]) + D_simulated[-1] * days
  #     "poly" = polynomial extrapolation with order given by the "order" param
  params["damage"]["extrapolate"] = "last"
  #params["damage"]["order"] = 2

  return params

def main(thepanel, dim, defomat):

  # Load the receiver/panel previously saved
  headerprint(' SOLVE PANEL {} '.format(thepanel))
  thername = 'panel{}-thermodel.hdf5'.format(thepanel)
  thermodel = receiver.Receiver.load(thername)
  strprint('Thermal file', thername)
  strprint('Thermal panels', thermodel.npanels)
  strprint('Thermal tubes', thermodel.ntubes)
  strprint('Thermal cycles', thermodel.days)
  mechname = 'panel{}-mechmodel.hdf5'.format(thepanel)
  mechmodel = receiver.Receiver.load(mechname)
  strprint('Mechanical file', mechname)
  strprint('Mechanical panels', mechmodel.npanels)
  strprint('Mechanical tubes', mechmodel.ntubes)
  strprint('Mechanical cycles', mechmodel.days)
  if (thermodel.npanels != mechmodel.npanels) and \
     (thermodel.ntubes != mechmodel.ntubes):
    raise ValueError("Thermal and mechanical metric don't match")

  ## Liquid sodium in 60OD, 1.2WT -> "base": 3m/s, "lowflow": 1.5 m/s
  fluid_mat = library.load_fluid("sodium", "lowflow")
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
  thersolve = managers.SolutionManager(
    thermodel, thermal_solver, thermal_mat, fluid_mat,
    structural_solver, deformation_mat, damage_mat,
    system_solver, damage_model, pset = params
  )
  mechsolve = managers.SolutionManager(
    mechmodel, thermal_solver, thermal_mat, fluid_mat,
    structural_solver, deformation_mat, damage_mat,
    system_solver, damage_model, pset = params
  )

  ## Use axial points of maximum temperature (e.g. from 3D thermal):
  zenMaxTemp = {
    'panel0': 5750.0,
    'panel1': 5250.0,
    'panel2': 9250.0,
    'panel3': 4750.0,
    'panel4': 9750.0,
    'panel5': 4250.0,
    'panel6': 4250.0,
    'panel7': 9750.0,
    'panel8': 4750.0,
    'panel9': 9250.0,
    'panel10': 5250.0,
    'panel11': 5750.0
  }
  ## Locations of maximum creep damage
  zenMaxDc = {
    'panel0': 8250.0,
    'panel1': 5750.0,
    'panel2': 8750.0,
    'panel3': 4750.0,
    'panel4': 9250.0,
    'panel5': 4750.0,
    'panel6': 4750.0,
    'panel7': 9250.0,
    'panel8': 4750.0,
    'panel9': 8750.0,
    'panel10': 5750.0,
    'panel11': 7750.0
  }

  if dim == '2D':
    ## Reduce problem to 2D-GPS:
    headerprint(" REDUCE TO 2D ", ' ')
    for pi, panel in thermodel.panels.items():
      for ti, tube in panel.tubes.items():
        valprint(ti, zenMaxDc[pi], 'mm')
        tube.make_2D(zenMaxDc[pi])
    for pi, panel in mechmodel.panels.items():
      for ti, tube in panel.tubes.items():
        tube.make_2D(zenMaxDc[pi])

  ## thermal solve:
  headerprint(' THERMAL SOLVE ', ' ')
  thersolve.solve_heat_transfer()

  ## Repeat single-cycle thermal results, copy to mechanical:
  for pi in thermodel.panels.keys():
    for ti in thermodel.panels[pi].tubes.keys():
      thertube = thermodel.panels[pi].tubes[ti]
      mechtube = mechmodel.panels[pi].tubes[ti]
      mechtemp = pre.copy_temperature(
        thertube, thermodel, mechtube, mechmodel
      )
      mechtube.add_results('temperature', mechtemp)

  ## Heuristics (set temperature to T0 at cycle boundaries)
  #mechsolve.add_heuristic(managers.CycleResetHeuristic())

  ## 2D structural:
  headerprint(' MECHANICAL SOLVE ', ' ')
  mechsolve.solve_structural()

  # Save the tube data for structural visualization and report tube lifetime
  if not path.exists('vtu'):
    os.makedirs('vtu')
  headerprint(' LIFETIME ', ' ')
  life = {}; Dc = {}; Df = {}
  for pi, panel in mechmodel.panels.items():
    headerprint(' '+pi+' ', ' ')
    strprint('Tube stiffness', panel.stiffness)
    for ti, tube in panel.tubes.items():
      tube.add_quadrature_results('vonmises', post.eff_stress(tube))
      #tube.add_quadrature_results('meeq', post.eff_strain(tube))
      tube.add_quadrature_results(
        'cumDc', post.cumulative_creep_damage(tube, damage_mat)
      )
      tube.add_quadrature_results('hbbt1413', post.eq_strain_range(tube))
      tube.write_vtk(
        "vtu" + sep + "%s-%s-%s-%s-%s" % (pi, ti, dim, defomat, mechmodel.days)
      )
      # creep and fatigue damage accumulated each cycle and estimated life:
      Dc[ti], Df[ti], life[ti] = post.creep_fatigue(
        damage_model, tube, damage_mat, mechmodel, fmult
      )
      headerprint(' '+ti+' ', ' ')
      valeprint('First cycle creep damage', Dc[ti][0])
      valeprint('First cycle fatigue damage', Df[ti][0])
      valeprint('Last cycle creep damage', Dc[ti][-1])
      valeprint('Last cycle fatigue damage', Df[ti][-1])
      valprint('Est. cycles to fail', life[ti])
  headerprint('')

  ## Save mechanical model to HDF5 file:
  mechmodel.save("panel{}-resu{}-{}-N{}.hdf5".format(
    thepanel, dim, defomat, int(mechmodel.days)
  ))
  ## Return mechmodel too in case of solver failure (debug):
  return Dc, Df, life

if __name__ == "__main__":
  """
  Units: stress in MPa, strain in mm/mm, time in hours, temperature in K
  """
  thepanel = 1
  dim = '2D'
  # Defomation model: "elastic_model|elastic_constant|elastic_creep|base":
  defo = "elastic_model"
  Dc, Df, life = main(thepanel, dim, defo)
