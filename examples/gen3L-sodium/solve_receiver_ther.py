#!/usr/bin/env python3

import sys
sys.path.append('../..')
from os import sep

import numpy as np

from srlife import receiver, solverparams, library, thermal, \
  structural, system, damage, managers

from printers import *

def sample_parameters():
  params = solverparams.ParameterSet()

  params["nthreads"] = 3
  params["progress_bars"] = True
  # If true store results on disk (slower, but less memory)
  params["page_results"] = False

  params["thermal"]["steady"] = False
  params["thermal"]["rtol"] = 1.0e-6
  params["thermal"]["atol"] = 1.0e-6
  params["thermal"]["miter"] = 50
  params["thermal"]["substep"] = 40

  params["structural"]["rtol"] = 1.0e-6
  params["structural"]["atol"] = 1.0e-8
  params["structural"]["miter"] = 50
  params["structural"]["verbose"] = False

  params["system"]["rtol"] = 1.0e-6
  params["system"]["atol"] = 1.0e-8
  params["system"]["miter"] = 50
  params["system"]["verbose"] = False

  # How to extrapolate damage forward in time based on the cycles provided
  # Options:
  #     "lump" = D_future = sum(D_simulated) / N * days
  #     "last" = D_future = sum(D_simulated[:-1]) + D_simulated[-1] * days
  #     "poly" = polynomial extrapolation with order given by the "order" param
  params["damage"]["extrapolate"] = "last"
  params["damage"]["order"] = 2

  return params

if __name__ == "__main__":

  """
  Units: stress in MPa, strain in mm/mm, time in hours, temperature in K
  """

  # Load the receiver/panel previously saved
  headerprint(' SOLVE RECEIVER ', '=')
  dim = '3D'
  filename = 'receiver.hdf5'
  model = receiver.Receiver.load(filename)
  strprint('File', filename)
  strprint('Panels', model.npanels)
  strprint('Tubes', model.ntubes)
  strprint('Cycles', model.days)

  ## Information necessary for thermal post-processing:
  days = ['summer', 'equinox', 'winter', 'equinox']
  ndays = len(days)
  period_day = model.period/ndays
  noon = 6

  ## Liquid sodium in 60OD, 1.2WT -> "base": 3m/s, "lowflow": 1.5 m/s
  fluid_mat = library.load_fluid("sodium", "base")
  ## Load rheology models:
  mat = "740H"
  thermat = "base"
  defomat = "elastic_model" # elastic_model|elastic_creep|base
  damat = "base"
  thermal_mat, deformation_mat, damage_mat = library.load_material(
    mat,
    thermat,
    defomat,
    damat
  )
  strprint('Material', mat)
  strprint('Rheology', defomat)

  # Load some customized solution parameters
  # These are all optional, all the solvers have default values
  # for parameters not provided by the user
  params = sample_parameters()

  # Define the thermal solver to use in solving the heat transfer problem
  thermal_solver = thermal.FiniteDifferenceImplicitThermalSolver(
      params["thermal"])

  # ## Test only (single tube) thermal solutions:
  # for panel in model.panels.values():
  #   for tube in panel.tubes.values():
  #     thermal_solver.solver.solve(tube, thermal_mat, fluid_mat)

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
        # valprint(ti, zenMaxDc[pi], 'mm')
        # tube.make_2D(zenMaxDc[pi])
        valprint(ti, zenMaxTemp[pi], 'mm')
        tube.make_2D(zenMaxTemp[pi])
    for pi, panel in mechmodel.panels.items():
      for ti, tube in panel.tubes.items():
        # tube.make_2D(zenMaxDc[pi])
        tube.make_2D(zenMaxTemp[pi])

  ## Heuristics
  #solver.add_heuristic(managers.CycleResetHeuristic())

  ## thermal solve:
  headerprint(' THERMAL SOLVE ', ' ')
  solver.solve_heat_transfer()

  ## Post-process:
  zTmax = {}; Tmax = {}; fluid = {}; flux = {}; tmax = {}; peakseas = {}
  for pi, panel in model.panels.items():
    headerprint(' '+pi+' ', ' ')
    for ti, tube in panel.tubes.items():
      headerprint(' '+ti+' ', ' ')
      ## This is really slow -- option to select timerange?:
      #tube.write_vtk("vtu"+sep+"3D-thermal-%s-%s" % (pi, ti))
      _, _, z = tube.mesh
      times = tube.times
      sids=np.array(np.floor(
        (times % (ndays*period_day)) / period_day
      ), dtype='int')
      Tmax[ti] = np.max(tube.results['temperature'])
      loc_max = np.where(tube.results['temperature'] == Tmax[ti])
      tmax[ti] = times[loc_max[0][0]]
      peakseas[ti] = days[sids[loc_max[0][0]]]
      zTmax[ti] = z[loc_max[1:]][0]
      valprint('max. temp.', Tmax[ti]-273.15, 'degC')
      valprint('off-noon', (tmax[ti] % period_day)-noon, 'hr')
      strprint('season', peakseas[ti])
      valprint('height (z)', zTmax[ti], 'mm')
      fluid[ti] = tube.inner_bc.fluid_temperature(
        times[loc_max[0]], zTmax[ti]
      )[0]
      valprint('fluid temp.', fluid[ti]-273.15, 'degC')
      flux[ti] = tube.outer_bc.flux(times[loc_max[0]], 0, zTmax[ti])[0]
      valprint('flux', flux[ti], 'MW/m^2')
  # ## tex-table:
  # print(r'Tube & Design pt. & z (mm) & '+\
  #       'T_f (degC) & T_t (degC) & flux (MW/m^2)')
  # for pi, panel in model.panels.items():
  #   for ti, tube in panel.tubes.items():
  #     print(r'{} & {} & {} & {} & {} & {}\\'.format(
  #       ti.replace('tube',''),
  #       '{} {:+.0f}'.format(peakseas[ti],(tmax[ti] % period_day)-noon),
  #       zTmax[ti], fluid[ti]-273.15, Tmax[ti]-273.15, flux[ti]
  #     ))
  ## Save thermal results to HDF5 file:
  model.save("receiver-resu{}-thermal.hdf5".format(dim))
