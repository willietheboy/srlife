# -*- python-indent-offset: 2 -*-
#!/usr/bin/env python3
from os import sep
import numpy as np

# Functions in this directory:
import preprocessors as pre
from printers import *

import sys
sys.path.append('../..')

from srlife import receiver, library

def main(thepanel, ncycles):

  headerprint(' SETUP PANEL {} '.format(thepanel))

  ############################## RECEIVER METRICS ##############################
  ## One "day" consists of 4 cycles representative of each season

  ndays = 4 ## summer, equinox, winter and equinox again
  days = ['summer', 'equinox', 'winter', 'equinox']
  period_day = 12 ## hours
  period_cycle = period_day * ndays
  panel_stiffness = "disconnect" # "rigid"|"disconnected"|spring_coeff
  tube_stiffness = "disconnect"

  ## Create two models; (single cycle) thermal, and ncycles for structural:
  thermodel = receiver.Receiver(period_cycle, 1, panel_stiffness)
  mechmodel = receiver.Receiver(period_cycle, ncycles, panel_stiffness)
  thername = 'panel{}-thermodel.hdf5'.format(thepanel)
  mechname = 'panel{}-mechmodel.hdf5'.format(thepanel)

  ## Solar Central Receiver (scr) geometry:
  height = 14500.0 # mm
  width = 13500.0 # diameter of receiver in mm
  r_scr = width / 2. # radius of receiver
  c_scr = 2 * np.pi * r_scr # scr circumference on which tubes are placed
  n_panels = 12

  ## Tube geometry:
  ro_tube = 60/2. # mm
  wt_tube = 1.2 # mm

  ## Tube discretization:
  nr = 9
  nt = 90
  nz = 30

  ########################## DESIGN CYCLE CONSTRUCTION #########################
  ## Create "function tablecloths" (interpolators) for Design Cycle definition
  ## using design points mapped to times-of-day:

  ## Single season cycle times marking changes in functions:
  ftime = {
    'summer':  np.array(
      [0,0.1,0.3,1,2,3,4,5,6,7,8,9,10,11,11.7,11.9,12]
    ),
    'equinox': np.array(
      [0,1,1.1,1.3,2,3,4,5,6,7,8,9,10,10.7,10.9,11,12]
    ),
    'winter':  np.array(
      [0,  2,2.1,2.3,3,4,5,6,7,8,9,9.7,9.9,10,     12]
    )
  }

  ## Default simulation timestepping over function points:
  nr0 = 3  # steps ramping pressure (over 0.1 hr)
  nr1 = 6  # steps ramping thermal (over 0.2 hr)
  nr2 = 6  # rest of morning/afternoon ramp (over 0.7 hr)
  nr3 = 3  # each hour between design points
  fsteps = {
    'summer':  np.array(
      [nr0,nr1,nr2,nr3,nr3,nr3,nr3,nr3,nr3,nr3,nr3,nr3,nr3,nr2,nr1,nr0]
    ),
    'equinox': np.array(
      [3,  nr0,nr1,nr2,nr3,nr3,nr3,nr3,nr3,nr3,nr3,nr3,nr2,nr1,nr0,  3]
    ),
    'winter':  np.array(
      [5,      nr0,nr1,nr2,nr3,nr3,nr3,nr3,nr3,nr3,nr2,nr1,nr0,      5]
    )
  }

  ## Pressure switch:
  fpres = {
    'summer':  np.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]),
    'equinox': np.array([0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]),
    'winter':  np.array([  0,0,1,1,1,1,1,1,1,1,1,1,1,0,0  ])
  }

  ## Mapping design points to seasonal cycles (symmetrical about noon=6)
  fmap = {
    'summer': [None,None,'+6','+5','+4','+3','+2','+1','noon',
               '+1','+2','+3','+4','+5','+6',None,None],
    'equinox': [None,None,None,'+5','+4','+3','+2','+1','noon',
                '+1','+2','+3','+4','+5',None,None,None],
    'winter': [None,None,None,'+4','+3','+2','+1','noon',
               '+1','+2','+3','+4',None,None,None]
  }

  ## Create simulation time and related functions
  thertimes, thersimtime, theronoff, thermapper = pre.make_simtime(
    1, days, ftime, fsteps, fpres, fmap, period_cycle, period_day
  )
  mechtimes, mechsimtime, mechonoff, mechmapper = pre.make_simtime(
    ncycles, days, ftime, fsteps, fpres, fmap, period_cycle, period_day
  )

  ############################## THERMOHYDRAULICS ##############################
  ## Load receiver design point conditions (Daggett, CA):
  res_panel = 10 # 10 lateral points per panel
  res_zen = 30 # 30 elements over receiver height
  design_pts = ['summer_noon','summer_+1','summer_+2',
                'summer_+3','summer_+4','summer_+5','summer_+6',
                'equinox_noon','equinox_+1','equinox_+2',
                'equinox_+3','equinox_+4','equinox_+5',
                'winter_noon','winter_+1','winter_+2',
                'winter_+3','winter_+4']
  ## Design points are located in the directory "thermohydraulics":
  dirname = "thermohydraulics"
  fptemp, fpvel, fpflux = pre.load_design_points(design_pts, dirname)

  ## Create mesh for interpolating flux and fluid temperatures at discretisation
  ## of design point information:
  azi_pts = np.linspace(0, 2*np.pi/n_panels, res_panel+1) + \
    (thepanel * 2*np.pi/n_panels)
  azi_surf = (azi_pts[:-1] + azi_pts[1:]) / 2. # tubes around receiver circumference
  zen_pts = np.linspace(0, height, res_zen)
  zen_surf = (zen_pts[:-1] + zen_pts[1:]) / 2. # flux/temp values also at surfaces
  ma, mz = np.meshgrid(azi_surf, zen_surf, indexing='ij')

  ## Make tablecloths of fluid temperature, velocity and net flux:
  ftemp, fvel, fflux, = pre.make_tablecoths(
    thertimes, azi_surf, zen_surf, thermapper,
    fpflux, fptemp, fpvel, period_day, 6
  )
  ## Make tablecloth of heat transfer coefficient (from temperature and velocity):
  fhtc = pre.make_tablecloth_htc(
    thertimes, ro_tube-wt_tube, ro_tube,
    azi_surf, zen_surf, ftemp, fvel
  )

  ################################# HEURISTICS #################################
  ## Results from setup_receiver.py heuristics

  ## Location of maximum crown thermal strain:
  aziMaxSig = {
    'panel0': 0.39269908169872414,
    'panel1': 0.7592182246175333,
    'panel2': 1.335176877775662,
    'panel3': 1.806415775814131,
    'panel4': 2.2776546738525996,
    'panel5': 2.853613327010729,
    'panel6': 3.4295719801688573,
    'panel7': 4.005530633326986,
    'panel8': 4.476769531365455,
    'panel9': 4.948008429403924,
    'panel10': 5.523967082562052,
    'panel11': 5.890486225480862
  }
  ## Location of lowest crown thermal strain (at same time and height as maximum):
  aziMinSig = {
    'panel0': 0.02617993877991494,
    'panel1': 0.5497787143782138,
    'panel2': 1.0733774899765125,
    'panel3': 2.0682151636132806,
    'panel4': 2.591813939211579,
    'panel5': 3.115412714809878,
    'panel6': 3.167772592369708,
    'panel7': 3.6913713679680065,
    'panel8': 4.214970143566306,
    'panel9': 5.209807817203073,
    'panel10': 5.733406592801372,
    'panel11': 6.257005368399671
  }
  ## Location of peak crown temperature during cycle
  aziMaxTemp = {
    'panel0': 0.34033920413889424,
    'panel1': 0.9686577348568528,
    'panel2': 1.4922565104551517,
    'panel3': 1.9111355309337907,
    'panel4': 2.4870941840919194,
    'panel5': 2.9583330821303884,
    'panel6': 3.324852225049198,
    'panel7': 3.796091123087667,
    'panel8': 4.372049776245795,
    'panel9': 4.790928796724434,
    'panel10': 5.314527572322733,
    'panel11': 5.942846103040692
  }
  ## Location of maximum cumulative creep damage:
  aziMaxDc = {
    'panel0': 0.445058959258554,
    'panel1': 0.9686577348568528,
    'panel2': 1.335176877775662,
    'panel3': 1.806415775814131,
    'panel4': 2.434734306532089,
    'panel5': 2.905973204570558,
    'panel6': 3.377212102609027,
    'panel7': 3.848451000647496,
    'panel8': 4.476769531365455,
    'panel9': 4.948008429403924,
    'panel10': 5.314527572322733,
    'panel11': 5.890486225480862
  }

  ################################### SRLIFE ###################################

  ## Tube circumferential flux component (cosine distribution):
  cos_theta = lambda theta: np.maximum(0,np.cos(theta))

  ## Flux with time and location on receiver
  flux_time = lambda t, theta, a, z: cos_theta(theta) * \
    fflux(np.array([t, a, z]))

  ## ID fluid temperature histories for each tube
  T_ref = 293.15
  fluid_temp_time = lambda t, a, z: ftemp(np.array([t, a, z]))

  ## ID pressure history (mechanical model)
  p_max = 1.5 # MPa
  pressure = lambda t, times, onoff: p_max * np.interp(t, times, onoff)

  ## A mesh over the times and height (for the fluid temperatures)
  time_h, z_h = np.meshgrid(
    thersimtime, zen_pts, indexing='ij'
  )
  ## A surface mesh over the outer surface (for the flux)
  time_s, theta_s, z_s = np.meshgrid(
    thersimtime, np.linspace(0,2*np.pi,nt+1)[:nt],
    zen_pts, indexing = 'ij'
  )

  # ## Setup two tubes per panel at locations of maximum/minimum thermal strain:
  # azitubes = [aziMaxSig, aziMinSig]
  # labels = ['maxSig', 'minSig']
  # ## Setup one tube at location of maximum estimated crown temperature
  # azitubes = [aziMaxTemp]
  # labels = ['maxTemp']
  ## Setup one tube at location of maximum cumulative creep damage
  azitubes = [aziMaxDc]
  labels = ['maxDc']
  therpanel = receiver.Panel(tube_stiffness)
  mechpanel = receiver.Panel(tube_stiffness)
  thertubes = [None]*len(azitubes)
  mechtubes = [None]*len(azitubes)
  for i, azi in enumerate(azitubes):
    # Setup each tube in turn and assign it to the correct panel
    pi = 'panel{}'.format(thepanel)
    strprint('Setting up thermal BCs', labels[i])
    thertubes[i] = receiver.Tube(ro_tube, wt_tube, height, nr, nt, nz, T0 = T_ref)
    mechtubes[i] = receiver.Tube(ro_tube, wt_tube, height, nr, nt, nz, T0 = T_ref)
    thertubes[i].set_times(thersimtime)
    mechtubes[i].set_times(mechsimtime)
    thertubes[i].set_bc(
      receiver.ConvectiveBC(
        ro_tube-wt_tube, height, nz, thersimtime,
        fluid_temp_time(time_h, azi[pi], z_h)
      ), "inner"
    )
    thertubes[i].set_bc(
      receiver.HeatFluxBC(
        ro_tube, height, nt, nz, thersimtime,
        flux_time(time_s, theta_s, azi[pi], z_s)
      ), "outer"
    )
    mechtubes[i].set_pressure_bc(
      receiver.PressureBC(
        mechsimtime, pressure(mechsimtime, mechtimes, mechonoff)
      )
    )
    ## Assign tubes to panels
    therpanel.add_tube(thertubes[i], labels[i])
    mechpanel.add_tube(mechtubes[i], labels[i])

  ## Add panels to model:
  thermodel.add_panel(therpanel, 'panel{}'.format(thepanel))
  mechmodel.add_panel(mechpanel, 'panel{}'.format(thepanel))

  ## Save the receiver to an HDF5 file
  strprint('Model saved', thername)
  thermodel.save(thername)
  strprint('Model saved', mechname)
  mechmodel.save(mechname)
  headerprint('')

if __name__ == '__main__':
  """
  Units: stress in MPa, strain in mm/mm, time in hours, temperature in K
  """
  thepanel = 0
  ncycles = 15
  main(thepanel, ncycles)
