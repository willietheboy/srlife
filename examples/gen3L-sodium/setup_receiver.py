# -*- python-indent-offset: 2 -*-
#!/usr/bin/env python3
from os import sep, path
import numpy as np

# Functions in this directory:
import preprocessors as pre
#import postprocessors as post
from printers import *

import sys
sys.path.append('../..')

from srlife import receiver, library

if __name__ == '__main__':
  """
  Units: stress in MPa, strain in mm/mm, time in hours, temperature in K
  """

  # Just grab the time_to_rupture function from srlife
  mat = library.load_damage("740H", "base")

  ############################## RECEIVER METRICS ##############################
  ## One "day" consists of 4 cycles representative of each season

  ndays = 4 ## summer, equinox, winter and equinox again
  days = ['summer', 'equinox', 'winter', 'equinox']
  noon = 6
  period_day = 12 ## hours
  period_cycle = period_day * ndays
  panel_stiffness = "disconnect" # "rigid"|"disconnected"|spring_coeff
  tube_stiffness = "disconnect"
  ncycles = 1 # repetition of Design Cycle (consisting of 4 days)
  model = receiver.Receiver(period_cycle, ncycles, panel_stiffness)
  modname = 'receiver.hdf5'

  ## Solar Central Receiver (scr) geometry:
  height = 14500.0 # mm
  width = 13500.0 # diameter of receiver in mm
  r_scr = width / 2. # radius of receiver
  c_scr = 2 * np.pi * r_scr # scr circumference on which tubes are placed
  n_panels = 12
  arc_panel = 2 * np.pi / n_panels

  ## Tube geometry:
  ro_tube = 60/2. # mm
  wt_tube = 1.2 # mm

  ## Tube discretization:
  nr = 9
  nt = 30
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
  times, simtime, onoff, mapper = pre.make_simtime(
    ncycles, days, ftime, fsteps, fpres, fmap, period_cycle, period_day
  )
  ## Numerical index of seasons (days)
  sids=np.array(np.floor(
    (times % (ndays*period_day)) / period_day
  ), dtype='int')

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
  azi_pts = np.linspace(0, 2*np.pi, (res_panel*n_panels)+1)
  azi_surf = (azi_pts[:-1] + azi_pts[1:]) / 2. # tubes around receiver circumference
  zen_pts = np.linspace(0, height, res_zen)
  zen_surf = (zen_pts[:-1] + zen_pts[1:]) / 2. # flux/temp values also at surfaces
  ma, mz = np.meshgrid(azi_surf, zen_surf, indexing='ij')

  ## Get tablecloths of fluid temperature, velocity and net flux:
  ftemp, fvel, fflux, = pre.make_tablecoths(
    times, azi_surf, zen_surf, mapper, fpflux, fptemp, fpvel, period_day, 6
  )
  ## Get tablecloth of heat transfer coefficient:
  fhtc = pre.make_tablecloth_htc(
    times, ro_tube-wt_tube, ro_tube,
    azi_surf, zen_surf, ftemp, fvel
  )

  ## Check to see if plot directory exists:
  if not path.exists(dirname + sep + 'plots'):
    os.makedirs(dirname + sep + 'plots')

  ## Summer:
  pre.plot_pcolor(
    ma, mz*1e-3, fflux(np.array([0.3,ma,mz])),
    r'\textsc{net flux density}, '+\
    r'$\vec{\phi}_\mathrm{q,net}$ (\si{\mega\watt\per\meter\squared})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'summer_-6_flux', compass=True
  )
  pre.plot_pcolor(
    ma, mz*1e-3, fflux(np.array([6,ma,mz])),
    r'\textsc{net flux density}, '+\
    r'$\vec{\phi}_\mathrm{q,net}$ (\si{\mega\watt\per\meter\squared})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'summer_noon_flux', compass=True
  )
  pre.plot_pcolor(
    ma, mz*1e-3, fflux(np.array([11.7,ma,mz])),
    r'\textsc{net flux density}, '+\
    r'$\vec{\phi}_\mathrm{q,net}$ (\si{\mega\watt\per\meter\squared})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'summer_+6_flux', compass=True
  )
  ## Equinox
  pre.plot_pcolor(
    ma, mz*1e-3, fflux(np.array([13.3,ma,mz])),
    r'\textsc{net flux density}, '+\
    r'$\vec{\phi}_\mathrm{q,net}$ (\si{\mega\watt\per\meter\squared})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'equinox_-5_flux', compass=True
  )
  pre.plot_pcolor(
    ma, mz*1e-3, fflux(np.array([18,ma,mz])),
    r'\textsc{net flux density}, '+\
    r'$\vec{\phi}_\mathrm{q,net}$ (\si{\mega\watt\per\meter\squared})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'equinox_noon_flux', compass=True
  )
  pre.plot_pcolor(
    ma, mz*1e-3, fflux(np.array([22.7,ma,mz])),
    r'\textsc{net flux density}, '+\
    r'$\vec{\phi}_\mathrm{q,net}$ (\si{\mega\watt\per\meter\squared})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'equinox_+5_flux', compass=True
  )
  pre.plot_pcolor(
    ma, mz*1e-3, fhtc(np.array([18,ma,mz])),
    r'\textsc{film coefficient}, '+\
    r'$h_\mathrm{f}$ (\si{\watt\per\milli\meter\per\kelvin})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'equinox_noon_htc', compass=True
  )
  pre.plot_pcolor(
    ma, mz*1e-3, ftemp(np.array([18, ma, mz]))-273.15,
    r'\textsc{fluid temperature}, $T_\mathrm{f}$ (\si{\celsius})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'equinox_noon_temp', compass=True
  )
  pre.plot_pcolor(
    ma, mz*1e-3, ftemp(np.array([22.7, ma, mz]))-273.15,
    r'\textsc{fluid temperature}, $T_\mathrm{f}$ (\si{\celsius})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'equinox_+5_temp', compass=True, cmap='inferno'
  )
  ## Winter
  pre.plot_pcolor(
    ma, mz*1e-3, fflux(np.array([26.3,ma,mz])),
    r'\textsc{net flux density}, '+\
    r'$\vec{\phi}_\mathrm{q,net}$ (\si{\mega\watt\per\meter\squared})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'winter_-4_flux', compass=True
  )
  pre.plot_pcolor(
    ma, mz*1e-3, fflux(np.array([30,ma,mz])),
    r'\textsc{net flux density}, '+\
    r'$\vec{\phi}_\mathrm{q,net}$ (\si{\mega\watt\per\meter\squared})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'winter_noon_flux', compass=True
  )
  pre.plot_pcolor(
    ma, mz*1e-3, fflux(np.array([33.7,ma,mz])),
    r'\textsc{net flux density}, '+\
    r'$\vec{\phi}_\mathrm{q,net}$ (\si{\mega\watt\per\meter\squared})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'winter_+4_flux', compass=True
  )

  ################################# HEURISTICS #################################
  ## Selection of critical azimuth/height in panels:
  tempCrown, sigCrown = pre.heuristics(
    times, ma, mz, ro_tube-wt_tube, ro_tube, ftemp, fhtc, fflux
  )
  ## Simple 1D crown temperature and stress intensity:
  cumDc = pre.cumulative_creep_damage(times, sigCrown, tempCrown, mat)
  pre.plot_pcolor(
    ma, mz*1e-3, cumDc[-1],
    r'\textsc{cycle creep damage}, '+\
    r'$\delta D_\mathrm{\scriptscriptstyle R}$',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'cumDc_receiver', compass=True, cmap='inferno', logNorm=True
  )
  pre.plot_pcolor(
    ma, mz*1e-3, np.max(tempCrown, axis=0)-273.15,
    r'\textsc{max. cycle crown temperature}, '+\
    r'$\overline{T}_\mathrm{c}$ (\si{\celsius})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'tempCrown_receiver', compass=True, cmap='inferno'
  )
  pre.plot_pcolor(
    ma, mz*1e-3, np.max(sigCrown, axis=0),
    r'\textsc{max. cycle crown stress}, '+\
    r'$\sigma_\mathrm{eq}$ (\si{\mega\pascal})',
    r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
    'sigCrown_receiver', compass=True,
  )

  ## Discover critical locations in receiver panels:
  maxTemp = {}; timeMaxTemp = {}; fluxMaxTemp = {}
  aziMaxTemp = {}; zenMaxTemp = {}
  maxSig = {}; timeMaxSig = {}; aziMaxSig = {}; aziMinSig = {}
  minSig = {}; zenMaxSig = {}; zenMinSig = {}
  maxDc = {}; aziMaxDc = {}; zenMaxDc = {}
  for i in range(n_panels):
    pi = 'panel{}'.format(i)
    pids = np.logical_and(
      azi_surf > i*arc_panel,
      azi_surf < i*arc_panel+arc_panel
    )
    headerprint(' HEURISTICS PANEL {} '.format(i), '=')
    ## maximum crown temperature
    maxTemp[pi] = np.max(tempCrown[:,pids])
    tids = np.where(tempCrown[:,pids]==maxTemp[pi])
    timeMaxTemp[pi] = '{} {:+.0f}'.format(
      days[sids[tids[0][0]]],(times[tids[0][0]] % period_day) - noon
    )
    aziMaxTemp[pi] = azi_surf[pids][tids[1][0]]
    zenMaxTemp[pi] = zen_surf[tids[2][0]]
    fluxMaxTemp[pi] = fflux(
      np.array([times[tids[0][0]],aziMaxTemp[pi],zenMaxTemp[pi]])
    )[0]
    valprint('max. crown temp.', maxTemp[pi]-273.15, 'degC')
    valprint('net flux', fluxMaxTemp[pi], 'MW/m2')
    valprint('at azimuth', aziMaxTemp[pi], 'rad.')
    valprint('at height', zenMaxTemp[pi], 'mm')
    ## maximum thermal strain:
    maxSig[pi] = np.max(sigCrown[:,pids])
    uids = np.where(sigCrown[:,pids]==maxSig[pi])
    timeMaxSig[pi] = '{} {:+.0f}'.format(
      days[sids[uids[0][0]]],(times[uids[0][0]] % period_day) - noon
    )
    aziMaxSig[pi] = azi_surf[pids][uids[1][0]]
    zenMaxSig[pi] = zen_surf[uids[2][0]]
    valprint('max. crown stress', maxSig[pi], 'MPa')
    valprint('at azimuth', aziMaxSig[pi], 'rad.')
    valprint('at height', zenMaxSig[pi], 'mm')
    ## minimum thermal strain (at same time and height as maximum):
    minSig[pi] = np.min(sigCrown[:,pids][uids[0],:,uids[2]])
    lids = np.where(sigCrown[:,pids]==minSig[pi])
    aziMinSig[pi] = azi_surf[pids][lids[1][0]]
    zenMinSig[pi] = zen_surf[lids[2][0]]
    valprint('min. peak crown stress', minSig[pi], 'MPa')
    valprint('at azimuth', aziMinSig[pi], 'rad.')
    valprint('at height', zenMinSig[pi], 'mm')
    ## cumulative creep damage (using simple 1d model):
    maxDc[pi] = np.max(cumDc[-1,pids])
    dids = np.where(cumDc[-1,pids]==maxDc[pi])
    aziMaxDc[pi] = azi_surf[pids][dids[0][0]]
    zenMaxDc[pi] = zen_surf[dids[1][0]]
    valeprint('max. creep damage', maxDc[pi])
    valprint('at azimuth', aziMaxDc[pi], 'rad.')
    valprint('at height', zenMaxDc[pi], 'mm')
    ## Plot cumulative creep damage for panels
    pre.plot_pcolor(
      ma[pids], mz[pids]*1e-3, cumDc[-1,pids],
      r'\textsc{cycle creep damage}, '+\
      r'$\delta D_\mathrm{\scriptscriptstyle R}$',
      r'\textsc{azimuth}, $\gamma$ (rad)',r'\textsc{height}, $z$ (m)',
      'cumDc_{}'.format(pi), compass=False, cmap='inferno'
    )
  # ## tex-table comparing heuristics of max. temp and cum. damage:
  # for i in range(n_panels):
  #   pi = 'panel{}'.format(i)
  #   print(r'{} & {} & {} & {} & {} & {:e} & {} & {}\\'.format(
  #     i+1, timeMaxTemp[pi], maxTemp[pi]-273.15, aziMaxTemp[pi],
  #     zenMaxTemp[pi], maxDc[pi], aziMaxDc[pi], zenMaxDc[pi])
  #   )

  # ## tex-table comparing location of highest/lowest thermal strain:
  # for i in range(n_panels):
  #   pi = 'panel{}'.format(i)
  #   print(r'{} & {} & {} & {} & {} & {} & {} & {}\\'.format(
  #     i+1, timeMaxSig[pi], maxSig[pi], aziMaxSig[pi], zenMaxSig[pi],
  #     minSig[pi], aziMinSig[pi], zenMinSig[pi])
  #   )

  ## Plot some time-series information for first six panels:
  tubeflux = {}; tubetemp = {}; tubevel = {}; tubehtc = {}
  h = 5750 # height of most creep damage in panel1
  for i in range(6):
    pi = 'panel{}'.format(i)
    tubeflux[pi] = fflux(
      np.array([simtime, aziMaxDc[pi], h])
    )
    tubetemp[pi] = ftemp(
      np.array([simtime, aziMaxDc[pi], h])
    )-273.15
    tubevel[pi] = fvel(
      np.array([simtime, aziMaxDc[pi], h])
    )
    tubehtc[pi] = fhtc(
      np.array([simtime, aziMaxDc[pi], h])
    )
  pre.plot_timeseries(
    simtime, tubeflux, r'\textsc{time}, $t$ (h)',
    r'\textsc{net flux density}, '+\
    r'$\vec{\phi}_\mathrm{q,net}$ (\si{\mega\watt\per\meter\squared})',
    'timeseries_flux_receiver_z{}'.format(int(h))
  )
  pre.plot_timeseries(
    simtime, tubetemp, r'\textsc{time}, $t$ (h)',
    r'\textsc{fluid temperature}, $T_\mathrm{f}$ (\si{\celsius})',
    'timeseries_temp_receiver_z{}'.format(int(h))
  )
  pre.plot_timeseries(
    simtime, tubevel, r'\textsc{time}, $t$ (h)',
    r'\textsc{bulk flow velocity}, '+\
    r'$U_\mathrm{f}$ (\si{\meter\per\second})',
    'timeseries_vel_receiver_z{}'.format(int(h))
  )
  pre.plot_timeseries(
    simtime, tubehtc, r'\textsc{time}, $t$ (h)',
    r'\textsc{film coefficient}, '+\
    r'$h_\mathrm{f}$ (\si{\watt\per\milli\meter\per\kelvin})',
    'timeseries_htc_receiver_z{}'.format(int(h))
  )

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
    simtime, zen_pts, indexing='ij'
  )
  ## A surface mesh over the outer tube circumference (for the flux)
  time_s, theta_s, z_s = np.meshgrid(
    simtime, np.linspace(0,2*np.pi,nt+1)[:nt],
    zen_pts, indexing = 'ij'
  )

  ## Setup the panels:
  panels = [None]*n_panels
  # ## Setup two tubes per panel at locations of maximum/minimum thermal strain:
  # azitubes = [aziMaxSig, aziMinSig]
  # labels = ['maxSig', 'minSig']
  ## Setup one tube at location of maximum estimated crown temperature
  azitubes = [aziMaxTemp]
  labels = ['maxTemp']
  # ## Setup one tube at location of maximum cumulative creep damage
  # azitubes = [aziMaxDc]
  # labels = ['maxDc']
  # for i in range(n_panels):
  for i in range(6):
    headerprint(' SETUP PANEL {} '.format(i), '=')
    pi = 'panel{}'.format(i)
    panels[i] = receiver.Panel(tube_stiffness)
    tubes = [None]*len(azitubes)
    for j, azi in enumerate(azitubes):
      # Setup each tube in turn and assign it to the correct panel
      strprint(
        'Setting up thermal BCs',
        'tube{}_'.format(i*len(azitubes)+j)+labels[j]
      )
      tubes[j] = receiver.Tube(
        ro_tube, wt_tube, height, nr, nt, nz, T0 = T_ref
      )
      tubes[j].set_times(simtime)
      tubes[j].set_bc(
        receiver.ConvectiveBC(
          ro_tube-wt_tube, height, nz, simtime,
          fluid_temp_time(time_h, azi[pi], z_h)
        ), "inner"
      )
      tubes[j].set_bc(
        receiver.HeatFluxBC(
          ro_tube, height, nt, nz, simtime,
          flux_time(time_s, theta_s, azi[pi], z_s)
        ), "outer"
      )
      tubes[j].set_pressure_bc(
        receiver.PressureBC(
          simtime, pressure(simtime, times, onoff)
        )
      )
      ## Assign tubes to panels
      panels[i].add_tube(
        tubes[j], 'tube{}_'.format(i*len(azitubes)+j)+labels[j]
      )

    ## Add panels to model:
    model.add_panel(panels[i], 'panel{}'.format(i))

  ## Save the receiver to an HDF5 file
  headerprint(' MODEL SAVED: {} '.format(modname))
  model.save(modname)
