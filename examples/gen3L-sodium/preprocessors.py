# -*- python-indent-offset: 2 -*-
#!/usr/bin/env python3
from os import sep
import numpy as np

## The UnregularGrid Interpolators are convenient but slow (factor 10!)
#from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
import matplotlib.colors as colors

## Class for tubular heat transfer with sodium, nitrate-salt and chloride-salt:
import coolant as cool

from printers import *

def make_simtime(ncycles, days, ftime, fsteps, fpres, fmap, p_cycle, p_day):
  """ Function for organising simulation timestepping:

    Parameters:
      ncycles     number of cycle repetitions
      days        list of design point days in cycle
      ftime       dictionary of design point day function times
      fsteps      dictionary of design point day simulation timesteps
      fpres       dictionary of design point day pressure switching
      fmap        dictionary mapping design points to simulation times
      p_cycle     period of cycle (all days)
      p_day       period of days within cycle

  """
  add_lin_steps = lambda times, start, stop, steps, cycle, day: np.append(
    times,
    p_cycle*cycle + p_day*day + np.linspace(start, stop, steps)[1:]
  )
  simtime = np.zeros(1); times = np.zeros(1); onoff = np.zeros(1); mapper = [None]
  for i in range(ncycles):
    for j, season in enumerate(days):
      for k in range(len(ftime[season])-1):
        simtime = add_lin_steps(
          simtime, ftime[season][k], ftime[season][k+1],
          fsteps[season][k], i, j
        )
        times = np.append(
          times, p_cycle*i + p_day*j + ftime[season][k+1]
        )
        onoff = np.append(onoff, fpres[season][k+1])
        if fmap[season][k+1]!=None:
          mapper.append(season+'_'+fmap[season][k+1])
        else:
          mapper.append(None)
  return times, simtime, onoff, mapper

def load_design_points(design_pts, dirname):
  """
  Load receiver design point conditions:
  -> saved in a "DELSOL3-like" flattened cylindrical shape, with:
   -> [i, j] index-notation the same as numpy.meshgrid(..., indexing='ij')
   -> i is azimuth on receiver aperture counter-clockwise from south
   -> j is height up panel/tubes from bottom
  """
  fptemp = {}; fpflux = {}; fpvel = {}
  for dp in design_pts:
    ## Returns function of f(azimuth, height), ndarrays only:
    fptemp[dp] = nearest_interp(dirname+sep+dp, 'fluid_temp')
    fpvel[dp] = nearest_interp(dirname+sep+dp, 'fluid_velocity')
    fpflux[dp] = linear_interp(
      dirname+sep+dp, 'net_flux', 1e-6 # convert W/m^2 to W/mm^2
    )
  return fptemp, fpvel, fpflux

def linear_interp(case, var, scale=1):
  """
  Load lumped-parameter thermohydraulic values for from file
  and create an Interpolator using the SolarPILOT azimuth and height index
  """
  pa = np.genfromtxt(case+sep+'azimuth.csv', delimiter=',')
  pz = np.genfromtxt(case+sep+'height.csv', delimiter=',')*1e3 # convert m to mm

  vals = np.genfromtxt(case+sep+'{}.csv'.format(var), delimiter=',')*scale

  ## interpolate linearly between (surface) values:
  vals_interp = RegularGridInterpolator(
    (pa[:,0], pz[0,:]), vals,
    bounds_error=False, fill_value=None, method='linear'
  )
  return vals_interp

def nearest_interp(case, var, scale=1):
  """
  Load lumped-parameter thermohydraulic values for from file
  and create an Interpolator using the SolarPILOT azimuth and height index
  """
  pa = np.genfromtxt(case+sep+'azimuth.csv', delimiter=',')
  pz = np.genfromtxt(case+sep+'height.csv', delimiter=',')*1e3 # convert m to mm

  vals = np.genfromtxt(case+sep+'{}.csv'.format(var), delimiter=',')*scale

  ## interpolate linearly between (surface) values:
  vals_interp = RegularGridInterpolator(
    (pa[:,0], pz[0,:]), vals,
    bounds_error=False, fill_value=None, method='nearest'
  )
  return vals_interp

def plot_pcolor(x, y, values, quantity, xlabel, ylabel, fname,
                compass=False, cmap='viridis', logNorm=False):
  """
  Plotting values using the SolarPILOT azimuth and height indexation:
  """
  fig = plt.figure(figsize=(3.5, 3.5))
  ax = fig.add_subplot(111)
  if logNorm:
    c = ax.pcolormesh(
      x, y, values, shading='auto', cmap=plt.get_cmap(cmap),
      norm=colors.LogNorm(vmin=values.min(), vmax=values.max())
    )
  else:
    c = ax.pcolormesh(x, y, values, cmap=plt.get_cmap(cmap), shading='auto')
  cb = fig.colorbar(c, ax=ax)
  cb.set_label(quantity)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  if compass:
    ax2 = ax.twiny()
    ax2.set_xlabel(r'\textsc{compass}')
    ax2.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
    ax2.set_xticklabels(['S', 'E', 'N', 'W', 'S'])
  fig.tight_layout()
  fig.savefig('thermohydraulics'+sep+'plots'+sep+fname+'.pdf')
  plt.close('all')

def plot_timeseries(x, Y, xlabel, ylabel, fname):
  """
  Plot time-series of temperature, flux and heat transfer coefficient:
    -> Y is a dictionary containing panel/tube values over time (x)
  """
  fig = plt.figure(figsize=(3.5, 3.5))
  ax = fig.add_subplot(111)
  for i, pi in enumerate(Y):
    ax.plot(x, Y[pi], label='Panel {}'.format(i+1))
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.legend(loc='lower left')
  fig.tight_layout()
  fig.savefig('thermohydraulics'+sep+'plots'+sep+fname+'.pdf')
  plt.close('all')

def _vector_interpolate(base, data):
  """ Interpolate as a vector

  Args:
    base (function): base interpolation method
    data (np.array): data to interpolate
  """
  res = np.zeros(data[0].shape)

  # pylint: disable=not-an-iterable
  for ind in np.ndindex(*res.shape):
    res[ind] = base([d[ind] for d in data])

  return res

def _make_ifn(base):
  """ Helper to deal with getting both a scalar and a vector input

  Args:
    base (function): base interpolation method

  Returns:
    function: a function that interpolates a vector using base at each component
  """
  def ifn(mdata):
    """
      Interpolation function that handles both the scalar and vector cases
    """
    allscalar = all(map(np.isscalar, mdata))
    anyscalar = any(map(np.isscalar, mdata))
    if allscalar:
      return base(mdata)
    elif anyscalar:
      shapes = [a.shape for a in mdata if not np.isscalar(a)]
      # Could check they are all the same, but eh
      shape = shapes[0]
      ndata = [np.ones(shape) * d for d in mdata]
      return _vector_interpolate(base, ndata)
    else:
      return _vector_interpolate(base, mdata)

  return ifn

def make_tablecoths(times, azi, zen, mapper, fflux, ftemp, fvel, period, noon):
  """
  Using the design-point Interpolators (azimuth and height), synthesise
  a Design Cycle (over time) making use of symmetry of design points
  """

  ## Create a tablecloth of values over time, azimuth and height
  mt, ma, mz = np.meshgrid(times, azi, zen, indexing='ij')
  flux = np.zeros(mt.shape)
  temp = np.ones(mt.shape) * 293.15
  vel = np.zeros(mt.shape)
  for i in range(len(times)):
    if mapper[i] != None:
      ## Check to see whether before or after noon:
      if times[i] % period < noon:
        ## invert (afternoon) results by azimuth:
        flux[i] = fflux[mapper[i]](
          [np.array([a, z]) for a in azi[::-1] for z in zen]
        ).reshape([len(azi), len(zen)])
        temp[i] = ftemp[mapper[i]](
          [np.array([a, z]) for a in azi[::-1] for z in zen]
        ).reshape([len(azi), len(zen)])
        vel[i] = fvel[mapper[i]](
          [np.array([a, z]) for a in azi[::-1] for z in zen]
        ).reshape([len(azi), len(zen)])
      else:
        ## actual (afternoon) results:
        flux[i] = fflux[mapper[i]](
          [np.array([a, z]) for a in azi for z in zen]
        ).reshape([len(azi), len(zen)])
        temp[i] = ftemp[mapper[i]](
          [np.array([a, z]) for a in azi for z in zen]
        ).reshape([len(azi), len(zen)])
        vel[i] = fvel[mapper[i]](
          [np.array([a, z]) for a in azi for z in zen]
        ).reshape([len(azi), len(zen)])
  ## Use 'linear' interpolation over time:
  fflux = RegularGridInterpolator(
    (times, azi, zen), flux,
    bounds_error=False, fill_value=None, method='linear'
  )
  ftemp = RegularGridInterpolator(
    (times, azi, zen), temp,
    bounds_error=False, fill_value=None, method='linear'
  )
  fvel = RegularGridInterpolator(
    (times, azi, zen), vel,
    bounds_error=False, fill_value=None, method='linear'
  )
  return _make_ifn(ftemp), _make_ifn(fvel), _make_ifn(fflux)

def make_tablecloth_htc(times, ri, ro, azi, zen, ftemp, fvel):
  """
  Use nashTubeStress (https://github.com/willietheboy/nashTubeStress)
  pipe calculator to calculate the heat transfer coefficient
  depending on flow velocity, mass-flow or heat-capacity-rate
    -> with conversion from m to mm!
  """
  verbose = False
  htc = np.zeros((len(times),len(azi),len(zen)))
  mt, ma, mz = np.meshgrid(times, azi, zen, indexing='ij')
  sodium = cool.LiquidSodium(verbose);
  for i in range(len(times)):
    for j in range(len(azi)):
      for k in range(len(zen)):
        if fvel(np.array([times[i], azi[j], zen[k]])) > 0:
          sodium.update(ftemp(np.array([times[i], azi[j], zen[k]])))
          ret = cool.heat_transfer_coeff(
            sodium, ri*1e-3, ro*1e-3, 'Chen', 'velocity',
            fvel(np.array([times[i], azi[j], zen[k]])), verbose
          ) # convert tube dims to mm
          htc[i,j,k] = ret*1e-6 # convert W/(m^2.K) to W/(mm^2.K)
  fhtc = RegularGridInterpolator(
    (times, azi, zen), htc,
    bounds_error=False, fill_value=None, method='linear'
  )
  return _make_ifn(fhtc)

def crown_stress(flux, T_f, htc, ri, ro):
  """ Kistler (1987), Kolb (2011) from Babcock & Wilcox (1984):
  Simple analytical estimation of thin-walled tubes subject to one-sided flux
  with temperature dependent tube conductivity, elastic modulus and
  coefficient of thermal expansion for A740 (copied from srlife)
  """
  nu = 0.31
  flambda = lambda T: np.interp(
    T,
    np.array([ 293.15,  373.15,  473.15,  573.15,  673.15,  773.15,  873.15,
               973.15, 1073.15, 1173.15]),
    np.array([0.0102, 0.0117, 0.013 , 0.0145, 0.0157, 0.0171, 0.0184, 0.0202,
       0.0221, 0.0238])
  )
  fyoungs = lambda T: np.interp(
    T,
    np.array([ 293.15,  373.15,  473.15,  573.15,  673.15,  773.15,  873.15,
               973.15, 1073.15, 1173.15]),
    np.array([221000., 218000., 212000., 206000., 200000., 193000., 186000.,
       178000., 169000., 160000.])
  )
  falpha = lambda T: np.interp(
    T,
    np.array([ 293.15,  373.15,  473.15,  573.15,  673.15,  773.15,  873.15,
               973.15, 1073.15, 1173.15]),
    np.array([1.238e-05, 1.304e-05, 1.350e-05, 1.350e-05, 1.393e-05, 1.427e-05,
       1.457e-05, 1.572e-05, 1.572e-05, 1.641e-05])
  )
  sigmaEq = np.zeros(flux.shape)
  T_mc = np.zeros(flux.shape)
  nonzeros = flux != 0
  T_ci = T_f[nonzeros] + ((flux[nonzeros]*(ro/ri)) / htc[nonzeros])
  T_co = T_ci + flux[nonzeros] * (ro/flambda(T_ci)) * np.log(ro/ri)
  T_mc[nonzeros] = (T_ci + T_co) / 2.
  T_m = T_f[nonzeros] + (1 / np.pi) * (T_mc[nonzeros] - T_f[nonzeros])
  sigmaR = 0.0
  sigmaTheta = falpha(T_mc[nonzeros]) * fyoungs(T_mc[nonzeros]) * \
    ((T_co - T_ci) / (2*(1-nu)))
  sigmaZ = falpha(T_mc[nonzeros]) * fyoungs(T_mc[nonzeros]) * \
    (T_mc[nonzeros] - T_m)
  sigmaEq[nonzeros] = np.sqrt(0.5 * ((sigmaR - sigmaTheta)**2 + \
                                     (sigmaTheta - sigmaZ)**2 + \
                                     (sigmaZ - sigmaR)**2))
  return sigmaEq, T_mc

def heuristics(times, ma, mz, ri, ro, ftemp, fhtc, fflux):
  """
  Calculate simple 1D crown temperature and stress for whole receiver
   -> material properties A740H @ 700degC

  """
  na = ma.shape[0]; nz = mz.shape[1]
  tempf = np.zeros([len(times), na, nz])
  htc = np.zeros([len(times), na, nz])
  flux = np.zeros([len(times), na, nz])
  sigEq = np.zeros([len(times), na, nz])
  tempc = np.zeros([len(times), na, nz])
  for j, t in enumerate(times):
    tempf[j,:,:] = ftemp(np.array([t, ma, mz]))
    htc[j,:,:] = fhtc(np.array([t, ma, mz]))
    flux[j,:,:] = fflux(np.array([t, ma, mz]))
  ## Crown temperature and stress using constant material values at 700degC:
  sigEq, tempc = crown_stress(
    flux, tempf, htc, ri, ro
  )
  return tempc, sigEq

def cumulative_creep_damage(times, sigEq, temp, material):
  """
  Calculate cumulative cycle creep damage (for heuristics)

  Parameters:
    tube        single tube with full results
    material    damage material model
  """
  tR = material.time_to_rupture(
    "averageRupture",
    temp,
    sigEq
  )
  dts = np.diff(times)
  dts = np.insert(dts, 0, 0.0)
  time_dmg = dts[:,np.newaxis,np.newaxis]/tR[:]
  cum_dmg = np.cumsum(time_dmg, axis=0)
  return cum_dmg


def copy_temperature(thertube, thermodel, mechtube, mechmodel):
  """
  Copy single-cycle thermal results to mechanical model (ncycles)

  """
  tm = np.mod(mechtube.times, thermodel.period)
  inds = list(np.where(tm == 0)[0])
  if len(inds) != (mechmodel.days + 1):
    raise ValueError("Thermal times not compatible with the mechanical"
                     " number of days and cycle period!")
  dimt = thertube.results['temperature'].shape
  lenm = mechtube.times.shape[0]
  mechtemp = np.ones([lenm, dimt[1], dimt[2]]) * thertube.T0
  for i in range(mechmodel.days):
    mechtemp[inds[i]:inds[i+1]+1] = thertube.results['temperature']
  return mechtemp
