# -*- python-indent-offset: 2 -*-
#!/usr/bin/env python3
from os import sep
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from printers import *

def eff_stress(tube):
  """
  Calculate von Mises effective stress

  Parameters:
    tube        single tube with complete structural results
  """
  vm = np.sqrt((
    (tube.quadrature_results['stress_xx'] -
     tube.quadrature_results['stress_yy'])**2.0 +
    (tube.quadrature_results['stress_yy'] -
     tube.quadrature_results['stress_zz'])**2.0 +
    (tube.quadrature_results['stress_zz'] -
     tube.quadrature_results['stress_xx'])**2.0 +
    3.0 * (tube.quadrature_results['stress_xy']**2.0 +
           tube.quadrature_results['stress_yz']**2.0 +
           tube.quadrature_results['stress_xz']**2.0))/2.0
  )
  return vm

def eff_strain(tube):
  """
  Calculate effective (mechanical) strain

  Parameters:
    tube        single tube with complete structural results
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

def eq_strain_range(tube, nu = 0.5):
  """
  Calculate ASME III, HBB T-1413 equivalent strain range referenced
  from a zero (starting) strain range (for visualisation/verification)

  Parameters:
    tube         single tube with full results

  Additional parameters:
    nu           effective Poisson's ratio to use
  """
  strain_names = [
    'mechanical_strain_xx', 'mechanical_strain_yy', 'mechanical_strain_zz',
    'mechanical_strain_yz', 'mechanical_strain_xz', 'mechanical_strain_xy'
  ]
  strain_factors = [1.0,1.0,1.0,np.sqrt(2),np.sqrt(2),np.sqrt(2)]
  strains = np.array(
    [ef*tube.quadrature_results[en]
     for en,ef in zip(strain_names, strain_factors)]
  )
  pt_eranges = np.zeros(strains[0].shape)
  nt = strains.shape[1]
  for i in range(nt):
    de = strains[:,i]
    eq = np.sqrt(2) / (2*(1+nu)) * np.sqrt(
      (de[0]-de[1])**2.0 + (de[1]-de[2])**2.0 + (de[2]-de[0])**2.0
      + 3.0/2.0 * (de[3]**2.0 + de[4]**2.0 + de[5]**2.0)
    )
    pt_eranges[i] = eq

  return pt_eranges

def creep_fatigue(dmodel, tube, material, receiver, n):
  """
    Calculate the single-tube number of repetitions to failure

    Parameters:
      tube        single tube with full results
      dmodel      damage material model
      receiver    receiver, for metadata
      n           numerator on fatigue LDS equation
  """
  # Material point cycle creep damage
  Dc = dmodel.creep_damage(tube, material, receiver)

  # Material point cycle fatigue damage
  Df = dmodel.fatigue_damage(tube, material, receiver) * n

  nc = receiver.days

  # This is going to be expensive, but I don't see much way around it
  return np.max(Dc.reshape(nc,-1).T, axis=0), \
    np.max(Df.reshape(nc,-1).T, axis=0), \
    min(dmodel.calculate_max_cycles(
      dmodel.make_extrapolate(c),
      dmodel.make_extrapolate(f), material
    ) for c,f in zip(Dc.reshape(nc,-1).T, Df.reshape(nc,-1).T))

def log10_interp1d (x, xx, yy, order=1):
  logx = np.log10(xx)
  logy = np.log10(yy)
  coeffs = np.polyfit(logx, logy, order)
  poly = np.poly1d(coeffs)
  lin_interp = lambda N: np.power(10, poly(np.log10(N)))
  return lin_interp(x), coeffs

def plot_cycle_cdamage(cycDc, ncycles, tubeid, filename, verbose=False):

  figD = plt.figure(figsize=(3.5, 3.5))
  axD = figD.add_subplot(111)
  ## extrapolate to 2750 cycles (11e3 days)
  Nex = np.arange(1, 2751)
  for i, pi in enumerate(cycDc):
    D = cycDc[pi][tubeid][:ncycles]
    N = np.arange(1,len(D)+1)[:ncycles]
    Dex, coeffs = log10_interp1d(Nex, N, D, 1)
    poly = r'$10^{'+'{:.2f}'.format(coeffs[0])+\
      '\log_{10}N'+'{:.2f}'.format(coeffs[1])+'}$'
    if verbose:
      headerprint(pi, ' ')
      strprint('Cycle creep damage eq.', poly)
      Dlife = np.sum(Dex)
      valprint('sum(Dc[:2750])', Dlife)
    axD.loglog(
      N, D,
      color='C{}'.format(i)
    )
    axD.loglog(
      Nex, Dex,
      label=r'Panel {} $\to'.format(i+1)+\
      r'\num{'+'{:.2e}'.format(Dex[-1])+'}$',
      # label=r'Panel {} $\to'.format(i+1)+': '+poly,
      linestyle='dashed', color='C{}'.format(i)
    )
  axD.set_xlabel(r'\textsc{cycle number}, $N$')
  axD.set_ylabel(r'\textsc{cycle creep damage}, '+\
                 '$\delta D_\mathrm{\scriptscriptstyle R}$')
  axD.legend(loc='best')
  figD.tight_layout()
  figD.savefig('{}'.format(filename)) # extension included
  plt.close(figD)

def plot_cycle_fdamage(cycDc, tubeid, filename):

  figD = plt.figure(figsize=(3.5, 3.5))
  axD = figD.add_subplot(111)
  for i, pi in enumerate(cycDc):
    D = cycDc[pi][tubeid]
    N = np.arange(1,len(D)+1)
    axD.plot(
      N, D, label='Panel {}'.format(i+1)
    )
    # axD.plot(
    #   N, np.ones(len(N))*aDf[pi], '--', color='C{}'.format(i)
    # )
  axD.set_xlabel(r'\textsc{cycle number}, $N$')
  axD.set_ylabel(r'\textsc{cycle fatigue damage}, '+\
                 '$\delta D_\mathrm{\scriptscriptstyle F}$')
  axD.legend(loc='best')
  figD.tight_layout()
  figD.savefig('{}'.format(filename)) # extension included
  plt.close(figD)
