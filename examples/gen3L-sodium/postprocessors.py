# -*- python-indent-offset: 2 -*-
#!/usr/bin/env python3
from os import sep
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

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

def creep_fatigue(dmodel, tube, material, receiver, fmult):
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
  Df = dmodel.fatigue_damage(tube, material, receiver) * fmult

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
