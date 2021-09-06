#!/usr/bin/env python3

import os.path

import numpy as np
import numpy.linalg as la
import scipy.io as sio

import matplotlib.pyplot as plt
from neml import models, elasticity, parse

import sys
sys.path.append('../../..')
from srlife import receiver, structural, spring

class TestCase:
  """
  Units are MPa, mm, MN
  """
  def __init__(self, name, T, analytic, z_force, z_strain = 0.0,
               ri = 8.0, ro = 10.0, h = 10.0, alpha = 1.0e-5,
               E = 100000.0, nu = 0.3, p = 100.0, dT = 0.0,
               spring = False):
    self.name = name
    self.Tfn = T
    self.afn = analytic
    self.fzfn = z_force

    self.ri = ri
    self.ro = ro
    self.h = h

    self.alpha = alpha
    self.E = E
    self.nu = nu

    self.p = p
    self.dT = dT

    self.sprung = spring
    self.ez = z_strain

  def T(self, r):
    return self.Tfn(r, self.ri, self.ro, self.dT)

  def exact(self, r):
    return self.afn(r, self.p, self.ri, self.ro, self.E, self.nu, self.dT, self.alpha)

  def axial_force(self):
    return self.fzfn(self.p, self.ri, self.ro)

  def axial_disp(self, z):
    return z * self.ez

  def make_mat(self):
    emodel = elasticity.IsotropicLinearElasticModel(self.E, "youngs",
        self.nu, "poissons")
    return models.SmallStrainElasticity(emodel, alpha = self.alpha)

  def make_tube(self, dim, nr = 10, nt = 20, nz = 10):
    tube = receiver.Tube(self.ro, self.ro - self.ri, self.h, nr, nt, nz)

    if dim == 1:
      tube.make_1D(self.h/2, 0)
    elif dim == 2:
      tube.make_2D(self.h/2)

    times = np.array([0,1])
    tube.set_times(times)

    R, _, _ = tube.mesh
    Ts = np.zeros((2,) + R.shape[:dim])
    if dim == 1:
      Ts[1] = self.T(R[:,0,0])
    elif dim == 2:
      Ts[1] = self.T(R[:,:,0])
    else:
      Ts[1] = self.T(R[:,:,:])

    tube.add_results("temperature", Ts)

    if self.p != 0:
      tube.set_pressure_bc(receiver.PressureBC(times, times * self.p))

    return tube

  def make_tube_spring(self, tube, smat, ssolver, axial_force):
    network = spring.SpringNetwork()
    network.add_node(0)
    network.add_node(1)
    network.add_edge(
      0,1,
      object = spring.TubeSpring(tube, ssolver, smat)
    )

    network.set_times([0,1])

    network.validate_setup()

    network.displacement_bc(0, lambda t: 0.0)
    network.force_bc(1, lambda t: axial_force)

    return network

  def run_comparison(self, dim, solver,
                     nr = 10, nt = 20, nz = 10):
    mat = self.make_mat()
    tube = self.make_tube(dim, nr, nt, nz)

    solver.setup_tube(tube)
    state_n = solver.init_state(tube, mat)

    if self.sprung:
      """
      Use a single spring network with imposed axial force
      (e.g. out-of-plane pressure or dead-weight)
      """
      network = self.make_tube_spring(tube, mat, solver, self.axial_force())
      network.solve(1)
      #print('Tube end displacement: %e' % network.displacements[1])
      edge = list(network.edges(data=True))
      spring = edge[0][2]['object']
      state_np1 = spring.state_np1
      solver.dump_state(spring.tube, 1, state_np1)
    else:
      if dim == 1 or dim == 2:
        state_np1 = solver.solve(tube, 1, state_n, self.axial_disp(1))
      else:
        state_np1 = solver.solve(tube, 1, state_n, self.axial_disp(self.h))
      solver.dump_state(tube, 1, state_np1)

    return state_np1, tube

  def get_comparison(self, tube):
    if tube.ndim == 3:
      z = tube.nz // 2
      x_avg = np.mean(tube.results['disp_x'])
      u = tube.results['disp_x'][1,:,0,z] - 2*x_avg
      r = tube.mesh[0][:,0,z]
    elif tube.ndim == 2:
      # The displacements tend to drift, need to recenter
      x_avg = np.mean(tube.results['disp_x'])
      u = tube.results['disp_x'][1,:,0] - 2*x_avg
      r = tube.mesh[0][:,0,0]
    else:
      u = tube.results['disp_x'][1]
      r = tube.mesh[0][:,0,0]

    return u, r

  def plot_comparison(self, tube):
    u, r = self.get_comparison(tube)

    plt.figure()
    plt.plot(r, u, 'k-')
    plt.plot(r, self.exact(r), 'k--')
    plt.xlabel("Radial position")
    plt.ylabel("Radial displacement")
    plt.title(self.name + ": " + "%iD" % tube.ndim)
    plt.show()

  def plot_comparison_pdf(self, tube):
    u, r = self.get_comparison(tube)

    fig1 = plt.figure(figsize=(3.5, 3.5))
    ax1 = fig1.add_subplot(111)
    ax1.plot(r, u, 'k-')
    ax1.plot(r, self.exact(r), 'k--')
    ax1.set_xlabel(r'\textsc{radial position}, $r$ (mm)')
    ax1.set_ylabel(r'\textsc{radial displacement}, $u$ (mm)')
    ax1.set_title(self.name + ": " + "%iD" % tube.ndim)
    fig1.tight_layout()
    fig1.savefig('pdf/'+self.afn.__name__ + "_%iD" % tube.ndim + '.pdf')
    fig1.savefig('pdf/'+self.afn.__name__ + "_%iD" % tube.ndim + '.png', dpi=150)
    plt.close('all')

  def evaluate_comparison(self, tube):
    u, r = self.get_comparison(tube)

    err = np.abs(u - self.exact(r))
    rel = err / np.abs(self.exact(r))

    return np.max(err), np.max(rel)

"""
Exact analytical functions
"""

def pressure_plane_strain_R_disp(r, p, ri, ro, E, nu, dT, alpha):
  """
  Assumption of plane strain (axial strain = 0)
  """
  A = ri**2 * ro**2 * -p / (ro**2 - ri**2)
  C = p * ri**2 / (ro**2 - ri**2)
  u = (-A*nu - A + C*r**2*(-2*nu**2 - nu + 1))/(E*r)
  return u

def pressure_out_of_plane_R_disp(r, p, ri, ro, E, nu, dT, alpha):
  """
  Assumption of cylinder closed at both ends (constant axial force)
  """
  A = ri**2 * ro**2 * -p / (ro**2 - ri**2)
  C = p * ri**2 / (ro**2 - ri**2)
  u = (-A*nu - A + C*r**2*(1 - 2*nu))/(E*r)
  return u

def thermal_plane_strain_R_disp(r, p, ri, ro, E, nu, dT, alpha):
  """
  Constants of integration (stress) assume a linear temperature gradient
  and constraint of plane strain (axial strain = 0)
  """
  C_1 = -alpha*dT*(nu + 1)*(2*nu - 1)*\
    (-ri**3/6 + ri*ro**2/2 - ro**3/3)/((nu - 1)*(ri - ro)**2*(ri + ro))
  C_2 = alpha*dT*ri**2*(nu + 1)*\
    (-ri**3/6 + ri*ro**2/2 - ro**3/3)/((nu - 1)*(ri - ro)**2*(ri + ro))
  C_3 = 0
  u = C_1*r + C_2/r + C_3*nu*r/E + alpha*(nu + 1)*\
    (-dT*r**3/(3*ri - 3*ro) + dT*r**2*ri/(2*ri - 2*ro) + \
     dT*ri**3/(3*ri - 3*ro) - dT*ri**3/(2*ri - 2*ro))/(r*(1 - nu))
  return u

def thermal_gen_plane_strain_R_disp(r, p, ri, ro, E, nu, dT, alpha):
  """
  Constants of integration (stress) assume a linear temperature gradient
  and a constant axial stress (C_3) required to annul axial force
  """
  C_1 = -alpha*dT*(nu + 1)*(2*nu - 1)*\
    (-ri**3/6 + ri*ro**2/2 - ro**3/3)/((nu - 1)*(ri - ro)**2*(ri + ro))
  C_2 = alpha*dT*ri**2*(nu + 1)*\
    (-ri**3/6 + ri*ro**2/2 - ro**3/3)/((nu - 1)*(ri - ro)**2*(ri + ro))
  C_3 = -E*alpha*dT*(ri + 2*ro)/(3*ri + 3*ro)
  u = C_1*r + C_2/r + C_3*nu*r/E + alpha*(nu + 1)*\
    (-dT*r**3/(3*ri - 3*ro) + dT*r**2*ri/(2*ri - 2*ro) + \
     dT*ri**3/(3*ri - 3*ro) - dT*ri**3/(2*ri - 2*ro))/(r*(1 - nu))
  return u

def total_plane_strain_R_disp(r, p, ri, ro, E, nu, dT, alpha):
  """
  Constants of integration (stress) assume a linear temperature gradient
  and constraint of plane strain
  """
  A = ri**2 * ro**2 * -p / (ro**2 - ri**2)
  C = p * ri**2 / (ro**2 - ri**2)
  C_1 = -alpha*dT*(nu + 1)*(2*nu - 1)*\
    (-ri**3/6 + ri*ro**2/2 - ro**3/3)/((nu - 1)*(ri - ro)**2*(ri + ro))
  C_2 = alpha*dT*ri**2*(nu + 1)*\
    (-ri**3/6 + ri*ro**2/2 - ro**3/3)/((nu - 1)*(ri - ro)**2*(ri + ro))
  C_3 = 0
  u = (-A*nu - A + C*r**2*(-2*nu**2 - nu + 1))/(E*r) + \
    C_1*r + C_2/r + C_3*nu*r/E + alpha*(nu + 1)*\
    (-dT*r**3/(3*ri - 3*ro) + dT*r**2*ri/(2*ri - 2*ro) + \
     dT*ri**3/(3*ri - 3*ro) - dT*ri**3/(2*ri - 2*ro))/(r*(1 - nu))
  return u

def total_gen_plane_strain_R_disp(r, p, ri, ro, E, nu, dT, alpha):
  """
  Constants of integration (stress) assume a linear temperature gradient,
  a constant out of plane pressure stress and the force required to annul
  thermally induced axial stress
  """
  A = ri**2 * ro**2 * -p / (ro**2 - ri**2)
  C = p * ri**2 / (ro**2 - ri**2)
  C_1 = -alpha*dT*(nu + 1)*(2*nu - 1)*\
    (-ri**3/6 + ri*ro**2/2 - ro**3/3)/((nu - 1)*(ri - ro)**2*(ri + ro))
  C_2 = alpha*dT*ri**2*(nu + 1)*\
    (-ri**3/6 + ri*ro**2/2 - ro**3/3)/((nu - 1)*(ri - ro)**2*(ri + ro))
  C_3 = -E*alpha*dT*(ri + 2*ro)/(3*ri + 3*ro)
  u = (-A*nu - A + C*r**2*(1 - 2*nu))/(E*r) + \
    C_1*r + C_2/r + C_3*nu*r/E + alpha*(nu + 1)*\
    (-dT*r**3/(3*ri - 3*ro) + dT*r**2*ri/(2*ri - 2*ro) + \
     dT*ri**3/(3*ri - 3*ro) - dT*ri**3/(2*ri - 2*ro))/(r*(1 - nu))
  return u

"""
Axial (e.g. out of plane pressure, dead-weight) boundary conditions
"""

def no_out_of_plane_Z_force(p, ri, ro):
  """
  e.g. for use in generalised plane strain
  """
  return 0.0

def pressure_out_of_plane_Z_force(p, ri, ro):
  """
  Assumption of tube closed at both ends (C is stress)
  """
  C = p * ri**2 / (ro**2 - ri**2)
  return -C * np.pi * (ro**2 - ri**2)

if __name__ == "__main__":

  solver = structural.PythonTubeSolver(verbose = False)

  cases = [
    TestCase("Pressure, plane strain",
             lambda r, ri, ro, dT: 0.0,
             pressure_plane_strain_R_disp,
             no_out_of_plane_Z_force,
             p = 100, ri = 8, ro = 10.0,
             spring = False),
    TestCase("Pressure, sprung",
             lambda r, ri, ro, dT: 0.0,
             pressure_out_of_plane_R_disp,
             pressure_out_of_plane_Z_force,
             p = 100, ri = 8, ro = 10.0,
             spring = True),
    TestCase("Thermal, plane strain",
             lambda r, ri, ro, dT: dT * (r - ri) / (ro - ri),
             thermal_plane_strain_R_disp,
             no_out_of_plane_Z_force,
             p = 1e-9, ri = 8, ro = 10.0, dT = 100,
             spring = False),
    TestCase("Thermal, sprung",
             lambda r, ri, ro, dT: dT * (r - ri) / (ro - ri),
             thermal_gen_plane_strain_R_disp,
             no_out_of_plane_Z_force,
             p = 1e-9, ri = 8, ro = 10.0, dT = 100,
             spring = True),
    TestCase("Pressure + thermal, plane strain",
             lambda r, ri, ro, dT: dT * (r - ri) / (ro - ri),
             total_plane_strain_R_disp,
             no_out_of_plane_Z_force,
             p = 100, ri = 8, ro = 10.0, dT = 100,
             spring = False),
    TestCase("Pressure + thermal, sprung",
             lambda r, ri, ro, dT: dT * (r - ri) / (ro - ri),
             total_gen_plane_strain_R_disp,
             pressure_out_of_plane_Z_force,
             p = 100, ri = 8, ro = 10.0, dT = 100,
             spring = True)
  ]

  print("Analytical comparison")
  print("=====================")
  print("")
  ## tube dimensions:
  nr = 8
  nt = 20
  nz = 10
  for d in range(1,4):
    for case in cases:
      state, tube = case.run_comparison(d, solver, nr, nt, nz)
      a, r = case.evaluate_comparison(tube)
      print(case.name + ": " "%iD" % d)
      print("Axial (resultant) force: %e (MN)" % state.force)
      print("Max absolute error: %e" % a)
      print("Max relative error: %e" % r)
      print("")
      ## graphical:
      #case.plot_comparison(tube)
      ## print to PDF:
      case.plot_comparison_pdf(tube)
