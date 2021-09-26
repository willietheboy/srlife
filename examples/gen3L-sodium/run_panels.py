#!/usr/bin/env python3

import sys
import numpy as np
from timeit import default_timer as timer
import setup_panel, solve_panel_mech

from printers import *

if __name__ == '__main__':
  """
  Units: stress in MPa, strain in mm/mm, time in hours, temperature in K
  """
  first = int(sys.argv[1])
  last = int(sys.argv[2])
  dim = '2D'
  ncycles = 30 # assume one cycle is equivalent to four days
  Dc = {}; Df = {}; life = {}; model = {}
  tsetup = {}; tsolve = {}
  for i in range(first, last):
      tick = timer()
      pi = 'panel{}'.format(i)
      setup_panel.main(i, ncycles)
      tock = timer()
      tsetup[pi] = tock - tick
      valprint('Elapsed time', tsetup[pi]/60., 'min')
      Dc[pi] = {}; Df[pi] = {}; life[pi] = {}
      Dc[pi], Df[pi], life[pi] = solve_panel_mech.main(i, dim)
      tick = timer()
      tsolve[pi] = tick - tock
      valprint('Elapsed time', tsolve[pi]/60., 'min')
