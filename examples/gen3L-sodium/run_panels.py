#!/usr/bin/env python3

import sys
import numpy as np
from timeit import default_timer as timer
import setup_panel, solve_panel_mech

from printers import *

if __name__ == '__main__':
  """
  Run (long) inelastic (repeat design cycle) analysis sequentially

  Usage: run_panels.py <first_panel> <last_panel> <ncycles>

  Where:
      first_panel    Index of first panel in series run [0:11]
      last_panel     Index of last panel in series run [1:12]
      ncycles        Number of times to repeat (structural) design cycle
  """

  if len(sys.argv) < 4 or len(sys.argv) > 4:
    RuntimeError("Usage: run_panels.py <first_panel> <last_panel> <ncycles>")
  first_panel = int(sys.argv[1])
  last_panel = int(sys.argv[2])
  ncycles = int(sys.argv[3])
  dim = '2D'
  defo = 'elastic_model'
  Dc = {}; Df = {}; life = {}; model = {}
  tsetup = {}; tsolve = {}
  for i in range(first_panel, last_panel):
      tick = timer()
      pi = 'panel{}'.format(i)
      setup_panel.main(i, ncycles)
      tock = timer()
      tsetup[pi] = tock - tick
      valprint('Elapsed time', tsetup[pi]/60., 'min')
      Dc[pi] = {}; Df[pi] = {}; life[pi] = {}
      Dc[pi], Df[pi], life[pi] = solve_panel_mech.main(i, dim, defo)
      tick = timer()
      tsolve[pi] = tick - tock
      valprint('Elapsed time', tsolve[pi]/60., 'min')
