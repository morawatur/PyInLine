# TODO list
# 1 - update code from PyHoLo
# 2 - switch to CPU
# 3 - unify Python style
# 4 - save/load current session (i.e. save/load current images and parameters to/from binary files)

from numba import cuda

import GUI as gui

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

gui.RunInLineWindow()
