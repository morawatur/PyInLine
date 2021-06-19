from numba import cuda

import GUI as gui

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

gui.RunInLineWindow()

# TODO list
# 1 - re-alignment doesn't work
# 2 - save/load current session (i.e. save/load current images and parameters to/from binary files)
# 3 - unify Python style
