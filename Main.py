from numba import cuda
import GUI as gui

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

# znowu cos nie tak z zasymulowanymi obrazami
gui.RunInLineWindow()