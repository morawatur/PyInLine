import numpy as np
from numba import cuda

# -------------------------------------------------------------------

def DetermineCudaConfig(dim):
    blockDim = 32, 32
    gridDim = np.ceil(dim / blockDim[0]).astype(np.int32), np.ceil(dim / blockDim[1]).astype(np.int32)
    return blockDim, gridDim

# -------------------------------------------------------------------

def DetermineCudaConfigNew(dims):
    blockDim = 32, 32
    gridDim = np.ceil(dims[0] / blockDim[0]).astype(np.int32), np.ceil(dims[1] / blockDim[1]).astype(np.int32)
    return blockDim, gridDim

# -------------------------------------------------------------------

def GetGPUMemoryUsed():
    freeMem, totalMem = cuda.current_context().get_memory_info()
    gpuMemUsedInMBs = (totalMem - freeMem) / (1024.0 ** 2)
    print('GPU memory used = {0:.2f} MB'.format(gpuMemUsedInMBs))

# -------------------------------------------------------------------

def GetGPUFreeMemory():
    gpuFreeMemInBytes = cuda.current_context().get_memory_info()[0]
    gpuFreeMemInMBs = gpuFreeMemInBytes / (1024.0 ** 2)
    print('Available GPU memory = {0:.2f} MB'.format(gpuFreeMemInMBs))