import math
import numpy as np
from numba import cuda

import aberrations as ab
import Constants as const
import CudaConfig as ccfg
import ArraySupport as arrsup
import ImageSupport as imsup
import CrossCorr as cc

# -------------------------------------------------------------------

def CalcTransferFunction(imgDim, pxDim, defocusChange, ap=0):
    blockDim, gridDim = ccfg.DetermineCudaConfig(imgDim)
    ctf = imsup.Image(imgDim, imgDim, imsup.Image.cmp['CAP'], imsup.Image.mem['GPU'])
    ctfCoeff = np.pi * const.ewfLambda * defocusChange
    CalcTransferFunction_dev[gridDim, blockDim](ctf.amPh.am, ctf.amPh.ph, imgDim, pxDim, ctfCoeff)
    ctf.defocus = defocusChange
    # -----
    if ap > 0:
        ctf = InsertAperture(ctf, ap)
    # -----
    return ctf

# -------------------------------------------------------------------

@cuda.jit('void(float32[:, :], int32, int32)')
def CalcRecSquareDistances_dev(rsd, imgDim, pxDim):
    x, y = cuda.grid(2)
    if x >= imgDim or y >= imgDim:
        return
    recPxWidth = 1.0 / (imgDim * pxDim)
    recOrigin = -1.0 / (2.0 * pxDim)
    recXDist = recOrigin + x * recPxWidth
    recYDist = recOrigin + y * recPxWidth
    rsd[x, y] = recXDist * recXDist + recYDist * recYDist

# -------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :], int32, float32, float32)')
def CalcTransferFunction_dev(ctfAm, ctfPh, imgDim, pxDim, ctfCoeff):
    x, y = cuda.grid(2)
    if x >= imgDim or y >= imgDim:
        return

    recPxWidth = 1.0 / (imgDim * pxDim)
    recOrigin = -1.0 / (2.0 * pxDim)

    recXDist = recOrigin + x * recPxWidth
    recYDist = recOrigin + y * recPxWidth
    recSquareDist = recXDist * recXDist + recYDist * recYDist

    ctfAm[x, y] = 1.0
    ctfPh[x, y] = ctfCoeff * recSquareDist

# -------------------------------------------------------------------

def calc_ctf(img_dim, px_dim, defocus, Cs=const.Cs, A1=ab.PolarComplex(const.A1_amp, const.A1_phs),
             df_spread=const.df_spread, conv_angle=const.conv_angle, aperture=const.aperture, A1_dir=1):
    df_coeff = np.pi * const.ewfLambda * defocus
    Cs_coeff = 0.5 * np.pi * (const.ewfLambda ** 3) * Cs
    Cs_spat_coeff = Cs * const.ewfLambda ** 2
    conv_ang_coeff = (np.pi * conv_angle) ** 2
    df_spread_coeff = 0.5 * np.pi * const.ewfLambda * df_spread
    A1_re_coeff = A1_dir * np.pi * const.ewfLambda * A1.real()
    A1_im_coeff = A1_dir * 2 * np.pi * const.ewfLambda * A1.imag()

    block_dim, grid_dim = ccfg.DetermineCudaConfig(img_dim)
    ctf = imsup.Image(img_dim, img_dim, imsup.Image.cmp['CAP'], imsup.Image.mem['GPU'])

    calc_ctf_dev[grid_dim, block_dim](ctf.amPh.am, ctf.amPh.ph, img_dim, px_dim, defocus, df_coeff, Cs_coeff, A1_re_coeff,
                                      A1_im_coeff, Cs_spat_coeff, conv_ang_coeff, df_spread_coeff)
    ctf.defocus = defocus
    # -----
    if aperture > 0:
        ctf = InsertAperture(ctf, aperture)
    # -----
    return ctf

# -------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :], int32, float32, float32, float32, float32, float32, float32, float32, float32, float32)')
def calc_ctf_dev(ctf_am, ctf_ph, img_dim, px_dim, df, df_cf, Cs_cf, A1_re_cf, A1_im_cf, Cs_spat_cf, conv_ang_cf, df_spread_cf):
    x, y = cuda.grid(2)
    if x >= img_dim or y >= img_dim:
        return

    rec_px_dim = 1.0 / (img_dim * px_dim)
    rec_orig = -1.0 / (2.0 * px_dim)

    kx = rec_orig + x * rec_px_dim
    ky = rec_orig + y * rec_px_dim
    k_squared = kx ** 2 + ky ** 2
    k_squares_diff = kx ** 2 - ky ** 2

    spat_env_fun = math.exp(-(k_squared * conv_ang_cf) * (df + Cs_spat_cf * k_squared) ** 2)
    temp_env_fun = math.exp(-(df_spread_cf * k_squared) ** 2)

    ctf_am[x, y] = spat_env_fun * temp_env_fun
    ctf_ph[x, y] = df_cf * k_squared + Cs_cf * (k_squared ** 2)
    ctf_ph[x, y] += A1_re_cf * k_squares_diff + A1_im_cf * kx * ky

# -------------------------------------------------------------------

def InsertAperture(img, ap_radius):
    img_dim = img.amPh.am.shape[0]
    blockDim, gridDim = ccfg.DetermineCudaConfig(img_dim)
    img.MoveToGPU()
    img_with_aperature = imsup.CopyImage(img)
    InsertAperture_dev[gridDim, blockDim](img_with_aperature.amPh.am, img_with_aperature.amPh.ph, img_dim, ap_radius)
    return img_with_aperature

# -------------------------------------------------------------------

def mult_by_hann_window(img, N=100):
    new_img = imsup.CopyImage(img)
    new_img.ReIm2AmPh()
    new_img.MoveToCPU()

    hann = np.hanning(N)
    hann_2d = np.sqrt(np.outer(hann, hann))

    hann_win = imsup.ImageExp(N, N)
    hann_win.LoadAmpData(hann_2d)

    hmin, hmax = (img.width - N) // 2, (img.width + N) // 2
    new_img.amPh.am[hmin:hmax, hmin:hmax] *= hann_2d
    new_img.amPh.ph[hmin:hmax, hmin:hmax] *= hann_2d

    return new_img

# -------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :], int32, int32)')
def InsertAperture_dev(img_am, img_ph, img_dim, ap_radius):
    x, y = cuda.grid(2)
    if x >= img_dim or y >= img_dim:
        return

    mid = img_dim // 2
    if (x - mid) ** 2 + (y - mid) ** 2 > ap_radius ** 2:
        img_am[y, x] = 0.0
        img_ph[y, x] = 0.0

# -------------------------------------------------------------------

def PropagateWave(img, ctf):
    fft = cc.FFT(img)
    fft.ReIm2AmPh()

    # ctf = cc.Diff2FFT(ctf)      # !!!
    ctf.ReIm2AmPh()
    ctf.MoveToCPU()
    ctf = cc.fft2diff_cpu(ctf)
    ctf.MoveToGPU()
    # ctf.ReIm2AmPh()

    fftProp = imsup.Image(img.height, img.width, imsup.Image.cmp['CAP'], imsup.Image.mem['GPU'])
    fftProp.amPh = imsup.MultAmPhMatrices(fft.amPh, ctf.amPh)

    imgProp = cc.IFFT(fftProp)
    # imgProp.ReIm2AmPh()
    imgProp.MoveToCPU()
    imgProp = imsup.re_im_2_am_ph_cpu(imgProp)      # !!!
    imgProp.defocus = img.defocus + ctf.defocus
    imgProp.px_dim = img.px_dim
    imgProp.MoveToGPU()

    ctf.ClearGPUMemory()
    fft.ClearGPUMemory()
    fftProp.ClearGPUMemory()
    img.MoveToCPU()
    imgProp.MoveToCPU()
    return imgProp

# -------------------------------------------------------------------

def PropagateToFocus(img, use_other_aberrs=True, hann_width=const.hann_win):

    if use_other_aberrs:
        ctf = calc_ctf(img.width, img.px_dim, -img.defocus, Cs=-const.Cs,
                       A1=ab.PolarComplex(const.A1_amp, const.A1_phs), df_spread=const.df_spread,
                       conv_angle=const.conv_angle, aperture=const.aperture, A1_dir=-1)
        ctf2 = mult_by_hann_window(ctf, N=hann_width)
        ctf.ClearGPUMemory()
    else:
        ctf2 = calc_ctf(img.width, img.px_dim, -img.defocus, Cs=0, A1=ab.PolarComplex(0, 0),
                        df_spread=0, conv_angle=0, aperture=0)

    return PropagateWave(img, ctf2)

# -------------------------------------------------------------------

def PropagateBackToDefocus(img, defocus, use_other_aberrs=True, hann_width=const.hann_win):

    if use_other_aberrs:
        ctf = calc_ctf(img.width, img.px_dim, defocus)
        ctf2 = mult_by_hann_window(ctf, N=hann_width)
        ctf.ClearGPUMemory()
    else:
        ctf2 = calc_ctf(img.width, img.px_dim, defocus, Cs=0, A1=ab.PolarComplex(0, 0),
                       df_spread=0, conv_angle=0, aperture=0)

    return PropagateWave(img, ctf2)

# -------------------------------------------------------------------

def PerformIWFR(images, N):
    imgHeight, imgWidth = images[0].height, images[0].width
    ewfResultsDir = 'results/ewf/'
    ewfAmName = 'am'
    ewfPhName = 'ph'
    # backCTFunctions = []
    # forwCTFunctions = []

    print('Starting IWFR...')

    # storing contrast transfer functions

    # for img in images:
    #     # print(imgWidth, img.px_dim, -img.defocus)
    #     ctf = CalcTransferFunction(imgWidth, img.px_dim, -img.defocus)
    #     ctf.AmPh2ReIm()
    #     # ctf.reIm = ccc.Diff2FFT(ctf.reIm)
    #     backCTFunctions.append(ctf)
    #
    #     ctf = CalcTransferFunction(imgWidth, img.px_dim, img.defocus)
    #     ctf.AmPh2ReIm()
    #     # ctf.reIm = ccc.Diff2FFT(ctf.reIm)
    #     forwCTFunctions.append(ctf)

    # start IWFR procedure

    exitWave = imsup.Image(imgHeight, imgWidth, imsup.Image.cmp['CRI'], imsup.Image.mem['GPU'])

    for i in range(0, N):
        print('Iteration no {0}...'.format(i+1))
        imsup.ClearImageData(exitWave)

        # backpropagation (to in-focus plane)
        ccfg.GetGPUMemoryUsed()

        for img, idx in zip(images, range(0, len(images))):
            img.MoveToGPU()
            # img = PropagateWave(img, backCTFunctions[idx])        # faster, but uses more memory
            img = PropagateToFocus(img, use_other_aberrs=True)      # slower, but uses less memory
            img.AmPh2ReIm()
            exitWave.reIm = arrsup.AddArrayToArray(exitWave.reIm, img.reIm)
            img.MoveToCPU()

        exitWave.reIm = arrsup.MultArrayByScalar(exitWave.reIm, 1/len(images))

        exitWave.MoveToCPU()
        ewfAmPath = ewfResultsDir + ewfAmName + str(i+1) + '.png'
        ewfPhPath = ewfAmPath.replace(ewfAmName, ewfPhName)
        # ewfAmplPhPath = ewfPhPath.replace(ewfPhName, ewfPhName + 'Ampl')

        imsup.SaveAmpImage(exitWave, ewfAmPath)
        imsup.SavePhaseImage(exitWave, ewfPhPath)
        # amplifExitPhase = FilterPhase(exitWave)
        # imsup.SavePhaseImage(amplifExitPhase, ewfAmplPhPath)
        # amplifExitPhase.ClearGPUMemory()
        exitWave.MoveToGPU()

        # forward propagation (to the original focus plane)
        ccfg.GetGPUMemoryUsed()

        for img, idx in zip(images, range(0, len(images))):
            img.MoveToGPU()
            # images[idx] = PropagateWave(exitWave, forwCTFunctions[idx])
            images[idx] = PropagateBackToDefocus(exitWave, img.defocus, use_other_aberrs=True)
            img.MoveToCPU()
            images[idx].amPh.am = np.copy(img.amPh.am)          # restore original amplitude

    print('All done')
    return exitWave

# -------------------------------------------------------------------

# mozna to wszystko uproscic;
# ale to filtrowanie (artefaktow) nie jest wlasciwie potrzebne
def FilterImage(img, var, stdFactor=1.0):
    if var == 'am':
        imgFiltered = FilterAmplitude(img, stdFactor)
    else:
        imgFiltered = FilterPhase(img, stdFactor)
    return imgFiltered

# -------------------------------------------------------------------

def FilterAmplitude(img, stdFactor=1.0):
    mt = img.memType
    img.MoveToCPU()
    amplifAmpImg = imsup.CopyImage(img)
    amplifAmpImg.ReIm2AmPh()
    amTemp = np.copy(amplifAmpImg.amPh.am)
    amAvg = np.average(amTemp)
    amStd = np.std(amTemp)
    amMin = amAvg - stdFactor * amStd
    amMax = amAvg + stdFactor * amStd
    amTemp2 = amTemp * (amTemp < amMax) + amMax * (amTemp > amMax)
    amplifAmpImg.amPh.am = amTemp2 * (amTemp2 > amMin) + amMin * (amTemp2 < amMin)
    amplifAmpImg.UpdateBuffer()     # !!!
    img.ChangeMemoryType(mt)
    return amplifAmpImg

# -------------------------------------------------------------------

def FilterPhase(img, stdFactor=1.0):
    mt = img.memType
    img.MoveToCPU()
    amplifPhaseImg = imsup.CopyImage(img)
    amplifPhaseImg.ReIm2AmPh()
    phTemp = np.copy(amplifPhaseImg.amPh.ph)
    phAvg = np.average(phTemp)
    phStd = np.std(phTemp)
    phMin = phAvg - stdFactor * phStd
    phMax = phAvg + stdFactor * phStd
    phTemp2 = phTemp * (phTemp < phMax) + phMax * (phTemp > phMax)
    amplifPhaseImg.amPh.ph = phTemp2 * (phTemp2 > phMin) + phMin * (phTemp2 < phMin)
    img.ChangeMemoryType(mt)
    return amplifPhaseImg
