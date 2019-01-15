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
    ctf = imsup.ImageExp(img_dim, img_dim, imsup.Image.cmp['CAP'], imsup.Image.mem['GPU'])
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

    # ctf = cc.Diff2FFT(ctf)
    ctf.ReIm2AmPh()
    ctf.MoveToCPU()
    ctf = cc.fft2diff_cpu(ctf)
    ctf.MoveToGPU()
    ctf.ReIm2AmPh()

    fftProp = imsup.ImageExp(img.height, img.width, imsup.Image.cmp['CAP'], imsup.Image.mem['GPU'])
    fftProp.amPh = imsup.MultAmPhMatrices(fft.amPh, ctf.amPh)
    imgProp = cc.IFFT(fftProp)

    # normalization
    norm_factor = 1.0 / (img.height * img.width)
    imgProp.AmPh2ReIm()
    imgProp.MoveToCPU()
    imgProp.reIm *= norm_factor

    imgProp.ReIm2AmPh()
    # imgProp.MoveToCPU()
    # imgProp = imsup.re_im_2_am_ph_cpu(imgProp)      # !!!
    imgProp.defocus = img.defocus + ctf.defocus
    imgProp.px_dim = img.px_dim

    ctf.ClearGPUMemory()
    fft.ClearGPUMemory()
    fftProp.ClearGPUMemory()
    img.MoveToCPU()
    imgProp.MoveToCPU()
    return imgProp

# -------------------------------------------------------------------

def PropagateToFocus(img, use_other_aberrs=True, aper=const.aperture, hann_width=const.hann_win):
    if use_other_aberrs:
        ctf = calc_ctf(img.width, img.px_dim, -img.defocus, Cs=-const.Cs,
                       A1=ab.PolarComplex(const.A1_amp, const.A1_phs), df_spread=const.df_spread,
                       conv_angle=const.conv_angle, aperture=aper, A1_dir=-1)
        ctf2 = mult_by_hann_window(ctf, N=hann_width)
        ctf.ClearGPUMemory()
    else:
        ctf2 = calc_ctf(img.width, img.px_dim, -img.defocus, Cs=0, A1=ab.PolarComplex(0, 0),
                        df_spread=0, conv_angle=0, aperture=0, A1_dir=-1)

    # proba uzycia FFT obrazu jako fazy funkcji przenoszenia kontrastu
    # fft = cc.FFT(img)
    # fft.ReIm2AmPh()
    # fft.MoveToCPU()
    # fft = cc.fft2diff_cpu(fft)
    # ctf2 = imsup.ImageExp(img.height, img.width, fft.cmpRepr, fft.memType)
    # ctf2.amPh.am = np.copy(np.ones(ctf2.amPh.am.shape, dtype=np.float32))
    # ctf2.amPh.ph = np.copy(-fft.amPh.am)

    return PropagateWave(img, ctf2)

# -------------------------------------------------------------------

def PropagateBackToDefocus(img, defocus, use_other_aberrs=True, aper=const.aperture, hann_width=const.hann_win):
    if use_other_aberrs:
        ctf = calc_ctf(img.width, img.px_dim, defocus, Cs=const.Cs,
                       A1=ab.PolarComplex(const.A1_amp, const.A1_phs), df_spread=const.df_spread,
                       conv_angle=const.conv_angle, aperture=aper, A1_dir=1)
        ctf2 = mult_by_hann_window(ctf, N=hann_width)
        ctf.ClearGPUMemory()
    else:
        ctf2 = calc_ctf(img.width, img.px_dim, defocus, Cs=0, A1=ab.PolarComplex(0, 0),
                        df_spread=0, conv_angle=0, aperture=0, A1_dir=1)

    return PropagateWave(img, ctf2)

# -------------------------------------------------------------------

def run_backprop_iter(imgs_to_ewr, use_aberrs=False, ap=const.aperture, hann=const.hann_win):
    n_imgs = len(imgs_to_ewr)
    img_w, img_h = imgs_to_ewr[0].width, imgs_to_ewr[0].height
    # exit_wave = imsup.ImageExp(img_h, img_w, imsup.Image.cmp['CRI'], imsup.Image.mem['GPU'], px_dim_sz=imgs_to_ewr[0].px_dim)
    exit_wave = imsup.ImageExp(img_h, img_w, imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'], px_dim_sz=imgs_to_ewr[0].px_dim)

    for img, idx in zip(imgs_to_ewr, range(0, n_imgs)):
        img.MoveToGPU()
        img = PropagateToFocus(img, use_other_aberrs=use_aberrs, aper=ap, hann_width=hann)
        # img.AmPh2ReIm()
        img.ReIm2AmPh()             # !!!
        img.MoveToCPU()             # !!!
        # exit_wave.reIm = arrsup.AddArrayToArray(exit_wave.reIm, img.reIm)
        # exit_wave.reIm = arrsup.AddTwoArrays(exit_wave.reIm, img.reIm)
        exit_wave.amPh.am += img.amPh.am
        exit_wave.amPh.ph += img.amPh.ph

    # exit_wave.reIm = arrsup.MultArrayByScalar(exit_wave.reIm, 1 / n_imgs)
    exit_wave.MoveToCPU()           # !!!
    exit_wave.amPh.am /= n_imgs     # !!!
    exit_wave.amPh.ph /= n_imgs     # !!!
    exit_wave.MoveToGPU()           # !!!
    exit_wave.AmPh2ReIm()           # !!!
    return exit_wave

# -------------------------------------------------------------------

def run_forwprop_iter(exit_wave, imgs_to_ewr, use_aberrs=False, ap=const.aperture, hann=const.hann_win):
    n_imgs = len(imgs_to_ewr)

    tot_error = 0.0
    for img, idx in zip(imgs_to_ewr, range(0, n_imgs)):
        imgs_to_ewr[idx] = PropagateBackToDefocus(exit_wave, img.defocus, use_other_aberrs=use_aberrs, aper=ap, hann_width=hann)
        img.MoveToCPU()
        sse = calc_sum_squared_error(img.amPh.am, imgs_to_ewr[idx].amPh.am)
        print('Pair {0}{1}: SSE = {2:.3f} %'.format('0' if idx < 9 else '', idx+1, sse * 100))
        tot_error += sse
        imgs_to_ewr[idx].amPh.am = np.copy(img.amPh.am)  # restore original amplitude

    tot_error /= n_imgs
    # print('Total error = {0:.3f}%'.format(tot_error * 100))
    return tot_error

# -------------------------------------------------------------------

def calc_sum_squared_error(arr_ref, arr):
    denom = np.sum(arr_ref)
    # print(np.sum(arr_ref), np.sum(arr))
    errors = np.power(np.sqrt(arr_ref) - np.sqrt(arr), 2)
    sse = np.sum(errors) / denom
    return sse

# -------------------------------------------------------------------

def run_iteration_of_iwfr(imgs_to_ewr, use_aberrs=False, ap=const.aperture, hann=const.hann_win):
    # n_imgs = len(imgs_to_ewr)
    # img_w, img_h = imgs_to_ewr[0].width, imgs_to_ewr[0].height
    # exit_wave = imsup.ImageExp(img_h, img_w, imsup.Image.cmp['CRI'], imsup.Image.mem['GPU'])
    #
    # for img, idx in zip(imgs_to_ewr, range(0, n_imgs)):
    #     img.MoveToGPU()
    #     img = PropagateToFocus(img, use_other_aberrs=use_aberrs, aper=ap, hann_width=hann)
    #     img.AmPh2ReIm()
    #     exit_wave.reIm = arrsup.AddArrayToArray(exit_wave.reIm, img.reIm)
    #     # exit_wave.reIm = arrsup.AddTwoArrays(exit_wave.reIm, img.reIm)
    #
    # exit_wave.reIm = arrsup.MultArrayByScalar(exit_wave.reIm, 1 / n_imgs)
    #
    # for img, idx in zip(imgs_to_ewr, range(0, n_imgs)):
    #     imgs_to_ewr[idx] = PropagateBackToDefocus(exit_wave, img.defocus, use_other_aberrs=use_aberrs, aper=ap, hann_width=hann)
    #     # print(imgs_to_ewr[idx].memType, img.memType)
    #     img.MoveToCPU()
    #     # print(imgs_to_ewr[idx].memType, img.memType)
    #     imgs_to_ewr[idx].amPh.am = np.copy(img.amPh.am)  # restore original amplitude

    exit_wave = run_backprop_iter(imgs_to_ewr, use_aberrs, ap, hann)
    run_forwprop_iter(imgs_to_ewr, exit_wave, use_aberrs, ap, hann)

    return imgs_to_ewr, exit_wave

# -------------------------------------------------------------------

def run_iwfr(imgs_to_iwfr, n_iters):
    ewf_dir = 'results/ewf/'
    amp_name = 'amp'
    phs_name = 'phs'
    amp_path_base = '{0}{1}_00.png'.format(ewf_dir, amp_name)
    phs_path_base = '{0}{1}_00.png'.format(ewf_dir, phs_name)

    print('Starting IWFR...')
    exit_wave = imsup.copy_am_ph_image(imgs_to_iwfr[0])

    for i in range(0, n_iters):
        print('Iteration no {0}...'.format(i+1))
        imgs_to_iwfr, exit_wave = run_iteration_of_iwfr(imgs_to_iwfr)
        ewf_amp_path = amp_path_base.replace('00', '0{0}'.format(i+1) if i < 10 else '{0}'.format(i+1))
        ewf_phs_path = phs_path_base.replace('00', '0{0}'.format(i+1) if i < 10 else '{0}'.format(i+1))
        imsup.SaveAmpImage(exit_wave, ewf_amp_path)
        imsup.SavePhaseImage(exit_wave, ewf_phs_path)
        ccfg.GetGPUMemoryUsed()

    print('All done')
    return exit_wave

# -------------------------------------------------------------------

def simulate_images(exit_wave, df1, df2=None, df3=None, use_aberrs=True, A1_amp=0.0, A1_phs=0.0, aper=const.aperture):
    if df2 is None or df3 is None:
        df2 = df1 + 1
        df3 = 2

    sim_imgs = imsup.ImageList()

    if use_aberrs:
        const.A1_amp = A1_amp       # !!!
        const.A1_phs = A1_phs       # !!!
        hann_win = 2.2 * aper
    else:
        hann_win = 0

    for df in cc.frange(df1, df2, df3):
        print('Sim. {0:.2f} nm'.format(df * 1e9))
        img = PropagateBackToDefocus(exit_wave, df, use_aberrs, aper, hann_win)
        img.MoveToCPU()
        img.defocus = df
        sim_imgs.append(img)

    return sim_imgs

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
