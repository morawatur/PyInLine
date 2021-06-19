import numpy as np

import Dm3Reader3 as dm3
import ImageSupport as imsup

#-------------------------------------------------------------------

def mask_fft(fft, mid, r, out=True):
    if out:
        mfft = np.copy(fft)
        mfft[mid[0] - r:mid[0] + r, mid[1] - r:mid[1] + r] = 0
    else:
        mfft = np.zeros(fft.shape, dtype=fft.dtype)
        mfft[mid[0] - r:mid[0] + r, mid[1] - r:mid[1] + r] = np.copy(fft[mid[0] - r:mid[0] + r, mid[1] - r:mid[1] + r])
    return mfft

#-------------------------------------------------------------------

def mask_fft_center(fft, r, out=True):
    mid = (fft.shape[0] // 2, fft.shape[1] // 2)
    return mask_fft(fft, mid, r, out)

#-------------------------------------------------------------------

def find_img_max(img):
    max_xy = np.array(np.unravel_index(np.argmax(img), img.shape))
    return list(max_xy)

#-------------------------------------------------------------------

def insert_aperture(img, ap_dia):
    img_ap = imsup.CopyImage(img)
    img_ap.ReIm2AmPh()
    img_ap.MoveToCPU()

    n = img_ap.width
    c = n // 2
    ap_r = ap_dia // 2
    y, x = np.ogrid[-c:n - c, -c:n - c]
    mask = x * x + y * y > ap_r * ap_r

    img_ap.amPh.am[mask] = 0.0
    img_ap.amPh.ph[mask] = 0.0
    return img_ap

#-------------------------------------------------------------------

def calc_phase_sum(img1, img2):
    img1.MoveToCPU()
    img2.MoveToCPU()

    phs_sum = imsup.ImageExp(img1.height, img1.width)
    phs_sum.amPh.am = img1.amPh.am * img2.amPh.am
    phs_sum.amPh.ph = img1.amPh.ph + img2.amPh.ph
    return phs_sum

#-------------------------------------------------------------------

def calc_phase_diff(img1, img2):
    img1.MoveToCPU()
    img2.MoveToCPU()

    phs_diff = imsup.ImageExp(img1.height, img1.width)
    phs_diff.amPh.am = img1.amPh.am * img2.amPh.am
    phs_diff.amPh.ph = img1.amPh.ph - img2.amPh.ph
    return phs_diff

#-------------------------------------------------------------------

def read_dm3_file(fpath):
    img_data, px_dims = dm3.ReadDm3File(fpath)
    imsup.Image.px_dim_default = px_dims[0]

    holo_img = imsup.ImageExp(img_data.shape[0], img_data.shape[1])
    holo_img.LoadAmpData(np.sqrt(img_data).astype(np.float32))

    return holo_img
