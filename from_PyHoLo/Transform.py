import numpy as np
import ImageSupport as imsup
from skimage import transform as tr
from skimage.restoration import unwrap_phase

#-------------------------------------------------------------------

def rescale_pixel_dim(px_dim, old_dim, new_dim):
    resc_factor = old_dim / new_dim
    return px_dim * resc_factor

#-------------------------------------------------------------------

def RotateImageSki(img, angle, mode='constant'):
    dt = img.cmpRepr
    img.ReIm2AmPh()

    amp_limits = [np.min(img.amPh.am), np.max(img.amPh.am)]
    phs_limits = [np.min(img.amPh.ph), np.max(img.amPh.ph)]

    if amp_limits[0] < -1.0 or amp_limits[1] > 1.0:
        amp_scaled = imsup.ScaleImage(img.amPh.am, -1.0, 1.0)
    else:
        amp_scaled = np.copy(img.amPh.am)

    if phs_limits[0] < -1.0 or phs_limits[1] > 1.0:
        phs_scaled = imsup.ScaleImage(img.amPh.ph, -1.0, 1.0)
    else:
        phs_scaled = np.copy(img.amPh.ph)

    amp_rot = tr.rotate(amp_scaled, angle, mode=mode).astype(np.float32)
    phs_rot = tr.rotate(phs_scaled, angle, mode=mode).astype(np.float32)

    if amp_limits[0] < -1.0 or amp_limits[1] > 1.0:
        amp_rot_rescaled = imsup.ScaleImage(amp_rot, amp_limits[0], amp_limits[1])
    else:
        amp_rot_rescaled = np.copy(amp_rot)

    if phs_limits[0] < -1.0 or phs_limits[1] > 1.0:
        phs_rot_rescaled = imsup.ScaleImage(phs_rot, phs_limits[0], phs_limits[1])
    else:
        phs_rot_rescaled = np.copy(phs_rot)

    img_rot = imsup.ImageExp(amp_rot.shape[0], amp_rot.shape[1], defocus=img.defocus, num=img.numInSeries, px_dim_sz=img.px_dim)
    img_rot.LoadAmpData(amp_rot_rescaled)
    img_rot.LoadPhsData(phs_rot_rescaled)
    if img.cos_phase is not None:
        img_rot.update_cos_phase()

    # resc_factor = img_rot.width / img.width
    # img_rot.px_dim *= resc_factor
    img_rot.px_dim = rescale_pixel_dim(img.px_dim, img.width, img_rot.width)

    img.ChangeComplexRepr(dt)
    img_rot.ChangeComplexRepr(dt)

    return img_rot

#-------------------------------------------------------------------

def RescaleImageSki(img, factor):
    dt = img.cmpRepr
    img.ReIm2AmPh()

    amp_limits = [np.min(img.amPh.am), np.max(img.amPh.am)]
    phs_limits = [np.min(img.amPh.ph), np.max(img.amPh.ph)]

    if amp_limits[0] < -1.0 or amp_limits[1] > 1.0:
        amp_scaled = imsup.ScaleImage(img.amPh.am, -1.0, 1.0)
    else:
        amp_scaled = np.copy(img.amPh.am)

    if phs_limits[0] < -1.0 or phs_limits[1] > 1.0:
        phs_scaled = imsup.ScaleImage(img.amPh.ph, -1.0, 1.0)
    else:
        phs_scaled = np.copy(img.amPh.ph)

    amp_mag = tr.rescale(amp_scaled, scale=factor, mode='constant').astype(np.float32)
    phs_mag = tr.rescale(phs_scaled, scale=factor, mode='constant').astype(np.float32)

    if amp_limits[0] < -1.0 or amp_limits[1] > 1.0:
        amp_mag_rescaled = imsup.ScaleImage(amp_mag, amp_limits[0], amp_limits[1])
    else:
        amp_mag_rescaled = np.copy(amp_mag)

    if phs_limits[0] < -1.0 or phs_limits[1] > 1.0:
        phs_mag_rescaled = imsup.ScaleImage(phs_mag, phs_limits[0], phs_limits[1])
    else:
        phs_mag_rescaled = np.copy(phs_mag)

    img_mag = imsup.ImageExp(amp_mag.shape[0], amp_mag.shape[1], defocus=img.defocus, num=img.numInSeries, px_dim_sz=img.px_dim)
    img_mag.LoadAmpData(amp_mag_rescaled)
    img_mag.LoadPhsData(phs_mag_rescaled)
    if img.cos_phase is not None:
        img_mag.update_cos_phase()

    # resc_factor = img_mag.width / img.width
    # img_mag.px_dim *= resc_factor
    img_mag.px_dim = rescale_pixel_dim(img.px_dim, img.width, img_mag.width)

    img.ChangeComplexRepr(dt)
    img_mag.ChangeComplexRepr(dt)

    return img_mag

#-------------------------------------------------------------------

def WarpImage(img, src_set, dst_set):
    dt = img.cmpRepr
    img.ReIm2AmPh()

    amp_limits = [np.min(img.amPh.am), np.max(img.amPh.am)]
    phs_limits = [np.min(img.amPh.ph), np.max(img.amPh.ph)]

    if amp_limits[0] < -1.0 or amp_limits[1] > 1.0:
        amp_scaled = imsup.ScaleImage(img.amPh.am, -1.0, 1.0)
    else:
        amp_scaled = np.copy(img.amPh.am)

    if phs_limits[0] < -1.0 or phs_limits[1] > 1.0:
        phs_scaled = imsup.ScaleImage(img.amPh.ph, -1.0, 1.0)
    else:
        phs_scaled = np.copy(img.amPh.ph)

    tform3 = tr.ProjectiveTransform()
    tform3.estimate(src_set, dst_set)
    amp_warp = tr.warp(amp_scaled, tform3, output_shape=amp_scaled.shape).astype(np.float32)
    phs_warp = tr.warp(phs_scaled, tform3, output_shape=amp_scaled.shape).astype(np.float32)

    if amp_limits[0] < -1.0 or amp_limits[1] > 1.0:
        amp_warp_rescaled = imsup.ScaleImage(amp_warp, amp_limits[0], amp_limits[1])
    else:
        amp_warp_rescaled = np.copy(amp_warp)

    if phs_limits[0] < -1.0 or phs_limits[1] > 1.0:
        phs_warp_rescaled = imsup.ScaleImage(phs_warp, phs_limits[0], phs_limits[1])
    else:
        phs_warp_rescaled = np.copy(phs_warp)

    img_warp = imsup.ImageExp(amp_warp.shape[0], amp_warp.shape[1], defocus=img.defocus, num=img.numInSeries)
    img_warp.LoadAmpData(amp_warp_rescaled)
    img_warp.LoadPhsData(phs_warp_rescaled)
    if img.cos_phase is not None:
        img_warp.update_cos_phase()

    img.ChangeComplexRepr(dt)
    img_warp.ChangeComplexRepr(dt)

    return img_warp

#-------------------------------------------------------------------

def DetermineCropCoordsAfterSkiRotation(oldDim, angle):
    return imsup.DetermineCropCoordsAfterRotation(oldDim, oldDim, angle)

#-------------------------------------------------------------------

class Line:
    def __init__(self, a_coeff, b_coeff):
        self.a = a_coeff
        self.b = b_coeff

    def getFromPoints(self, p1, p2):
        self.a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.b = p1[1] - self.a * p1[0]

    def getFromDirCoeffAndPoint(self, a_coeff, p1):
        self.a = a_coeff
        self.b = p1[1] - self.a * p1[0]

# -------------------------------------------------------------------

def FindPerpendicularLine(line, point):
    linePerp = Line(-1 / line.a, 0)
    linePerp.getFromDirCoeffAndPoint(linePerp.a, point)
    return linePerp

#-------------------------------------------------------------------

def FindRotationCenter(pts1, pts2):
    A1, B1 = pts1
    A2, B2 = pts2

    Am = [np.average([A1[0], A2[0]]), np.average([A1[1], A2[1]])]
    Bm = [np.average([B1[0], B2[0]]), np.average([B1[1], B2[1]])]

    aLine = Line(0, 0)
    bLine = Line(0, 0)
    aLine.getFromPoints(A1, A2)
    bLine.getFromPoints(B1, B2)

    aLinePerp = FindPerpendicularLine(aLine, Am)
    bLinePerp = FindPerpendicularLine(bLine, Bm)

    rotCenterX = (bLinePerp.b - aLinePerp.b) / (aLinePerp.a - bLinePerp.a)
    rotCenterY = aLinePerp.a * rotCenterX + aLinePerp.b

    return [rotCenterX, rotCenterY]

#-------------------------------------------------------------------

def RotatePoint(p1, angle):
    z1 = np.complex(p1[0], p1[1])
    r = np.abs(z1)
    phi = np.angle(z1) + imsup.Radians(angle)
    p2 = [r * np.cos(phi), r * np.sin(phi)]
    return p2