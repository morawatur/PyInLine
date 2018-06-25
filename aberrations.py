import numpy as np

# ---------------------------------------------------------------

def polar2complex(amp, phs):
    return amp * np.exp(1j * phs)

# ---------------------------------------------------------------

def complex2polar(x):
    return np.abs(x), np.angle(x)

# ---------------------------------------------------------------

class PolarComplex:
    def __init__(self, am, ph):
        self.amp = np.copy(am)
        self.phs = np.copy(ph)
        self.A1 = polar2complex(am, ph)

    def real(self):
        return self.A1.real

    def imag(self):
        return self.A1.imag

    def set_ri_from_ap(self):
        self.A1 = polar2complex(self.amp, self.phs)

    def set_ap_from_ri(self):
        self.amp, self.phs = complex2polar(self.A1)

    def set_re_im(self, re, im):
        self.A1 = np.complex64(re, im)
        self.set_ap_from_ri()

    def set_am_ph(self, am, ph):
        self.amp = am
        self.phs = ph
        self.set_ri_from_ap()

# ---------------------------------------------------------------

class Aberrations:
    ewf_length = 1.97e-12

    def __init__(self, C1=0, Cs=0, A1=PolarComplex(0,0), A2=PolarComplex(0,0), df_sp=0, conv_ang=0):
        self.C1 = C1
        self.Cs = Cs
        self.A1 = A1
        self.A2 = A2

        self.df_spread = df_sp
        self.conv_angle = conv_ang

    def set_C1(self, df):
        self.C1 = df

    def set_Cs(self, Cs):
        self.Cs = Cs

    def set_A1(self, a, b):
        self.A1 = PolarComplex(a, b)

    def set_A2(self, a, b):
        self.A2 = PolarComplex(a, b)

    def set_df_spread(self, df_sp):
        self.df_spread = df_sp

    def set_conv_angle(self, conv_ang):
        self.conv_angle = conv_ang

    def get_C1_cf(self):
        return np.pi * self.ewf_length * self.C1

    def get_Cs_cf(self):
        return 0.5 * np.pi * (self.ewf_length ** 3) * self.Cs

    def get_A1_cf(self):
        A1_cf_re = np.pi * self.ewf_length * self.A1.real()
        A1_cf_im = 2 * np.pi * self.ewf_length * self.A1.imag()
        return A1_cf_re + 1j * A1_cf_im

    def get_A2_cf(self):
        A2_cf_re = (2.0 / 3.0) * np.pi * (self.ewf_length ** 2) * self.A2.real()
        A2_cf_im = 0.0
        return A2_cf_re + 1j * A2_cf_im

# ---------------------------------------------------------------

class ContrastTransferFunction2D:
    def __init__(self, w, h, px, aberrs=Aberrations()):
        self.w = w
        self.h = h
        self.px = px
        amp = np.array((h, w), dtype=np.float32)
        phs = np.array((h, w), dtype=np.float32)
        self.ctf = PolarComplex(amp, phs)
        self.abset = aberrs
        self.spat_env = 0
        self.temp_env = 0
        # self.C1_fun = 0
        # self.Cs_fun = 0
        # self.A1_fun = 0

    def calc_env_funs(self):
        kx, ky = calc_kx_ky(self.w, self.px)
        k_squared = kx ** 2 + ky ** 2

        self.spat_env = np.exp(-(k_squared * (np.pi * self.abset.conv_angle) ** 2) *
                               (self.abset.C1 + self.abset.Cs * self.abset.ewf_length ** 2 * k_squared) ** 2)
        self.temp_env = np.exp(-(0.5 * np.pi * self.abset.ewf_length * self.abset.df_spread * k_squared) ** 2)

    def calc_ctf(self):
        kx, ky = calc_kx_ky(self.w, self.px)
        k_squared = kx ** 2 + ky ** 2
        k_squares_diff = kx ** 2 - ky ** 2

        C1_cf = self.abset.get_C1_cf()
        Cs_cf = self.abset.get_Cs_cf()
        A1_cf = self.abset.get_A1_cf()
        # A2_cf = self.abset.get_A2_cf()

        self.ctf.amp = self.spat_env * self.temp_env
        self.ctf.phs = C1_cf * k_squared + Cs_cf * (k_squared ** 2)
        self.ctf.phs += A1_cf.real * k_squares_diff + A1_cf.imag * kx * ky      # two-fold astigmatism
        # self.ctf.phs += A2_cf.real() * kx * (kx ** 2 - 3 * ky ** 2)           # three-fold astigmatism

    def get_ctf_sine(self):
        return -self.ctf.amp * np.sin(self.ctf.phs)

# ---------------------------------------------------------------

def calc_kx_ky(dim, px):
    rec_px = 1.0 / (dim * px)
    rec_orig = -1.0 / (2.0 * px)
    x, y = np.mgrid[0:dim:1, 0:dim:1]
    kx = rec_orig + x * rec_px
    ky = rec_orig + y * rec_px
    return kx, ky

# ---------------------------------------------------------------

class ContrastTransferFunction1D:
    def __init__(self, w, px, aberrs=Aberrations()):
        self.width = w
        self.bin = px
        amp = np.array(w, dtype=np.float32)
        phs = np.array(w, dtype=np.float32)
        self.ctf = PolarComplex(amp, phs)
        self.abset = aberrs
        self.spat_env = 0
        self.temp_env = 0
        self.kx = 0
        self.calc_spat_freqs()

    def calc_spat_freqs(self):
        rec_px_dim = 1.0 / (self.width * self.bin)
        x = np.arange(0, self.width, 1)
        self.kx = x * rec_px_dim

    def calc_env_funs(self):
        k_squared = self.kx ** 2

        self.spat_env = np.exp(-(k_squared * (np.pi * self.abset.conv_angle) ** 2) *
                               (self.abset.C1 + self.abset.Cs * self.abset.ewf_length ** 2 * k_squared) ** 2)
        self.temp_env = np.exp(-(0.5 * np.pi * self.abset.ewf_length * self.abset.df_spread * k_squared) ** 2)

    def calc_ctf(self):
        k_squared = self.kx ** 2

        C1_cf = self.abset.get_C1_cf()
        Cs_cf = self.abset.get_Cs_cf()
        A1_cf = self.abset.get_A1_cf()
        # A2_cf = self.abset.get_A2_cf()

        self.ctf.amp = self.spat_env * self.temp_env
        self.ctf.phs = C1_cf * k_squared + Cs_cf * (k_squared ** 2)
        self.ctf.phs += A1_cf.real * k_squared + A1_cf.imag * self.kx       # two-fold astigmatism
        # self.ctf.phs += A2_cf.real() * (kx ** 3)                          # three-fold astigmatism

    def get_ctf_sine(self):
        return -self.ctf.amp * np.sin(self.ctf.phs)