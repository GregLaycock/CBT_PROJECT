import numpy as np
from matplotlib import pyplot as plot
from scipy.optimize import minimize as min
from scipy.signal import lti
import sympy as sp
import math
from steady_state_values import steady_state
filename = 'Fit_results.csv'
import csv

with open(filename) as p:
    # reads csv into a list of lists
    my_list = [rec for rec in csv.reader(p, delimiter=',')]

all_params = [[float(i) for i in my_list[j]] for j, lis in enumerate(my_list)]

import sympy as sp

from matplotlib import pyplot as plot
from scipy.optimize import minimize as min
from scipy.signal import lti
import sympy as sp
import math

np.set_printoptions(precision=2)
from numpy import exp
from steady_state_values import steady_state

# path = 'C:/Users/Greg/Desktop/Gregs Workshop/CBT/Project/Code/Simulation and step testing/CBT-project-CSTR-/System simulation/Steptesting and modelling/Code/fit_results.csv'
filename = 'Fit_results.csv'
import csv

with open(filename) as p:
    # reads csv into a list of lists
    my_list = [rec for rec in csv.reader(p, delimiter=',')]

all_params = [[float(i) for i in my_list[j]] for j, lis in enumerate(my_list)]


def round2SignifFigs(vals, n):
    import numpy as np
    np.set_printoptions(precision=2)
    """
    (list, int) -> numpy array
    (numpy array, int) -> numpy array

    In: a list/array of values
    Out: array of values rounded to n significant figures

    Does not accept: inf, nan, complex

    >>> m = [0.0, -1.2366e22, 1.2544444e-15, 0.001222]
    >>> round2SignifFigs(m,2)
    array([  0.00e+00,  -1.24e+22,   1.25e-15,   1.22e-03])
    """
    if np.all(np.isfinite(vals)) and np.all(np.isreal((vals))):
        eset = np.seterr(all='ignore')
        mags = 10.0 ** np.floor(np.log10(np.abs(vals)))  # omag's
        vals = np.around(vals / mags, n) * mags  # round(val/omag)*omag
        np.seterr(**eset)
        vals[np.where(np.isnan(vals))] = 0.0  # 0.0 -> nan -> 0.0
    else:
        raise IOError('Input must be real and finite')
    return vals


all_params = [round2SignifFigs(i, 2) for i in all_params]


def get_xfer(params, type):
    s = sp.Symbol('s')
    e = sp.Symbol('e')

    if type == 'FOPTD':
        k, tau, theta = params

        return k * (sp.exp(-theta * s)) / (tau * s + 1)
    elif type == 'SOPTD':
        k, tau, zeta, theta = params
        return k * sp.exp(-theta * s) / (tau ** 2 * s ** 2 + 2 * zeta * tau * s + 1)

    elif type == 'SOZPTD':
        c1, c2, tau1, tau2, theta = params
        return (c1 * s + c2) * sp.exp(-theta * s) / ((tau1 * s + 1) * (tau2 * s + 1))


Mvs = ['Ps1', 'Ps2', 'Ps3']
outputs = ['Cc_measured', 'T', 'H']
Dvs = ['Cao', 'Tbo', 'F1']

stepped_vars = ['Ps1', 'Ps2', 'Ps3', 'Cao', 'Tbo', 'F1']
outputs = ['Cc_measured', 'T', 'H']
names = []
for i, input in enumerate(stepped_vars):
    for j, output in enumerate(outputs):
        names.append(str(input) + str(output))


def get_type(name):  # based on intuition after seeing curves
    if name == 'F1T' or name == 'Ps3T' or name == 'Ps3Cc_measured':
        fit_type = 'SOPTD'

    elif name == 'Ps2T' or name == 'Ps2Cc_measured' or name == 'F1Cc_measured':
        fit_type = 'SOZPTD'

    else:
        fit_type = 'FOPTD'

    return fit_type


types = [get_type(name) for name in names]
types[15] = 'FOPTD'
funcs = [get_xfer(count, types[i]) for i, count in enumerate(all_params)]
all_funcs = dict(zip(names, funcs))

# process
ps1cc = all_funcs['Ps1Cc_measured']
ps1t = all_funcs['Ps1T']
ps1h = all_funcs['Ps1H']
ps2cc = all_funcs['Ps2Cc_measured']
ps2t = all_funcs['Ps2T']
ps2h = all_funcs['Ps2H']
ps3cc = all_funcs['Ps3Cc_measured']
ps3t = all_funcs['Ps3T']
ps3h = all_funcs['Ps3H']

Gp_sym = sp.Matrix([[ps1cc, ps2cc, ps3cc],
                    [ps1t, ps2t, ps3t],
                    [ps1h, ps2h, ps3h]])

# Disturbance
caocc = all_funcs['CaoCc_measured']
caot = all_funcs['CaoT']
caoh = all_funcs['CaoH']
tbocc = all_funcs['TboCc_measured']
tbot = all_funcs['TboT']
tboh = all_funcs['TboH']
f1cc = all_funcs['F1Cc_measured']
f1t = all_funcs['F1T']
f1h = all_funcs['F1H']

Gd_sym = sp.Matrix([[caocc, tbocc, f1cc],
                    [caot, tbot, f1t],
                    [caoh, tboh, f1h]])


from Scaling import scaling,umax,dmax,emax
import numpy as np
scale = scaling(umax,dmax,emax)
new = scale.get_scaled_xfer(Gp_sym,Gd_sym)

Gp_sym = new[0]
Gd_sym = new[1]

class system:

    def __init__(self):
        import sympy as sp
        s = sp.Symbol('s')
        self.Gp_sym = Gp_sym
        self.Gd_sym = Gd_sym
        self.umax = scale.umax
        self.dmax = scale.dmax
        self.emax = scale.emax
        self.De = scale.De
        self.Dd = scale.Dd
        self.Du = scale.Du
        self.Gp_ssgain = Gp_sym.subs(s,0)
        self.Gd_ssgain = Gd_sym.subs(s,0)
        self.ssRGA = self.Gp_ssgain.multiply_elementwise(self.Gp_ssgain.inv().T)
    def fo(var, s):
        a, b, c = var
        return (a * exp(b * s)) / (c * s + 1)

    def so(var, s):
        a, b, c, d = var
        return (a * exp(b * s)) / (c * s ** 2 + d * s + 1)

    def soz(var, s):
        a, b, c, d, e = var
        return ((a * s + b) * exp(c * s)) / ((d * s + 1) * (e * s + 1))

    def g11(s):
        return fo([-0.0138, -138, 162], s)

    def g12(s):
        return soz([-1.15, 0.000181, -187, 122, 343], s)

    def g13(s):
        return soz([-0.00284, -135, 9428.41, 132.44], s)

    def g21(s):
        return fo([-0.234, -34.5, 198], s)

    def g22(s):
        return soz([-18.6, 0.232, -113, 116, 348], s)

    def g23(s):
        return soz([-0.126, -16.2, 8190.25, 123.623], s)

    def g31(s):
        return fo([0.0112, -17.7, 506], s)

    def g32(s):
        return fo([-0.0109, -43.3, 342], s)

    def g33(s):
        return fo([6.01e-5, -136, 190], s)

    def gd11(s):
        return fo([0.384, -116.0, 191], s)

    def gd12(s):
        return fo([0.0175, -136.0, 73.4], s)

    def gd13(s):
        return fo([2640, -132, 170], s)

    def gd21(s):
        return fo([6.96, -42.5, 82.5], s)

    def gd22(s):
        return fo([0.962, -22.5, 74.4], s)

    def gd23(s):
        return so([47300, -9.03, 4502.41, 88.1694], s)

    def gd31(s):
        return fo([-0.00839, -3.57, 342.0], s)

    def gd32(s):
        return fo([-0.000376, -0.264, 290], s)

    def g33(s):
        return fo([520, -0.353, 478], s)

    def Gp(self,s):
        return np.matrix([[g11(s), g12(s), g13(s)],
                          [g21(s), g22(s), g23(s)],
                          [g31(s), g32(s), g33(s)]])

    def Gd(self,s):
        return np.matrix([[gd11(s), gd12(s), gd13(s)],
                          [gd21(s), gd22(s), gd23(s)],
                          [gd31(s), gd32(s), gd33(s)]])