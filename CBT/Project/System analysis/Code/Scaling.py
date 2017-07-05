from steady_state_values import steady_state

# description

' The following are some assumptions and descriptions used in modelling scaling for this system '

"y = [y1,y2,y3] = [Cc_measured,T,H] "
"u = [u1,u2,u3] = [Ps1,Ps2,Ps3] "
"d = [d1,d2,d3] = [Cao,Tbo,F1] "


ss = steady_state()
# Disturbances (scaled to largest  expected deviation"

d1max = 0.1 * 7.4  #kmol/m^3
d2max = 5          # deg C
d3max = 0.1* 7.334e-4 # m^3/s
dmax = [d1max,d2max,d3max]
# Manipulated variables

uallmax = 40 # same for all valves in Kpa (pressure signal to valves)
umax = [uallmax,uallmax,uallmax]
# errors
e1max = 0.1 * ss['Cc'] #kmol/m3  assumed that downstream separation will allow for high throughput as long as production rate is reasonable (10% variation is acceptable)
e2max = 5        # 5 degree Celcius freedom in temperature is acceptable to prevent runaway reaction
e3max = 0.25 *ss['H'] #m  # level is not important as long as reactor does not run dry or lose too much volume as rapid heating would occur
emax = [e1max,e2max,e3max]

class scaling:

    def __init__(self,umax,dmax,emax):
        import numpy as np
        import sympy as sp
        self.umax = umax
        self.dmax = dmax
        self.emax = emax
        self.Dd = np.matrix(np.eye(len(dmax)) * self.dmax)
        self.De = np.matrix(np.eye(len(emax)) * self.emax)
        self.Du = np.matrix(np.eye(len(umax)) * self.umax)

    def get_scaled_xfer(self,G_sym,Gd_sym):
        G_sym1 = self.De.I * G_sym * self.Du
        Gd_sym1 = self.De.I * Gd_sym * self.Dd
        return G_sym1,Gd_sym1
