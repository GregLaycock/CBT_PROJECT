import csv
from matplotlib import pyplot as plot
import numpy
from Fitting_curves import get_type
from steady_state_values import steady_state
import Fitting_module

filename = 'fit_results.csv'
with open(filename, 'rU') as p:
    #reads csv into a list of lists
    my_list = [rec for rec in csv.reader(p, delimiter=',')]

all_params = [[float(i) for i in my_list[j]] for j,lis in enumerate(my_list)]

from Stepping_all import run_sim,get_results

def round2SignifFigs(vals,n):
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
        mags = 10.0**np.floor(np.log10(np.abs(vals)))  # omag's
        vals = np.around(vals/mags,n)*mags             # round(val/omag)*omag
        np.seterr(**eset)
        vals[np.where(np.isnan(vals))] = 0.0           # 0.0 -> nan -> 0.0
    else:
        raise IOError('Input must be real and finite')
    return vals

#all_params = [round2SignifFigs(i,2) for i in all_params]

tspan = numpy.linspace(0, 2000, 1000)
output_vars = ['Cc','T','H','Cc','T','H','Cc','T','H','Cc','T','H','Cc','T','H','Cc','T','H']
stepped_vars = ['Ps1','Ps2','Ps3','Cao','Tbo','F1']
outputs = ['Cc', 'T', 'H']
names = []
stepped = ['Ps1','Ps1','Ps1','Ps2','Ps2','Ps2','Ps3','Ps3','Ps3','Cao','Cao','Cao','Tbo','Tbo','Tbo','F1','F1','F1']
for i, input in enumerate(stepped_vars):
    for j, output in enumerate(outputs):
        names.append(str(input) + str(output))
# print(names)
data = get_results()
ss_values = steady_state()
tss = ss_values['T']
ccss = ss_values['Cc']
hss = ss_values['H']
types = [get_type(name) for name in names]
# print(types)
u_vals = [20,20,20,20,20,20,20,20,20,0.2*7.4,0.2*7.4,0.2*7.4,0.2*24,0.2*24,0.2*24,0.2*7.334e-4,0.2*7.334e-4,0.2*7.334e-4]
yo_vals = [ccss,tss,hss,ccss,tss,hss,ccss,tss,hss,ccss,tss,hss,ccss,tss,hss,ccss,tss,hss]
simdata = data

# getting data from fits
fitted = []

for i,fit in enumerate(names):
    parameters = all_params[i]
    model = types[i]
    u = u_vals[i]
    t = tspan
    yo = yo_vals[i]
    vals = Fitting_module.get_model_vals(parameters,model,u,t,yo)
    fitted.append(vals)


for i, name in enumerate(names):
    plot.figure()
    plot.plot(tspan,simdata[name],'b-',label = 'simulated')
    plot.plot(tspan,fitted[i],'r-',label='fitted')
    plot.title("Results of fitting a "+str(types[i]) + " model to step response of "+str(output_vars[i])+" to "+str(stepped[i]))
    plot.axis()
    plot.legend(loc=4)
    plot.xlabel('Time in sec')
    plot.ylabel('value of output in SI units')

plot.show()