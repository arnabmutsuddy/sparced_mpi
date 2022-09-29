import sys
import os

cd = os.getcwd()
wd = os.path.dirname(cd)
sys.path.append(os.path.join(wd,'bin'))

import libsbml
import importlib
import amici
import numpy as np
import re
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from modules.RunSPARCED import RunSPARCED

mpl.rcParams['figure.dpi'] = 120

#%%
sbml_file = "SPARCED.xml"
model_name= sbml_file[0:-4]
model_output_dir = model_name
sys.path.insert(0, os.path.join(wd,model_output_dir))
model_module = importlib.import_module(model_name)
model = model_module.getModel()

species_all = list(model.getStateIds())

solver = model.getSolver()          # Create solver instance
solver.setMaxSteps = 1e10

ts = 30

model.setTimepoints(np.linspace(0,ts))

#%% define stimulations
STIMligs = [1000.0,100.0,100.0,100.0,100.0,100.0,1721.0] # EGF, Her, HGF, PDGF, FGF, IGF, INS

STIMligs_id = ['E', 'H', 'HGF', 'P', 'F', 'I', 'INS']



spIn2 = np.array(model_module.getModel().getInitialStates())

for s,sp in enumerate(STIMligs_id):
    spIn2[species_all.index(sp)] = STIMligs[s]


model.setInitialStates(spIn2)

#%% hybrid deterministic simulation

flagD = 1

th = 72

xoutS_all, xoutG_all, tout_all = RunSPARCED(flagD,th,spIn2,[],sbml_file,model)

#%% hybrid stochastic simulation

flagD = 0

th = 72

xoutS_all2, xoutG_all2, tout_all2 = RunSPARCED(flagD,th,spIn2,[],sbml_file,model)

#%% plot species trajectories

def timecourse(xoutS,sp,species_all):
    x_t = xoutS[:,species_all.index(sp)]
    tout = np.array(range(0,len(x_t)))*30/3600
    
    plt.plot(tout,x_t)
    plt.ylim(0,max(x_t)*1.25)
    plt.xlabel('Time (hours)')
    plt.ylabel(str(sp))
    plt.show()
    
#%%

timecourse(xoutS_all2,'ppERK',species_all)



#%%