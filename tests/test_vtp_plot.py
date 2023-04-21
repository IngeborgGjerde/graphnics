'''
Copyright (C) 2022-2023 by Ingeborg Gjerde

This file is a part of the graphnics project (https://arxiv.org/abs/2212.02916)

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.
'''

import networkx as nx
from fenics import *
import sys

sys.path.append("../")
set_log_level(40)
from graphnics import *

# We simply test that the plot functions run, not the output


def test_vtp_plot():
    
    G = YY_bifurcation(dim=3)
    for e in G.edges():
            G.edges()[e]['radius'] = 1
            
    ffile = TubeFile(G, 'test.pvd')
    V = FunctionSpace(G.mesh, 'CG', 1)
    v = Function(V)

    # Test that we can write a function to plot file    
    v.rename('u', '0.0')
    ffile << v
    
    # Test that we can write time dependent function to plot file
    for i in range(10):
        for j, e in enumerate(G.edges()):
            G.edges()[e]['radius'] = j+i
        
        v.rename('u', '0.0')
        v.name()

        ffile << (v, i)
    
    # Remove file again
    import os
    os.remove('test.pvd') 

    import glob    
    fileList = glob.glob('*.vtp')
    for fname in fileList:
        os.remove(fname) 