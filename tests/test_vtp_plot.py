
import networkx as nx
from fenics import *
import sys

sys.path.append("../")
set_log_level(40)
from graphnics import *

# We simply test that the plot functions run, not the output
def test_vtp_plot():
    
    G = make_double_Y_bifurcation(dim=3)
    for e in G.edges():
        G.edges()[e]['radius'] = 1
    
    V = FunctionSpace(G.global_mesh, 'CG', 1)
    v = Function(V)
    v.rename('u', '0.0')
    v.name()

    TubeFile(G, 'test.vtp') << v
    
    # Remove file again
    import os
    os.remove('test.vtp') 