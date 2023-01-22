
import networkx as nx
from fenics import *
import sys

sys.path.append("../")
set_log_level(40)
from graphnics import *

# We simply test that the plot functions run, not the output
def test_vtp_plot():
    
    G = make_double_Y_bifurcation(dim=3)
    G.make_mesh(3)

    f = Expression('x[1]+0.1*x[0]', degree=2)
    radius = Expression('x[1]+x[0]', degree=2)

    write_vtp(G, functions=[(f, 'f'), (radius, 'radius')], fname='test.vtp')
    
    # Remove file again
    import os
    os.remove('test.vtp') 