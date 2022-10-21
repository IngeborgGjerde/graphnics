

import networkx as nx
from fenics import *
import sys
sys.path.append('../')
set_log_level(40)
from graphnics import *


def test_graph_color():
    
    # Check that a line graph gets all the same color
    G = make_line_graph(3)
    color_graph(G)
    assert G.edges()[(0,1)]['color'] == G.edges()[(1,2)]['color']

    # check that a Y bifurcation gets three different colors
    G = make_Y_bifurcation()
    color_graph(G)
    colors = nx.get_edge_attributes(G, 'color')
    assert len(set(colors.values())) == 3


    # check that 2x3 honeycomb gets 18 different colors
    
    G = honeycomb(2,3)
    color_graph(G)
    
    colors = nx.get_edge_attributes(G, 'color')
    num_colors = len(set(list(colors.values())))
    assert num_colors == 18
    
    # if something goes wrong you can plot the colors using
    # > plot_graph_color(G)