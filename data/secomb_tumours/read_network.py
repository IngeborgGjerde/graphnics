import networkx as nx
import sys
import requests
import re
sys.path.append("../../")
from graphnics import *

'''
Scripts for reading vasculature graph datasets from Secomb's group
please see https://physiology.arizona.edu/people/secomb/network for information on how to cite
their dataset
'''

def read_network(url):
    '''
    Parse the dataset given at url
    
    The datasets differ a bit in file structure, but it should be straightforward to adapt this script
    to the rest of their datasets
    '''

    # Get dataset
    print(f'Downoading data from {url} \n')
    print('Please visit https://physiology.arizona.edu/people/secomb/network for information on how to cite the source for this data\n  ')
    req = requests.get(url, allow_redirects=True)
    data = req.text
        
    header = data.split('\n')[0:3] #header
    data = data.split('\n')[2:-2] # actual data

    G = nx.DiGraph()
    
    # the data is a text file formatted as a table
    # we split the data using whitespace between entries
    
    for row in data:
        row =  ' ' + row # make sure all rows are zero padded
        vals = re.split(r'\s{1,}', row) # split with variable amount of white space
        
        v1 = int(vals[3]) # vertex 1
        v2 = int(vals[4]) # vertex 2
        diam = float(vals[5]) # diameter
        
        # coordinates of vertices
        pos_v1 = [float(c) for c in vals[7:10]]
        pos_v2 = [float(c) for c in vals[10:13]]
        
        # Add to networkx graph
        G.add_edge(v1, v2)
        G.edges()[(v1, v2)]['radius'] = diam/2
        G.nodes()[v1]["pos"] = pos_v1
        G.nodes()[v2]["pos"] = pos_v2

    G = nx.convert_node_labels_to_integers(G)
    G = copy_from_nx_graph(G)
    
    return G


if __name__ == "__main__":

    url = 'https://physiology.arizona.edu/sites/default/files/rattum98_0.txt'

    # Read and plot network
    G = read_network(url)
    G.make_mesh(3)
    File("network.pvd") << Function(FunctionSpace(G.global_mesh, "CG", 1))

    # Plot a box around
    mesh = UnitCubeMesh(10, 10, 10)
    c = mesh.coordinates()

    # scale and recenter box
    pos = nx.get_node_attributes(G, "pos")
    node_coords = np.asarray(list(pos.values()))

    xmin, ymin, zmin = np.min(node_coords, axis=0)
    xmax, ymax, zmax = np.max(node_coords, axis=0)

    c[:, 0] *= (xmax - xmin) * 1.1
    c[:, 1] *= (ymax - ymin) * 1.1
    c[:, 2] *= (zmax - zmin) * 1.4
    c[:, 0] -= xmin
    c[:, 1] -= ymin
    c[:, 2] -= zmin

    File("box.pvd") << Function(FunctionSpace(mesh, "CG", 1))
