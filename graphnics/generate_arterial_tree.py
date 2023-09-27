import networkx as nx
import numpy as np
import random
import copy
from graphnics import *

"""
Creation of a tree-like network
This is an adaption of code created by Alexandra Vallet (University of Oslo)
See https://gitlab.com/ValletAlexandra/NetworkGen for the original


Background: Murray's law 
We consider that the sum of power 3 of daughter vessels radii equal to the power three of the parent vessel radius
D0^3=D1^3 + D2^3
We consider that the ratio between two daughter vessels radii is gam
D2/D1=gam
Then we have
D2=D0(gam^3+1)^(-1/3)
D1=gam D2

We consider that the length L of a vessel segment in a given network is related to its diameter d via L = λd, 
here the positive constant λ is network-specific

The angle of bifurcation can be expressed based on blood volume conservation and minimum energy hypothesis 
FUNG, Y. (1997). Biomechanics: Circulation. 2nd edition. Heidelberg: Springer.
HUMPHREY, J. and DELANGE, S. (2004). An Introduction to Biomechanics. Heidelberg: Springer.
VOGEL, S. (1992). Vital Circuits. Oxford: Oxford University Press.
See here for derivation : https://www.montogue.com/blog/murrays-law-and-arterial-bifurcations/
Then it depends on the diameters 
"""


def make_arterial_tree(
    N, radius0=1, gam=0.8, L0=10, directions=False, uniform_lengths=False
):
    """
    N (int): number of levels in the arterial tree
    radius0 (float): radius of first vessel
    gam (float): ratio between daughter vessel radii
    directions (list): vector of choices (+-1) of vessel direction. If no vector is given this is assigned randomly.
    uniform_lengths (bool): uniform branch length
    
    Uniform lengths is typically only of interest in numerical tests
    Assigning directions is useful for reproducability of results.
    """

    # Parameters
    # Origin location
    p0 = [0, 0, 0]

    # Initial direction
    direction = [0, 1, 0]

    # First vessel diameter
    D0 = 2 * radius0

    # By convention we chose gam <=1 so D1 will always be smaller or equal to D2
    if gam > 1:
        raise Exception("Please choose a value for gamma lower or equal to 1")

    # Surface normal function
    # The surface normal here is fixed because we want to stay in the x,y plane.
    # But this could be the normal of any surface.
    def normal(x, y, z):
        return [0, 0, 1]

    #### Creation of the graph

    # Create a networkx graph
    G = nx.DiGraph()

    # Create the first vessel
    #########################
    L = L0
    lmbda = L/(D0)

    G.add_edge(0, 1)

    nx.set_node_attributes(G, p0, "pos")
    nx.set_edge_attributes(G, D0 / 2, "radius")

    G.nodes[1]["pos"] = np.asarray(p0) + np.asarray(direction)*L
    new_node_ix = 1

    #### Iteration to create the other vessels following a given law

    # list of the vessels from the previous generation
    previous_edges = [(0, 1)]

    for igen in range(1, N):
        current_edges = []
        for e in previous_edges:
            # Parent vessel properties
            previousvessel = [G.nodes[e[0]]["pos"], G.nodes[e[1]]["pos"]]
            D0 = G.edges[e]["radius"] * 2

            # Daughter diameters
            D2 = D0 * (gam**3 + 1) ** (-1 / 3)
            D1 = gam * D2
            # Daughter lengths
            L2 = lmbda * D2
            L1 = lmbda * 0.6 * D1

    

            if uniform_lengths:
                L1, L2 = L, L
            # Bifurcation angles
            # angle for the smallest vessel
            cos1 = (D0**4 + D1**4 - (D0**3 - D1**3) ** (4 / 3)) / (
                2 * D0**2 * D1**2
            )
            angle1 = np.degrees(np.arccos(cos1))
            # angle for the biggest vessel
            cos2 = (D0**4 + D2**4 - (D0**3 - D2**3) ** (4 / 3)) / (
                2 * D0**2 * D2**2
            )
            angle2 = np.degrees(np.arccos(cos2))

            # direction-vector choose which vessel go to the right/left
            
            if not directions:
                sign1 = random.choice((-1, 1))
            else:
                sign1 = directions[0]
                del directions[0]
            sign2 = -1 * sign1

            branch1 = [sign1, angle1, L1, D1]
            branch2 = [sign2, angle2, L2, D2]

            parent_edge_v2 = e[1]  # vertex we want to connect to

            # Check that the new edge does not overlap other edges
            for sign, angle, L, D in [branch1, branch2]:
                new_node_pos = compute_vessel_endpoint(
                    previousvessel, normal(*previousvessel[1]), sign * angle, L
                )

                all_pos = np.asarray(list(nx.get_node_attributes(G, "pos").values()))
                intersecting_lines = 0
                C = Point(all_pos[parent_edge_v2])
                DD = Point(new_node_pos)

                non_neighbor_edges = copy.deepcopy(list(G.edges()))

                neighbor_edges = copy.deepcopy(
                    list(G.in_edges(parent_edge_v2)) + list(G.out_edges(parent_edge_v2))
                )

                for en in list(set(neighbor_edges)):
                    non_neighbor_edges.remove(en)

                for v1, v2 in non_neighbor_edges:
                    A = Point(all_pos[v1])
                    B = Point(all_pos[v2])
                    if doIntersect(A, B, C, DD):
                        intersecting_lines += 1


                #if no edges overlap with this new one
                # we go ahead and add it
                if intersecting_lines < 1:
                    new_node_ix += 1

                    new_edge = (e[1], new_node_ix)
                    G.add_edge(*new_edge)

                    # Set the location according to length and angle
                    G.nodes[new_node_ix]["pos"] = new_node_pos

                    # Set radius
                    G.edges[new_edge]["radius"] = D / 2

                    # Add to the pool of vessels for this generation
                    current_edges.append(new_edge)

        previous_edges = current_edges
        
    
    # Convert to FenicsGraph
    G_ = nx.convert_node_labels_to_integers(G)

    nodes = G_.nodes()
    for n in G_.nodes:
        nodes[n]['pos'] = nodes[n]['pos'][:2]
    
    G = copy_from_nx_graph(G_)
    G.make_mesh(1)

    return G


## Helper functions from NetworkGen

def orientation(p, q, r):
    '''
    to find the orientation of an ordered triplet (p,q,r)
    function returns the following values:
    0 : Collinear points
    1 : Clockwise points
    2 : Counterclockwise

    See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    for details of below formula.
    '''

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if val > 0:

        # Clockwise orientation
        return 1
    elif val < 0:

        # Counterclockwise orientation
        return 2
    else:

        # Collinear orientation
        return 0


# This code is contributed by Ansh Riyal
def doIntersect(p1, q1, p2, q2):

    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and onSegment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and onSegment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and onSegment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and onSegment(p2, q1, q2):
        return True

    # If none of the cases
    return False



def translation(p0,direction,length) :
    #normalise the direction vector
    direction=normalize(direction)

    # compute the location of the end of the new vessel
    pnew=[(p0[i]+direction[i]*length) for i in range(len(p0))]
    return pnew

def rotate_in_plane(x,n,angle):
    """ angle : rotation degree """

    rotation_radians = np.radians(angle)

    rotation_axis = np.array(n)

    rotation_vector = rotation_radians * rotation_axis

    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_rotvec(rotation_vector)

    rotated_vec = rotation.apply(x)

    return rotated_vec


def normalize(x):
    return [x[i] / np.linalg.norm(x) for i in range(len(x))]

def project_onto_plane(x, n):
    d = np.dot(x, n) / np.linalg.norm(n)
    p = [d * normalize(n)[i] for i in range(len(n))]
    return [x[i] - p[i] for i in range(len(x))]


def compute_vessel_endpoint (previousvessel, surfacenormal,angle,length) :
    """ From a previous vessel defined in 3D, the brain surface at the end of the previous vessel, angle and length : 
        compute the coordinate of the end node of the current vessel """

    # project the previous vessel in the current plane
    # and get the direction vector of the previous vessel 
    pm1=previousvessel[0]
    p0=previousvessel[1]

    vector_previous=[p0[i]-pm1[i] for i in range(len(pm1))]

    previousdir = project_onto_plane(vector_previous, surfacenormal)


    # compute the direction vector of the new vessel with the angle
    newdir=rotate_in_plane(previousdir,surfacenormal,angle)

    # compute the location of the end of the new vessel
    pnew=translation(p0,newdir,length)


    return pnew




## Functions for checking if lines intersect
# Shamelessly copied from stackoverflow

class Point:
    def __init__(self, xx):
        self.x = xx[0]
        self.y = xx[1]
# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if (
        (q.x <= max(p.x, r.x))
        and (q.x >= min(p.x, r.x))
        and (q.y <= max(p.y, r.y))
        and (q.y >= min(p.y, r.y))
    ):
        return True
    return False

