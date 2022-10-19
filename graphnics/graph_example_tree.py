import networkx as nx
import numpy as np
import networkgen.geometrytools  as geometrytools
import networkgen.generationtools as generationtools
import random

### Creation of an tree-like network ###
# This is an adaption of code created by Alexandra Vallet (Simula Research Laboratory)
# See https://gitlab.com/ValletAlexandra/NetworkGen for the original

'''
Murray's law 
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
'''

def make_arterial_tree(N, uniform_lengths = False):
    '''
    N (int): number of levels in the arterial tree
    uniform_lengths (bool): if true we have same length of all branches
    
    uniform lengths is only of interest in numerical tests
    '''
    
    # Parameters
    # Origin location
    p0=[0,0,0]

    # Initial direction
    direction=[0,1,0]

    # First vessel diameter
    D0=1

    # lambda
    lmbda=8

    # gamma
    gam=0.8

    # By convention we chose gam <=1 so D1 will always be smaller or equal to D2
    if gam > 1:
        raise Exception('Please choose a value for gamma lower or equal to 1')


    #Surface normal function
    #The surface normal here is fixed because we want to stay in the x,y plane. 
    #But this could be the normal of any surface.
    def normal (x,y,z) :
        return [0,0,1]

    #### Creation of the graph

    # Create a networkx graph 
    G=nx.DiGraph()

    # Create the first vessel
    #########################
    L=D0*lmbda
    

    G.add_edge(0,1)

    nx.set_node_attributes(G, p0, "pos")
    nx.set_edge_attributes(G, D0/2, "radius")


    G.nodes[1]['pos']=geometrytools.translation(p0,direction,L)
    inode=1

    #### Iteration to create the other vessels following a given law

    #list of the vessels from the previous generation
    previous_edges=[(0,1)]

    for igen in range(1,N):
        current_edges=[]
        for e in previous_edges : 
            # Parent vessel properties
            previousvessel=[G.nodes[e[0]]['pos'],G.nodes[e[1]]['pos']]
            D0=G.edges[e]['radius']*2

            # Daughter diameters
            D2=D0*(gam**3+1)**(-1/3)
            D1=gam*D2
            # Daughter lengths
            L2=lmbda*D2
            L1=lmbda*D1
            
            if uniform_lengths: 
                L1, L2 = L, L
            # Bifurcation angles
            # angle for the smallest vessel
            cos1= (D0**4 +D1**4 -(D0**3 - D1**3)**(4/3))/(2*D0**2*D1**2)
            angle1= np.degrees(np.arccos(cos1))
            # angle for the biggest vessel
            cos2=(D0**4 +D2**4 -(D0**3 - D2**3)**(4/3))/(2*D0**2*D2**2)
            angle2=np.degrees(np.arccos(cos2))

            #randomly chose which vessel go to the right/left
            sign1=random.choice((-1, 1))
            sign2=-1*sign1

            inode+=1
            new_edge=(e[1],inode)
            G.add_edge(*new_edge)

            # Set the location according to length and angle
            G.nodes[inode]['pos']=generationtools.compute_vessel_endpoint (previousvessel, normal(*previousvessel[1]),sign1*angle1,L1)

            # Set radius
            G.edges[new_edge]['radius']=D1/2

            # Add to the pool of vessels for this generation
            current_edges.append(new_edge)

            inode+=1
            new_edge=(e[1],inode)
            G.add_edge(*new_edge)

            # Set the location according to length and angle
            G.nodes[inode]['pos']=generationtools.compute_vessel_endpoint (previousvessel, normal(*previousvessel[1]),sign2*angle2,L2)

            # Set radius
            G.edges[new_edge]['radius']=D2/2

            # Add to the pool of vessels for this generation
            current_edges.append(new_edge)
        previous_edges=current_edges

    return G

