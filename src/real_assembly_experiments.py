from pyexpat import model
import networkx as nx
from fenics import *
from fenics_graph import *
from utils import *
from graph_examples import *
    
parameters["form_compiler"]["cpp_optimize"] = True

def hydraulic_network_with_custom_assembly(G, f=Constant(0), p_bc=Constant(0)):
    '''
    Solve hydraulic network model 
        R q + d/ds p = 0
            d/ds q = f
    on graph G, with bifurcation condition q_in = q_out
    and custom assembly of the bifurcatio condition 

    Args:
        G (fg.FenicsGraph): problem domain
        f (df.function): source term
        p_bc (df.function): neumann bc for pressure
    '''
    import time    
    t_ = time.time()
    
    mesh = G.global_mesh

    # Flux spaces on each segment, ordered by the edge list
    submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
    P2s = [FunctionSpace(msh, 'CG', 3) for msh in submeshes] 
    
    # Pressure space on global mesh
    P1 = FunctionSpace(mesh, 'CG', 2) # Pressure space (on whole mesh)
    
    ### Function spaces
    spaces = P2s + [P1]
    W = MixedFunctionSpace(*spaces) 


    # Trial and test functions
    vphi = TestFunctions(W)
    qp = TrialFunctions(W)

    # split out the components
    qs = qp[0:G.num_edges]
    p = qp[-1]

    vs = vphi[0:G.num_edges]
    phi = vphi[-1]
    
    elapsed = time.time()-t_
    info = f'* Setting up spaces: {elapsed:1.3f}s' 
    with open("profiling.txt",'a') as file:
        file.write(info + '\n')
    print(info)
    t_ = time.time()

    ## Assemble variational formulation 

    # Initialize blocks in a and L to zero
    # (so fenics-mixed-dim does not throw an error)
    dx = Measure('dx', domain=mesh)
    a = Constant(0)*p*phi*dx
    L = Constant(0)*phi*dx


    # Using methodology from firedrake we assemble the jumps as a vector
    # and input the jumps in the matrix later
    vecs = [[G.jump_vector(q, ix, j) for j in G.bifurcation_ixs] for ix, q in enumerate(qs)] 
    # now we can index by vecs[branch_ix][bif_ix]
    elapsed = time.time()-t_
    info = f'* Adding jump terms to varform: {elapsed:1.3f}s' 
    with open("profiling.txt",'a') as file:
        file.write(info + '\n')
    print(info)
    t_ = time.time()
    
    # Assemble edge contributions to a and L
    for i, e in enumerate(G.edges):
        
        msh = G.edges[e]['submesh']
        vf = G.edges[e]['vf']
        #res = G.edges[e]['res']
        
        dx_edge = Measure("dx", domain = msh)
        ds_edge = Measure('ds', domain=msh, subdomain_data=vf)

        # Add variational terms defined on edge
        a += qs[i]*vs[i]*dx_edge        
        a -= p*G.dds(vs[i])*dx_edge
        a += phi*G.dds(qs[i])*dx_edge

        # Add boundary condition for inflow/outflow boundary node
        L += p_bc*vs[i]*ds_edge(BOUN_IN)
        L -= p_bc*vs[i]*ds_edge(BOUN_OUT)
    
    elapsed = time.time()-t_
    info = f'* Adding edge terms to varform: {elapsed:1.3f}s' 
    with open("profiling.txt",'a') as file:
        file.write(info + '\n')
    print(info)
    t_ = time.time()
    
    # Solve
    qp0 = mixed_dim_fenics_solve_custom(a, L, W, mesh, vecs, G)
    return qp0


import argparse
import time
import os
from models import hydraulic_network

if __name__ == '__main__':
    '''
    Do time profiling for hydraulic network model implemented with fenics-mixed-dim
    Args:
        customassembly (bool): whether to assemble real spaces separately as vectors 
    customassembly should lead to speed up
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-customassembly', help='whether to do custom assembly of real spaces', default=True, type=str)
    args = parser.parse_args()

    if args.customassembly: 
        modelfunc = hydraulic_network_with_custom_assembly
    else: 
        modelfunc = hydraulic_network
    
    
    # Clear fenics cache
    os.system('dijitso clean') 
    
    # Profile against a simple line graph with n nodes    
    n = 10
    G = make_line_graph(n)
    G.make_mesh(1) # we use just one cell per edge 
    mesh = G.global_mesh
    num_bifs = len(G.bifurcation_ixs)

    with open("profiling.txt",'w') as file:
        file.write('Profiling for hydraulic network model')
        if args.customassembly:
             file.write(' with custom assembly of reals')
        file.write(f'\n Number of bifurcations: {num_bifs} \n')
        file.write('\n With cache cleared: \n')


    # Run with cache cleared and record times
    t = time.time()
    
    p_bc = Expression('x[0]', degree=1)
    qp0 = modelfunc(G, p_bc = p_bc)
    elapsed = time.time()-t
    
    info = f'Total solver time: {elapsed:1.3f}s'
    with open("profiling.txt",'a') as file:
        file.write(info)
    
    
    # Run again without clearing cache
    with open("profiling.txt",'a') as file:
        file.write('\n \n Without cache cleared: \n')
    t = time.time()
    qp0 = modelfunc(G, p_bc = p_bc)
    elapsed = time.time()-t
    
    info = f'Total solver time: {elapsed:1.3f}s'
    with open("profiling.txt",'a') as file:
        file.write(info)
    
            
    list_timings(TimingClear.keep, [TimingType.wall])
        
    vars = qp0.split()
    p = vars[-1]
    
    q = GlobalFlux(G, vars[0:-1])
    qi = interpolate(q, VectorFunctionSpace(G.global_mesh, 'DG', 2, G.geom_dim))
    p = vars[-1]
    
    qi.rename('q', '0.0')
    p.rename('p', '0.0')
    File('plots/p.pvd')<<p
    File('plots/q.pvd')<<qi