
from cmath import e
import networkx as nx
from fenics import *
from fenics_graph import *
from utils import *


def dds(f):
    return dot(grad(f), G.global_tangent)


def hydraulic_network_model(G, inlets=[], outlets=[]):

    # Make list of edges
    edges= list(nx.get_edge_attributes(G, 'submesh').keys())
    # and list of submeshes
    submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
    
    mesh = G.global_mesh

    # Function spaces
    P2s = [FunctionSpace(msh, 'CG', 2) for msh in submeshes] # Flux spaces in each segment
    P1 = FunctionSpace(mesh, 'CG', 1) # Pressure space (on whole mesh)
    LM = FunctionSpace(mesh, 'R', 0) # Lagrange multiplier (to impose bifurcation conditions)

    spaces = P2s; spaces.append(LM); spaces.append(P1)
    W = MixedFunctionSpace(*spaces) 


    # Trial and test functions
    vphi = TestFunctions(W)
    qp = TrialFunctions(W)


    dx = Measure('dx', domain=mesh)
    dx_ = [Measure("dx", domain = msh) for msh in submeshes]
    

    ## Assemble variational formulation 

    # Initialize a and L to zero
    a = Constant(0)*qp[-2]*vphi[-2]*dx 
    L = Constant(0)*vphi[-1]*dx + Constant(0)*vphi[-2]*dx

    # Add in branch contributions
    for i in range(0, len(edges)):
        a += qp[i]*vphi[i]*dx_[i]
        a -= qp[-1]*dds(vphi[i])*dx_[i]
        a += vphi[-1]*dds(qp[i])*dx_[i]


    # Add in vertex contributions
    for branch_ix, e in enumerate(edges):
        msh = G.edges[e]['submesh']
        vf = G.edges[e]['vf']

        ds_branch = Measure('ds', domain=msh, subdomain_data=vf)

        # Add bifurcation condition 
        a += qp[branch_ix]*vphi[-2]*ds_branch(BIF_IN) - qp[branch_ix]*vphi[-2]*ds_branch(BIF_OUT)
        a += vphi[branch_ix]*qp[-2]*ds_branch(BIF_IN) - vphi[branch_ix]*qp[-2]*ds_branch(BIF_OUT)
        
        # Add boundary condition for inflow/outflow boundary node
        for inlet_tag in inlets:
            L += vphi[branch_ix]*ds_branch(inlet_tag) # (inflow boundary node has edge pointing out of it)
        for outlet_tag in outlets:
            L -= vphi[branch_ix]*ds_branch(outlet_tag) # (outflow boundary node has edge pointing into it)

    # Solve
    qp0 = mixed_dim_fenics_solve(a, L, W, mesh)
    return qp0


if __name__ == '__main__':

    # Make simple graph
    from graph_examples import *
    #G = make_double_Y_bifurcation()
    
    G = honeycomb(1,1)
    G.make_mesh(4)
    qp0 = hydraulic_network_model(G, inlets=[BOUN_IN])

    vars = qp0.split()
    p = vars[-1]
    lam = vars[-2]


    q = GlobalFlux(G, vars[0:-2])
    qi = interpolate(q, VectorFunctionSpace(G.global_mesh, 'DG', 2, 2))
    p = vars[-1]
    lam = vars[-2]
    
    qi.rename('q', '0.0')
    p.rename('p', '0.0')
    File('plots/p.pvd')<<p
    File('plots/q.pvd')<<qi
    G.global_tangent.rename('tau', '0.0')
    File('plots/tangent.pvd')<<G.global_tangent

    






    # Port to scipy 
    #from scipy.sparse import coo_matrix, bmat
    #A_arrays_flattened = [coo_matrix(block.array()) for block in A_list]
    #A = bmat([A_arrays_flattened[len(spaces)*i:len(spaces)*(i+1)] for i in range(0, len(spaces))])

    #b = np.concatenate( [block.get_local() for block in rhs_blocks] )
    

    # Input bifurcation-jump contribution
    #c = mesh.coordinates()
    #for b_ix in G.bifurcation_ixs:
    #    vertex_ix = np.where((c == c[1,:]).all(axis=1))[0][0]

        #for e in G.edges():

    #scipy.sparse.linalg.spsolve(A, b, permc_spec=None, use_umfpack=True)


    #mesh_file = "mesh.xdmf"
    #with XDMFFile(mesh.mpi_comm(), mesh_file) as out:
    #    out.write(mesh)
        
    #mesh = Mesh()
    #with XDMFFile(MPI.comm_world, mesh_file) as xdmf:
    #    xdmf.read(mesh)