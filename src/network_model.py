
from cmath import e
import networkx as nx
from fenics import *
from fenics_graph import *
from utils import *
from graph_examples import *
    

def dds_on_G(f,G):
    return dot(grad(f), G.global_tangent)


def hydraulic_network_model(G, inlets=[], outlets=[]):

    # Make list of edges
    edges= list(nx.get_edge_attributes(G, 'submesh').keys())
    num_edges = len(edges)
    # and list of submeshes
    submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
    
    mesh = G.global_mesh

    # Function spaces
    P2s = [FunctionSpace(msh, 'CG', 2) for msh in submeshes] # Flux spaces in each segment
    P1 = FunctionSpace(mesh, 'CG', 1) # Pressure space (on whole mesh)
    LMs = [FunctionSpace(mesh, 'R', 0) for b in G.bifurcation_ixs] # Lagrange multiplier (to impose bifurcation conditions)

    spaces = P2s + LMs + [P1]
    W = MixedFunctionSpace(*spaces) 


    # Trial and test functions
    vphi = TestFunctions(W)
    qp = TrialFunctions(W)


    dx = Measure('dx', domain=mesh)
    dx_ = [Measure("dx", domain = msh) for msh in submeshes]
    

    ## Assemble variational formulation 

    # Initialize a and L to zero
    a = Constant(0)*qp[-1]*vphi[-1]*dx
    for ix, b, in enumerate(G.bifurcation_ixs):
        a += Constant(0)*qp[-2-ix]*vphi[-2-ix]*dx
    L = Constant(0)*vphi[-1]*dx + Constant(0)*vphi[-2]*dx

    # Add in branch contributions
    def dds(f):
        return dds_on_G(f,G)

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
        node_out, node_in = e # the edge goes from one node to the other
        
        # If node_out is bifurcation point we add contributions from lagrange multipliers
        if node_out in G.bifurcation_ixs: 
            node_lm_ix = num_edges+node_out-1 # 
            a += qp[branch_ix]*vphi[node_lm_ix]*ds_branch(BIF_OUT)
            a += vphi[branch_ix]*qp[node_lm_ix]*ds_branch(BIF_OUT) 
        if node_in in G.bifurcation_ixs:
            node_lm_ix = num_edges+node_in-1 # 
            a -= qp[branch_ix]*vphi[node_lm_ix]*ds_branch(BIF_IN)
            a -= vphi[branch_ix]*qp[node_lm_ix]*ds_branch(BIF_IN) 
        
        # Add boundary condition for inflow/outflow boundary node
        for inlet_tag in inlets:
            L += vphi[branch_ix]*ds_branch(inlet_tag)
        for outlet_tag in outlets:
            L -= vphi[branch_ix]*ds_branch(outlet_tag)

    # Solve
    qp0 = mixed_dim_fenics_solve(a, L, W, mesh)
    return qp0



def test_mass_conservation():

    # Test mass conservation on double y bifurcation mesh
    G = make_double_Y_bifurcation()
    G.make_mesh(0)
    qp0 = hydraulic_network_model(G, inlets=[BOUN_IN])

    vars = qp0.split()
    
    edges= list(nx.get_edge_attributes(G, 'submesh').keys())
    

    for b in G.bifurcation_ixs:
        flux_in, flux_out = 0, 0
        
        for e_in in G.in_edges(b):
            msh = G.edges[e_in]['submesh']
            vf = G.edges[e_in]['vf']
            b_ix = np.where(vf.array()==BIF_IN)[0]
            if b_ix:
                b_coord = msh.coordinates()[b_ix[0],:] 
    
                branch_ix = edges.index(e_in)
                flux_in += vars[branch_ix](b_coord)
            
        for e_out in G.out_edges(b):
            msh = G.edges[e_out]['submesh']
            vf = G.edges[e_out]['vf']
            b_ix = np.where(vf.array()==BIF_OUT)[0]
            if b_ix:
                b_coord = msh.coordinates()[b_ix[0],:] 
        
                branch_ix = edges.index(e_out)
                flux_out += vars[branch_ix](b_coord)
           
        assert near(flux_in, flux_out, 1e-3), f'Mass is not conserved at bifurcation {b}'


if __name__ == '__main__':

    test_mass_conservation()

    # Make simple graph
    from graph_examples import *
    G = honeycomb(1,1)
    #G = make_double_Y_bifurcation()
    G.make_mesh(2)
    qp0 = hydraulic_network_model(G, inlets=[BOUN_OUT])

    vars = qp0.split()
    p = vars[-1]
    lam = vars[-2]

    q = GlobalFlux(G, vars[0:-2])
    qi = interpolate(q, VectorFunctionSpace(G.global_mesh, 'DG', 2, G.geom_dim))
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