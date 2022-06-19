import networkx as nx
from fenics import *
from fenics_graph import *
from utils import *
from graph_examples import *
    



def hydraulic_network_model(G, inlets=[], outlets=[]):

    mesh = G.global_mesh

    # Flux spaces on each segment, ordered by the edge list
    submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
    P2s = [FunctionSpace(msh, 'CG', 2) for msh in submeshes] 
    
    # Real space on each bifurcation, ordered by G.bifurcation_ixs
    LMs = [FunctionSpace(mesh, 'R', 0) for b in G.bifurcation_ixs] 

    # Pressure space on global mesh
    P1 = FunctionSpace(mesh, 'CG', 1) # Pressure space (on whole mesh)
    
    ### Function spaces
    spaces = P2s + LMs + [P1]
    W = MixedFunctionSpace(*spaces) 


    # Trial and test functions
    vphi = TestFunctions(W)
    qp = TrialFunctions(W)

    # split out the components
    qs = qp[0:G.num_edges]
    lams = qp[G.num_edges:-1]
    p = qp[-1]

    vs = vphi[0:G.num_edges]
    xis = vphi[G.num_edges:-1]
    phi = vphi[-1]


    ## Assemble variational formulation 

    # Initialize blocks in a and L to zero
    # (so fenics-mixed-dim does not throw an error)
    dx = Measure('dx', domain=mesh)
    a = Constant(0)*p*phi*dx
    for ix in range(0, len(G.bifurcation_ixs)):
        a += Constant(0)*lams[ix]*xis[ix]*dx
    L = Constant(0)*phi*dx


    # We assemble a and L edge by edge
    for i, e in enumerate(G.edges):
        
        msh = G.edges[e]['submesh']
        vf = G.edges[e]['vf']
        
        dx_edge = Measure("dx", domain = msh)
        ds_edge = Measure('ds', domain=msh, subdomain_data=vf)

        # Add variational terms defined on edge
        a += qs[i]*vs[i]*dx_edge        
        a -= p*G.dds(vs[i])*dx_edge
        a += phi*G.dds(qs[i])*dx_edge

        
        # Add bifurcation condition contribution from this edge
        node_out, node_in = e # the edge goes from one node to the other
        
        # If node_out is bifurcation point we add contributions from lagrange multipliers
        if node_out in G.bifurcation_ixs:
            node_lm_ix = G.bifurcation_ixs.index(node_out) 
            a += qs[i]*xis[node_lm_ix]*ds_edge(BIF_OUT)
            a += vs[i]*lams[node_lm_ix]*ds_edge(BIF_OUT) 
        
        # same if node_in is a bifurcation point
        if node_in in G.bifurcation_ixs:
            node_lm_ix = G.bifurcation_ixs.index(node_in)  
            a -= qs[i]*xis[node_lm_ix]*ds_edge(BIF_IN)
            a -= vs[i]*lams[node_lm_ix]*ds_edge(BIF_IN) 
        
        
        
        # Add boundary condition for inflow/outflow boundary node
        for inlet_tag in inlets:
            L += Expression('x[1]', degree=2)*vphi[i]*ds_edge(inlet_tag)
        for outlet_tag in outlets:
            L -= Expression('x[1]', degree=2)*vphi[i]*ds_edge(outlet_tag)

    # Solve
    qp0 = mixed_dim_fenics_solve(a, L, W, mesh)
    return qp0




def test_mass_conservation():

    # Test mass conservation on double y bifurcation mesh
    # and 1x1 honeycomb mesh
    
    Gs = []
    Gs.append(make_double_Y_bifurcation())
    Gs.append(honeycomb(1,1))

    for G in Gs:
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

    # Make simple graph
    from graph_examples import *
    G = honeycomb(3,3)
    #G = make_double_Y_bifurcation()
    G.make_mesh(1)
    qp0 = hydraulic_network_model(G, inlets=[BOUN_OUT])

    vars = qp0.split()
    p = vars[-1]
    
    q = GlobalFlux(G, vars[0:-2])
    qi = interpolate(q, VectorFunctionSpace(G.global_mesh, 'DG', 2, G.geom_dim))
    p = vars[-1]
    
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