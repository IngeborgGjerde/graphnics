
import networkx as nx
import numpy as np
from fenics import *


if __name__ == '__main__':

    G = FenicsGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.nodes[0]['pos']=[0,0,0]
    G.nodes[1]['pos']=[0,0.5,0]
    G.nodes[2]['pos']=[-0.5,1,0]
    G.nodes[3]['pos']=[0.5,1,0]

    G.add_edge(0,1)
    G.add_edge(1,2)
    G.add_edge(1,3)
    G.make_mesh()
    mesh = G.global_mesh

    vix_submeshes_list = list( nx.get_edge_attributes(G, 'submesh').items() )



    ## Assemble problem ##
    P2s = [FunctionSpace(vx_submesh[1], 'CG', 2) for vx_submesh in vix_submeshes_list] # Flux spaces in each segment
    P1 = FunctionSpace(mesh, 'CG', 1) # Pressure space (on whole mesh)
    LM = FunctionSpace(mesh, 'R', 0) # Lagrange multiplier (to impose bifurcation conditions)

    spaces = P2s; spaces.append(LM); spaces.append(P1)
    total_dim = np.sum( [space.dim() for space in spaces] )

    W = MixedFunctionSpace(*spaces) 

    dx = Measure('dx', domain=mesh)
    dx_ = [Measure("dx", domain = vx_submesh[1]) for vx_submesh in vix_submeshes_list]
    
    
    vphi = TestFunctions(W)
    qp = TrialFunctions(W)


    def dds(f):
        return dot(grad(f), G.global_tangent) 

    a = qp[-2]*vphi[-2]*dx
    
    for i in range(0, len(G.edges)):
        a += qp[i]*vphi[i]*dx_[0]
        a -= dds(qp[-1])*vphi[i]*dx_[0]
        a += vphi[-1]*qp[i]*dx_[0]

    L = Constant(0)*vphi[-1]*dx + Constant(0)*vphi[-2]*dx

    qp0 = Function(W)
    system = assemble_mixed_system(a == L, qp0)
    A_list = system[0]
    rhs_blocks = system[1]



    # Port to scipy 
    from scipy.sparse import coo_matrix, bmat
    A_arrays_flattened = [coo_matrix(block.array()) for block in A_list]
    A = bmat([A_arrays_flattened[len(spaces)*i:len(spaces)*(i+1)] for i in range(0, len(spaces))])

    b = np.concatenate( [block.get_local() for block in rhs_blocks] )
    

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