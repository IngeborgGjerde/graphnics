from telnetlib import DO
import networkx as nx
from fenics import *

import sys
sys.path.append('../')
from graphnics import *
from xii import *

def coupled_1D_3D(mesh3, G, radius = 0.1, f3=Constant(0), f1=Constant(0), beta=Constant(1),
                  p_bc_3=Expression('x[0]', degree=2), p_bc_1=Expression('x[0]', degree=2)):
    '''
    Solve coupled 1D-3D model
        - \Delta p3 = (u3-u1) \delta_\Lambda + f3
        - \partial_{ss} p1 = -(u3-u1) + f1
    on graph G

    Args:
        mesh3 (df.mesh): 3D mesh
        G (fg.FenicsGraph): problem domain
        radius (float): inclusion radius
        f3 (df.function): 3D source term
        f1 (df.function): 1D source term
        p_bc_3 (df.function): neumann bc for 3D pressure
        p_bc_1 (df.function): neumann bc for 1D pressure
    '''
    
    beta = Constant(1)
    
    mesh1 = G.global_mesh

    # Pressure space on global mesh
    V3 = FunctionSpace(mesh3, 'CG', 1)
    V1 = FunctionSpace(mesh1, 'CG', 1)
    W = [V3, V1]
    
    u3, u1 = list(map(TrialFunction, W))
    v3, v1 = list(map(TestFunction, W))

    # Averaging surface
    cylinder = Circle(radius=radius, degree=10)

    Pi_u = Average(u3, mesh1, cylinder)
    T_v = Average(v3, mesh1, None)  # This is 3d-1d trace

    dxGamma = Measure('dx', domain=mesh1)

    a00 = inner(grad(u3), grad(v3))*dx + beta*inner(Pi_u, T_v)*dxGamma
    a01 = -beta*inner(u1, T_v)*dxGamma
    a10 = -beta*inner(Pi_u, v1)*dxGamma
    a11 = inner(grad(u1), grad(v1))*dx + beta*inner(u1, v1)*dxGamma
    
    
    L0 = inner(f3, T_v)*dxGamma
    L1 = inner(f1, v1)*dxGamma

    a = [[a00, a01], [a10, a11]]
    L = [L0, L1]
    
    W_bcs = [[DirichletBC(V3, p_bc_3, 'on_boundary')], [DirichletBC(V1, p_bc_1, 'on_boundary')]]
    
    A, b = map(ii_assemble, (a, L))
    A, b = apply_bc(A, b, W_bcs)
    A, b = map(ii_convert, (A, b))

    wh = ii_Function(W)
    solver = LUSolver(A, 'mumps')
    solver.solve(wh.vector(), b)

    return wh



def line_in_box(refs=3, a=[0.5,0.5,0.25], b=[0.5,0.5,0.75]):    
    '''
    Make meshes for a 1D line in a 3D unit cube box
    
    Args:
        refs (int): number of refinements
        a (list): coords of line start
        b (list) coords of line end
        
    Returns:
        mesh3 
        mesh1
    '''
    
    G = FenicsGraph()
    G.add_nodes_from(range(0,2))
    G.nodes[0]['pos']=a
    G.nodes[1]['pos']=b
    G.add_edge(0, 1)

    G.make_mesh(refs)
    
    # Make 3D mesh around it
    N = 2**refs
    mesh3 = UnitCubeMesh(N, N, N)
    
    return mesh3, G


if __name__ == '__main__':
    
    
    
    import mms
    u3_a, u1_a, beta = mms.coupled_1D_3D([0,0,0.25], [0,0,0.75], 0.1)
    
    for N in [3, 4, 5, 6]:
        mesh3, G = line_in_box(refs=N)    
        wh = coupled_1D_3D(mesh3, G, p_bc_3 = u3_a, p_bc_1=u1_a, beta=beta)
        uh3, uh1 = wh
        
        u3_ai = interpolate(u3_a, FunctionSpace(mesh3, 'CG', 2))
        u1_ai = interpolate(u1_a, FunctionSpace(G.global_mesh, 'CG', 2))
        print('%1.1e'%mesh3.hmin(), '%1.1e'%G.global_mesh.hmin(), '%1.1e'%errornorm(uh3, u3_ai), '%1.1e'%errornorm(uh1, u1_ai))
    
    uh3.rename('uh3', '0.0')
    uh1.rename('uh1', '0.0')
    
    File('../plots/uh3.pvd')<<uh3
    File('../plots/uh1.pvd')<<uh1
    
    
