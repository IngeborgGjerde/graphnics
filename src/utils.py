from fenics import *

def mixed_dim_fenics_solve(a, L, W, mesh):
        
    # Assemble the system
    qp0 = Function(W)
    system = assemble_mixed_system(a == L, qp0)
    A_list = system[0]
    rhs_blocks = system[1]

    # Solve the system
    A_ = PETScNestMatrix(A_list) # recombine blocks
    b_ = Vector()
    A_.init_vectors(b_, rhs_blocks)
    A_.convert_to_aij() # Convert MATNEST to AIJ for LU solver

    sol_ = Vector(mesh.mpi_comm(), sum([P2.dim() for P2 in W.sub_spaces()]))
    solver = PETScLUSolver()
    solver.solve(A_, sol_, b_)

    # Transform sol_ into qp0 and update qp_
    dim_shift = 0
    for s in range(W.num_sub_spaces()):
        qp0.sub(s).vector().set_local(sol_.get_local()[dim_shift:dim_shift + qp0.sub(s).function_space().dim()])
        dim_shift += qp0.sub(s).function_space().dim()
        qp0.sub(s).vector().apply("insert")
    return qp0



def assemble_global_flux(qs, G):
    [q.set_allow_extrapolation(True) for q in qs]

    #global_DG2 = FunctionSpace(G.global_mesh, 'DG', 2)
    #q_e = [interpolate(q, global_DG2) for q in qs]
    #q_global = 0
    #for i in range(0,len(q_e)):
    #    q_global += q_e[i]*CharacteristicFunction(G,i)

    #q_global = project(q_global)
    #return q_global
    pass




class CharacteristicFunction(UserExpression):
    '''
    Characteristic function on edge i
    '''
    def __init__(self, G, edge, **kwargs):
        '''
        Args:
            G (nx.graph): Network graph
            edge (int): edge index
        '''
        super().__init__(**kwargs)
        self.G=G
        self.edge=edge

    def eval_cell(self, values, x, cell):
        edge = self.G.mf[cell.index]
        values[0] = (edge==self.edge)
    
    