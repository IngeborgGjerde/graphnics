from lib2to3.pytree import convert
from fenics import *
import scipy.sparse as sp
from petsc4py import PETSc


def mixed_dim_fenics_solve_custom(a, L, W, mesh, real_rows):
        
    # Assemble the system
    qp0 = Function(W)
    system = assemble_mixed_system(a == L, qp0)
    
    A_list = system[0]
    rhs_blocks = system[1]

    # Convert our real rows to PetSc
    rows = [convert_vec_to_petscmatrix(row) for row in real_rows]
    # and get the transpos
    rows_T = [PETScMatrix(row.mat().transpose(PETSc.Mat())) for row in rows]
    
    zero = zero_PETScMat(1,1)
    zero_row = zero_PETScMat(1,W.sub_space(2).dim())
    zero_col = zero_PETScMat(W.sub_space(2).dim(),1)
    
    # Add lagrange multipliers
    A_list_n = A_list[0:3] + [rows_T[0]]   # first row
    A_list_n += A_list[3:6] + [rows_T[1]]  # second row
    A_list_n += A_list[6:9] + [zero_col]     # third row  
    A_list_n += rows  + [zero_row, zero]    # fourth row
    
    # Solve the system
    A_ = PETScNestMatrix(A_list_n) # recombine blocks now with Lagrange multipliers
    b_ = Vector()
    rhs_blocks_n = rhs_blocks + [zero_PETScVec(1)]
    A_.init_vectors(b_, rhs_blocks_n)
    A_.convert_to_aij() # Convert MATNEST to AIJ for LU solver

    sol_ = Vector(mesh.mpi_comm(), sum([P2.dim() for P2 in W.sub_spaces()]) + 1)
    solver = PETScLUSolver()
    solver.solve(A_, sol_, b_)

    # Transform sol_ into qp0 and update qp_
    dim_shift = 0
    for s in range(W.num_sub_spaces()):
        qp0.sub(s).vector().set_local(sol_.get_local()[dim_shift:dim_shift + qp0.sub(s).function_space().dim()])
        dim_shift += qp0.sub(s).function_space().dim()
        qp0.sub(s).vector().apply("insert")
    return qp0



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


def convert_vec_to_petscmatrix(vec):
    '''
    Convert a fenics vector from assemble into 
    dolfin.cpp.la.PETScMatrix
    '''
    
    # Make sparse
    sparse_vec = sp.coo_matrix(vec.get_local())
    #row = sp.coo_matrix(vec)
    
    # Init PETSC matrix
    petsc_mat = PETSc.Mat().createAIJ(size=sparse_vec.shape)
    petsc_mat.setUp()
    
    # Input values from sparse_vec
    for i,j,v in zip(sparse_vec.row, sparse_vec.col, sparse_vec.data): 
        petsc_mat.setValue(i,j,v)
    
    petsc_mat.assemble()
    
    return PETScMatrix(petsc_mat)

def zero_PETScVec(n):
    '''
    Make PETScVec with n zeros 
    '''
    
    petsc_vec = PETSc.Vec().createSeq(n)
    return PETScVector(petsc_vec)


def zero_PETScMat(n,m):
    '''
    Make n,m PETScMatrix with zeros
    '''
        
    petsc_mat = PETSc.Mat().createAIJ((n,m))
    petsc_mat.setUp()
    petsc_mat.assemble()
    
    return PETScMatrix(petsc_mat)


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
    
    