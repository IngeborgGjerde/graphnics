
import networkx as nx
from fenics import *
from xii import *
import sys
sys.path.append('../') 
from graphnics import *
import models

from slepc4py import SLEPc

class HydraulicNetwork_EP(models.HydraulicNetwork):
    '''
    Bilinear forms a and L for the hydraulic equations
            Res*q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous 
    normal stress
    
    Args:
        G (FenicsGraph): Network domain
        Res (dict): dictionary with edge->resistance
        f (df.expr): source term
        p_bc (df.expr): pressure boundary condition
    '''
    
    def b_form_eigval(self):
        '''
        The right-hand side linear form
        '''
        
        a = self.init_a_form()
        qp, vphi = self.qp, self.vphi
        G = self.G
        
        # split out the components
        n_edges = G.num_edges

        qs, vs = qp[:n_edges], vphi[:n_edges]
        
        # edge contributions to form
        for i, e in enumerate(G.edges):
            dx_edge = Measure("dx", domain = G.edges[e]['submesh'])
    
            a[i][i] += qs[i]*vs[i]*dx_edge + G.dds_i(qs[i], i)*G.dds_i(vs[i], i)*dx_edge
       
        a[n_edges][n_edges] += qp[n_edges]*vphi[n_edges]*dx 
        
        for i in range(0, G.num_bifurcations):
            a[n_edges+1+i][n_edges+1+i] += qp[n_edges+1+i]*vphi[n_edges+1+i]*dx 
          
        
        return a


class NetworkStokes_EP(models.NetworkStokes):
    '''
    Bilinear forms a and L for the hydraulic equations
            Res*q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous 
    normal stress
    
    Args:
        G (FenicsGraph): Network domain
        Res (dict): dictionary with edge->resistance
        f (df.expr): source term
        p_bc (df.expr): pressure boundary condition
    '''
    
    def b_form_eigval(self):
        '''
        The right-hand side linear form
        '''
        
        a = self.init_a_form()
        qp, vphi = self.qp, self.vphi
        G = self.G
        
        # split out the components
        n_edges = G.num_edges

        qs, vs = qp[:n_edges], vphi[:n_edges]
        
        # edge contributions to form
        for i, e in enumerate(G.edges):
            dx_edge = Measure("dx", domain = G.edges[e]['submesh'])
    
            a[i][i] += qs[i]*vs[i]*dx_edge + G.dds_i(qs[i], i)*G.dds_i(vs[i], i)*dx_edge
       
        a[n_edges][n_edges] += qp[n_edges]*vphi[n_edges]*dx 
        
        for i in range(0, G.num_bifurcations):
            a[n_edges+1+i][n_edges+1+i] += qp[n_edges+1+i]*vphi[n_edges+1+i]*dx 
          
        
        return a




def solve_infsup_eigproblem(model):
    '''
    Solve the eigenproblem 
        Ax = lamda B x
    associated with the infsup condition for the bilinear form of a mixed problem
    
    Args:
        model: class containing the bilinear forms for A and B 
    '''

    a = model.a_form()
    a_r = model.b_form_eigval()
    L = model.L_form()

    A, b = map(ii_assemble, (a,L))

    #num_inlets, num_outlets = model.G.get_num_inlets_outlets()
    #inflow, outflow = 1/num_inlets, 1/num_outlets 

    V_bcs = []
    for i, e in enumerate(model.G.edges()):
        vf = model.G.edges[e]['vf']
        bcs_edge = []
        #if inflow_bcs: bcs_edge.append( DirichletBC(model.W[i], Constant(inflow), vf, BOUN_IN) )
        #if outflow_bcs: bcs_edge.append( DirichletBC(model.W[i], Constant(outflow), vf, BOUN_OUT) )

        V_bcs.append(bcs_edge)

    V_bcs = V_bcs + [[]]*(model.G.num_bifurcations+1)
    A, b = apply_bc(A, b, V_bcs)

    AA = ii_convert(A)
    AAA = PETSc.Mat().createDense(AA.array().shape)
    AAA.setUp()
    AAA[:, :] = AA.array()
    AAA.assemble()

    BB = ii_convert(ii_assemble(a_r))
    BBB = PETSc.Mat().createDense(BB.array().shape)
    BBB.setUp()
    BBB[:, :] = BB.array()
    BBB.assemble()

    E = SLEPc.EPS()
    E.create()

    E.setOperators(AAA, BBB)
    E.setWhichEigenpairs(E.Which.SMALLEST_MAGNITUDE)

    E.setDimensions(50)

    E.solve()
    nconv = E.getConverged()
    
    u_lams, eigvals = [], []
    for i in range(0, min(nconv, 20)):

        eigvals.append(np.abs(np.real(E.getEigenvalue(i))))

        u_r, u_im = ii_Function(model.W), ii_Function(model.W)
        E.getEigenpair(i, u_r.vector().vec(), u_im.vector().vec())
        u_lams.append(u_r)
        
    return eigvals, u_lams