'''
Copyright (C) 2022-2023 by Ingeborg Gjerde

This file is a part of the graphnics project (https://arxiv.org/abs/2212.02916)

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.
'''


from fenics import *
from xii import *
from tqdm import tqdm
from .flow_models import *
from graphnics import *
set_log_level(40)

time_stepping_schemes = {"IE": {"b1": 0, "b2": 1}, "CN": {"b1": 0.5, "b2": 0.5}}


class TimeDepHydraulicNetwork(HydraulicNetwork):
    
    def __init__(self, G, f=Constant(0), g = Constant(0), p_bc=Constant(0), Res=Constant(1), Ainv=Constant(1)):
        
        self.Ainv = Ainv
        super().__init__(G, f=f, g=g, p_bc=p_bc, Res=Res)
        
    def mass_matrix_q(self, coeff):
        '''
        Assemble mass matrix 
            coeff*(q, v)_\Lambda
            
        Args:
            qp_n (ii_function): solution at previous time step
            coeff (df.function): coefficient in front of the mass matrix
        '''

        dx = Measure("dx", domain=self.G.mesh)
        
        # Init all blocks to zero
        M = [[0 for i in range(0, len(self.qp))] for j in range(0, len(self.qp))]
        
        M[0][0] += coeff*self.qp[0]*self.vphi[0]*dx # Assemble mass matrix        
                
        # Init rest as zero blocks so fenics_ii has something to work with
        M[1][1] = Constant(0)*self.qp[1]*self.vphi[1]*dx
        
        return M

    def mass_vector_q(self, u, coeff):
        '''
        Assemble mass vector 
            coeff*(q, v)_\Lambda
        
        '''
        
        dx = Measure("dx", domain=self.G.mesh)
        
        Mv = [0 for i in range(0, len(self.qp))] # Init all rows to zero
        Mv[0] = coeff*u[0]*self.vphi[0]*dx # Assemble mass vector for q
        
        # Init rest as zero blocks so fenics_ii has something to work with
        Mv[1] = Constant(0)*self.vphi[1]*dx 
        
        return Mv



class TimeDepMixedHydraulicNetwork(MixedHydraulicNetwork):

    def __init__(self, G, f=Constant(0), g = Constant(0), p_bc=Constant(0), Res=Constant(1), Ainv=Constant(1)):
        
        self.Ainv = Ainv
        super().__init__(G, f=f, g=g, p_bc=p_bc)
        
    def mass_matrix_q(self, coeff):
        '''
        Assemble mass matrix for fluxes
            \sum_i coeff*(q_i, v_i)_{\Lambda_i}
            
        Args:
            qp_n (ii_function): solution at previous time step
            coeff (df.function): coefficient in front of the mass matrix
        '''

        dx = Measure("dx", domain=self.G.mesh)
        
        # Init all blocks to zero
        M = [[0 for i in range(0, len(self.qp))] for j in range(0, len(self.qp))]
        
        # Assemble mass matrix for fluxes
        for i, e in enumerate(self.G.edges):
            dx_edge = Measure("dx", domain=self.G.edges()[e]['submesh'])
            M[i][i] += coeff*self.qp[i]*self.vphi[i]*dx_edge
                
        # Init rest as zero blocks so fenics_ii has something to work with
        dx = Measure("dx", domain=self.G.mesh)
        
        for i in range(self.G.num_edges, len(self.qp)):
            M[i][i] = Constant(0)*self.qp[i]*self.vphi[i]*dx
        
        return M

    def mass_vector_q(self, u, coeff):
        '''
        Assemble mass vector 
            coeff*(q, v)_\Lambda
        
        '''
        
        dx = Measure("dx", domain=self.G.mesh)
        
        Mv = [0 for i in range(0, len(self.qp))] # Init all rows to zero
        
        # Assemble mass vector for fluxes
        for i, e in enumerate(self.G.edges):
            dx_edge = Measure("dx", domain=self.G.edges()[e]['submesh'])
            Mv[i] += coeff*u[i]*self.vphi[i]*dx_edge
                
        # Init rest as zero blocks so fenics_ii has something to work with
        for i in range(self.G.num_edges, len(self.qp)):
            Mv[i] = Constant(0)*self.vphi[i]*dx
        
        return Mv


def time_stepping_stokes(
    model, t=Constant(0), t_steps=10, T=1, qp0=None, t_step_scheme="IE", reassemble_lhs=True
):
    """
    Do time stepping for models of the type
        1/A d/dt q + a(U, V) = L(V;f,g,ns)
    on graph G, where U=(q, ..., p) and V=(v, ..., phi)

    Args:
        model (class): class containing G, W, a and L
        t (Constant): time
        t_steps (int): number of time steps
        T (float): end time
        qp_n (ii_function): initial solution
        t_step_scheme (str): 'IE' for Implicit Euler or 'CN' for Crank-Nicholson
        reassemble (bool): reassemble matrices at each time step
        
    The model parameters can be expressions depending on class attribute t
    or ufl functions depending on the Constant t.
    """
    
    if qp0 is None:
        qp0 = ii_Function(model.W)  # initialize as zero

    dt = T / t_steps # time step size
    cn, cn1 = time_stepping_schemes[t_step_scheme].values() # Runge Kutta coefficients
    
    
    Dn1 = model.mass_matrix_q(model.Ainv)
    Dn = model.mass_vector_q(qp0, model.Ainv)
    
    a, L = model.a_form(), model.L_form()

    qpn1 = ii_Function(model.W)

    # Finally we time step
    qp_0 = ii_Function(model.W)
    for i, qp0_comp in enumerate(qp_0):
        qp0_comp.vector()[:] = project(qp0[i], model.W[i]).vector().get_local()
    qp0 = qp_0
    
    qps = [qp_0]

    
    An, Ln, DDn = [ii_assemble(term) for term in [a, L, Dn]]

    t.assign(dt)
    
    An1, Ln1, DDn1 = [ii_assemble(term) for term in [a, L, Dn1]]
    
    for t_val in tqdm(np.linspace(dt, T, t_steps - 1)):
    #for t_val in np.linspace(dt, T, t_steps - 1):   
        
        Ln1 = ii_assemble(L)
        if reassemble_lhs:
            An1, DDn1 = [ii_assemble(term) for term in [a, Dn1]]
            
        # Assemble and solve the system at the current time step
        A = (DDn1 + cn1*dt*An1)
        b = (DDn + cn1*dt*Ln1 - dt*cn*An*qp0.block_vec() + dt*cn*Ln)

        A = A.block_collapse()
        A, b = apply_bc(A, b, model.get_bc())
        
        A, b = ii_convert(A), ii_convert(b)
        solver = LUSolver(A, "mumps")
        solver.solve(qpn1.vector(), b)

        # Store solution
        sol = ii_Function(model.W)
        [sol[i].assign(func) for i, func in enumerate(qpn1)]
                
        qps.append(sol)
        # Prepare for next time step        
        # 1) update qp_n
        [qp0[i].assign(func) for i, func in enumerate(qpn1)]

        # 2) update time
        t.assign(t_val+dt) #in case one of the model parameters are ufl type
        
        # in case one of the model parameters are expressions       
        for func in [model.p_bc, model.g, model.f]:
            try: 
                func.t = t_val+dt 
            except AttributeError:
                pass # if this is a ufl function it won't have a t attribute        
        
        # 2) update matrices
        An = An1
        Ln = Ln1
        DDn = DDn1*qp0.block_vec()
        
        a = model.a_form()
        L = model.L_form()

    return qps




if __name__ == "__main__":
    print("Testing time stepping")
    from IPython import embed; embed()
    test_time_stepping()