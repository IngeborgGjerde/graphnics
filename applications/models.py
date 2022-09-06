
import networkx as nx
from fenics import *
from xii import *
import sys
sys.path.append('../') 
from graphnics import *

time_stepping_schemes = {'IE':{'b1':Constant(0), 'b2':Constant(1)},
                         'CN':{'b1':Constant(0.5), 'b2':Constant(0.5)}}
    

class HydraulicNetwork:
    '''
    Bilinear forms a and L for the hydraulic equations
            R*q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous 
    normal stress
    
    Args:
        G (FenicsGraph): Network domain
        f (df.function): fluid source term
        ns (df.function): normal stress for neumann bcs
    '''
    
    def __init__(self, G):
        '''
        Set up function spaces and store model paramters f and ns
        '''
        
        # Graph on which the model lives
        self.G = G

        # Setup function spaces: 
        # - flux space on each segment, ordered by the edge list
        # - pressure space on the full mesh 
        # - real space on each bifurcation
        submeshes = list(nx.get_edge_attributes(G, 'submesh').values())

        P2s = [FunctionSpace(msh, 'CG', 2) for msh in submeshes] 
        P1s = [FunctionSpace(G.global_mesh, 'CG', 1)] 
        LMs = [FunctionSpace(G.global_mesh, 'R', 0) for b in G.bifurcation_ixs]
    
        ### Function spaces
        W = P2s + P1s + LMs
        self.W = W

        # Trial and test functions
        self.qp = list(map(TrialFunction, W))
        self.vphi = list(map(TestFunction, W))
            

    def a_form(self):
        '''
        The bilinear form
        '''

        qp, vphi = self.qp, self.vphi
        G = self.G
        
        # split out the components
        n_edges = G.num_edges

        qs, p, lams = qp[0:n_edges], qp[n_edges], qp[n_edges+1:]
        vs, phi, xis = vphi[0:n_edges], vphi[n_edges], vphi[n_edges+1:]

        submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
        ps = [Restriction(p, msh) for msh in submeshes]
        phis = [Restriction(phi, msh) for msh in submeshes]

        ## Init a as list of lists        
        a = [[ 0 for i in range(0, len(qp))  ] for j in range(0, len(qp))]
        
        # edge contributions to form
        for i, e in enumerate(G.edges):
            
            dx_edge = Measure("dx", domain = G.edges[e]['submesh'])
            
            a[i][i] += qs[i]*vs[i]*dx_edge 
            a[n_edges][i] += + G.dds_i(qs[i], i)*phis[i]*dx_edge
            a[i][n_edges] += - ps[i]*G.dds_i(vs[i], i)*dx_edge

        
        edge_list = list(G.edges.keys())
        

        # bifurcation condition contributions to form
        for b_ix, b in enumerate(G.bifurcation_ixs):

            # make info dict of edges connected to this bifurcation
            conn_edges  = { **{e :  (1,  BIF_IN,  edge_list.index(e)) for e in G.in_edges(b) }, 
                            **{e :  (-1, BIF_OUT, edge_list.index(e)) for e in G.out_edges(b) }}
            
            for e in conn_edges:
                ds_edge = Measure('ds', domain=G.edges[e]['submesh'], subdomain_data=G.edges[e]['vf'])
                
                sign, tag, e_ix = conn_edges[e]

                a[e_ix][n_edges+1+b_ix] += sign*vs[e_ix]*lams[b_ix]*ds_edge(tag)
                a[n_edges+1+b_ix][e_ix] += sign*qs[e_ix]*xis[b_ix]*ds_edge(tag)

           
        return a 
   
    def L_form(self, f, p_bc):
        '''
        The right-hand side linear form

        Args:
            f (df.expr): source term
            p_bc (df.expr): pressure boundary condition
        '''
        
        vphi = self.vphi

        # split out the components
        n_edges = self.G.num_edges
        vs, phi, xis = vphi[0:n_edges], vphi[n_edges], vphi[n_edges+1:]

        submeshes = list(nx.get_edge_attributes(self.G, 'submesh').values())
        phis = [Restriction(phi, msh) for msh in submeshes]

        
        L = [ 0 for i in range(0, len(vphi))  ]

        # Assemble edge contributions to a and L
        for i, e in enumerate(self.G.edges):
            ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
            dx_edge = Measure("dx", domain = self.G.edges[e]['submesh'])
            
            L[i] += p_bc*vs[i]*ds_edge(BOUN_OUT) - p_bc*vs[i]*ds_edge(BOUN_IN)
            L[n_edges] += f*phis[i]*dx_edge

        for i in range(0, len(self.G.bifurcation_ixs)):       
            L[n_edges+1+i] += Constant(0)*xis[i]*dx
        return L
    




### ------------- Tests ------------ ###

def test_mass_conservation():
    '''
    Test mass conservation of hydraulic network model at bifurcation points
    '''
    
    tests = {'Graph line' : make_line_graph(3, dim=3),
             'Y bifurcation': make_Y_bifurcation(),
             'YYY bifurcation': make_double_Y_bifurcation(),
             'honeycomb': honeycomb(4,4)
            }

    for test_name in tests:
        G = tests[test_name]
        
        G.make_mesh(5)

        model = HydraulicNetwork(G)
        
        W = model.W
        a = model.a_form()
        L = model.L_form(f=Constant(0), p_bc=Expression(('x[0]'), degree=2))
        
        A, b = map(ii_assemble, (a,L))
        A, b = map(ii_convert, (A,b))

        qp = ii_Function(W)
        solver = LUSolver(A, 'mumps')
        solver.solve(qp.vector(), b)

        plot_solution = False
        if plot_solution:
            qq = GlobalFlux(G, qp[:G.num_edges])
            pp = qp[G.num_edges]
            pp.rename('pp', '0')
            File('pp.pvd') << pp      
            qq_i = interpolate(qq, VectorFunctionSpace(G.global_mesh, 'DG', 1, G.global_mesh.geometric_dimension()))
            File('qq.pvd') << qq_i        
            qq_i.rename('qq', '0')

        edge_list = list(G.edges.keys())

        for b in G.bifurcation_ixs:
            total_flux = 0

            # make info dict of edges connected to this bifurcation
            conn_edges  = { **{e :  (1,  BIF_IN,  edge_list.index(e)) for e in G.in_edges(b) }, 
                            **{e :  (-1, BIF_OUT, edge_list.index(e)) for e in G.out_edges(b) }}
            
            for e in conn_edges:
                sign, tag, e_ix = conn_edges[e]
                total_flux += sign*qp[e_ix](G.nodes[b]['pos'])

            assert near(total_flux, 1e-3, 2e-3), f'Mass is not conserved at bifurcation {b} for {test_name}'


if __name__ == '__main__':
    
    #import os
    #os.system('dijitso clean') 
    
    test_mass_conservation()