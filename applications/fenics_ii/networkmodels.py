
import networkx as nx
from fenics import *
from xii import *
import sys
sys.path.append('../../') 
from graphnics import *

time_stepping_schemes = {'IE':{'b1':Constant(0), 'b2':Constant(1)},
                         'CN':{'b1':Constant(0.5), 'b2':Constant(0.5)}}
    

class HydraulicNetwork_ii:
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
    
    def __init__(self, G, f, ns):
        self.G = G
        self.f = f
        self.ns = ns
        
    def a_edge(self, qp, vphi):
        '''
        Args:
            qp (list of df.trialfunction)
            vphi (list of df.testfunction)
        '''
        
        # split out the components
        n_edges = self.G.num_edges

        qs, ps, lams = qp[0:n_edges], qp[n_edges:2*n_edges], qp[2*n_edges:]
        vs, phis, xis = vphi[0:n_edges], vphi[n_edges:2*n_edges], vphi[2*n_edges:]


        ## Init a as list of lists        
        a = [[ 0 for i in range(0, len(qp))  ] for j in range(0, len(qp))]
        
        # now fill in the non-zero blocks
        for i, e in enumerate(self.G.edges):
            
            dx_edge = Measure("dx", domain = self.G.edges[e]['submesh'])
            
            # Diagonal block with q and its test function
            a[i][i] += qs[i]*vs[i]*dx_edge 
        
            a[n_edges+i][i] += + self.G.dds(qs[i])*phis[i]*dx_edge
            a[i][n_edges+i] += - ps[i]*self.G.dds(vs[i])*dx_edge
        
        
        edge_list = list(self.G.edges.keys())
        for ix_bf, node in enumerate(self.G.bifurcation_ixs):
            
            for e in self.G.in_edges(node):
                ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
                edge_ix = edge_list.index(e)
                a[edge_ix][n_edges*2+ix_bf] += vs[edge_ix]*lams[ix_bf]*ds_edge(BIF_IN)
                a[n_edges*2+ix_bf][edge_ix] += qs[edge_ix]*xis[ix_bf]*ds_edge(BIF_IN)

            for e in self.G.out_edges(node):
                ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
                edge_ix = edge_list.index(e)
                a[edge_ix][n_edges*2+ix_bf] -= vs[edge_ix]*lams[ix_bf]*ds_edge(BIF_OUT)
                a[n_edges*2+ix_bf][edge_ix] -= qs[edge_ix]*xis[ix_bf]*ds_edge(BIF_OUT)

        return a 
   
    def L(self, vphi):
        '''
        Args:
            vphi (list of df.testfunction)
        '''
        
        # split out the components
        n_edges = G.num_edges
        vs, phis, xis = vphi[0:n_edges], vphi[n_edges:2*n_edges], vphi[2*n_edges:]
        
        L = [ 0 for i in range(0, len(vphi))  ]

        # Assemble edge contributions to a and L
        for i, e in enumerate(self.G.edges):
            ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
            dx_edge = Measure("dx", domain = self.G.edges[e]['submesh'])
            
            L[i] += self.ns*vs[i]*ds_edge
            L[i+n_edges] += self.f*phis[i]*dx_edge

        for i in range(0, len(G.bifurcation_ixs)):        
            L[n_edges*2+i] += Constant(0)*xis[i]*dx
        return L
    

@timeit
def hydraulic_network_simulation(G, f=Constant(0), p_bc=Constant(0)):
    '''
    Set up spaces and solve hydraulic network model 
        R q + d/ds p = 0
            d/ds q = f
    on graph G, with bifurcation condition q_in = q_out 

    Args:
        G (fg.FenicsGraph): problem domain
        f (df.function): source term
        p_bc (df.function): neumann bc for pressure
    '''
    
    # Flux spaces on each segment, ordered by the edge list
    submeshes = list(nx.get_edge_attributes(G, 'submesh').values())

    P2s = [FunctionSpace(msh, 'CG', 2) for msh in submeshes] 
    P1s = [FunctionSpace(msh, 'CG', 1) for msh in submeshes] 
    LMs = [FunctionSpace(G.global_mesh, 'R', 0) for b in G.bifurcation_ixs]
    
    ### Function spaces
    W = P2s + P1s + LMs
    
    # Trial and test functions
    qp = list(map(TrialFunction, W))
    vphi = list(map(TestFunction, W))
    
    ## Assemble variational formulation 
    model = HydraulicNetwork_ii(G, f, p_bc)
    
    a = model.a_edge(qp, vphi)
    L = model.L(vphi)
    
    A = ii_assemble(a)
    b = ii_assemble(L)

    from IPython import embed
    #embed()

    A = ii_convert(A)
    b = ii_convert(b)

    wh = ii_Function(W)
    solver = LUSolver(A, 'mumps')
    solver.solve(wh.vector(), b)

    return wh

 


### ------------- Tests ------------ ###

def test_mass_conservation():
    '''
    Test mass conservation of hydraulic network model
    on double y bifurcation mesh and 1x1 honeycomb mesh
    '''
    
    Gs = []
    Gs.append(make_double_Y_bifurcation())
    Gs.append(honeycomb(1,1))


    for G in Gs:
        G.make_mesh(0)
        for e in G.in_edges(): G.in_edges()[e]['res'] = 1
        for e in G.out_edges(): G.out_edges()[e]['res'] = 1
        
        
        qp0 = hydraulic_network_simulation(G)

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
    
    import os
    #os.system('dijitso clean') 
    G = make_line_graph(22)
    G.make_mesh(2)
    
    qp0 = hydraulic_network_simulation(G, p_bc = Expression('x[0]', degree=2))