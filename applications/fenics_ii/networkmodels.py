
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
        
    def a(self, qp, vphi):
        '''
        Args:
            qp (list of df.trialfunction)
            vphi (list of df.testfunction)
        '''
        
        # split out the components
        qs, lams, p = qp[0:self.G.num_edges], qp[self.G.num_edges:-1], qp[-1]
        vs, xis, phi = vphi[0:self.G.num_edges], vphi[self.G.num_edges:-1], vphi[-1]
        
        ## Init a as list of lists        
        a = [[0]*len(vphi)]*len(qp)
        
        # now fill in the non-zero blocks
        for i, e in enumerate(self.G.edges):
            
            resist = self.G.edges[e]['res']
            dx_edge = Measure("dx", domain = self.G.edges[e]['submesh'])
            
            Tp = Restriction(p, self.G.edges[e]['submesh'])
            Tphi = Restriction(phi, self.G.edges[e]['submesh'])
            
            a[i][i] += resist*qs[i]*vs[i]*dx_edge 
            
            A = ii_assemble(a)
            
            a[-1][i] += + qs[i]*Tphi*dx_edge
            print('>> Trying to assemble restricted element with global element')
            A = ii_assemble(a)
            
            a[i][-1] += - Tp*self.G.dds(vs[i])*dx_edge
            A = ii_assemble(a)
            
        
        assemble_mixed()
        
        # Add bifurcation condition
        edge_list = list(self.G.edges.keys())

        if False:
            for j, b in enumerate(self.G.bifurcation_ixs):
                lm_ix = self.G.num_edges + j
                
                for e in self.G.in_edges(j):
                    ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
                    edge_ix = edge_list.index(e)
                    a[lm_ix][edge_ix] += qs[edge_ix]*xis[j]*ds_edge(BIF_IN)
                    a[edge_ix][lm_ix] += vs[edge_ix]*lams[j]*ds_edge(BIF_IN)

                for e in self.G.out_edges(j):
                    ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
                    edge_ix = edge_list.index(e)
                    a[lm_ix][edge_ix] -= qs[edge_ix]*xis[j]*ds_edge(BIF_OUT)
                    a[edge_ix][lm_ix] -= vs[edge_ix]*lams[j]*ds_edge(BIF_OUT)
                
            
        return a 
   
    def L(self, vphi):
        '''
        Args:
            vphi (list of df.testfunction)
        '''
        
        # split out the components
        vs, xis, phi = vphi[0:self.G.num_edges], vphi[self.G.num_edges:-1], vphi[-1]
    
        dx = Measure('dx', domain=self.G.global_mesh)
        L = self.f*phi*dx

        # Assemble edge contributions to a and L
        for i, e in enumerate(self.G.edges):
            ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
            L += self.ns*vs[i]*ds_edge(BOUN_OUT) - self.ns*vs[i]*ds_edge(BOUN_IN)
       
        return L
    


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
    
    
    mesh = G.global_mesh

    # Flux spaces on each segment, ordered by the edge list
    submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
    P2s = [FunctionSpace(msh, 'CG', 2) for msh in submeshes] 
    
    # Real space on each bifurcation, ordered by G.bifurcation_ixs
    LMs = [FunctionSpace(mesh, 'R', 0) for b in G.bifurcation_ixs] 

    # Pressure space on global mesh
    P1 = FunctionSpace(mesh, 'CG', 1) # Pressure space (on whole mesh)
    
    ### Function spaces
    W = P2s + LMs + [P1]
    
    # Trial and test functions
    qp = list(map(TrialFunction, W))
    vphi = list(map(TestFunction, W))
    
    ## Assemble variational formulation 

    model = HydraulicNetwork_ii(G, f, p_bc)
    
    a = model.a(qp, vphi)
    L = model.L(vphi)
    
    A = ii_assemble(a)
    b = ii_assemble(L)
    
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
    
    G = make_line_graph(2)
    G.make_mesh(2)
    for e in G.in_edges(): G.in_edges()[e]['res'] = 1
    for e in G.out_edges(): G.out_edges()[e]['res'] = 1
    
    qp0 = hydraulic_network_simulation(G)