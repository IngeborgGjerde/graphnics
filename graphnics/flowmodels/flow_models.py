import networkx as nx
from fenics import *
from xii import *
import sys

sys.path.append("../")
from graphnics import *


RT = {
    "flux_space": "CG",
    "flux_degree": 1,
    "pressure_space": "DG",
    "pressure_degree": 0,
}


TH = {
    "flux_space": "CG",
    "flux_degree": 2,
    "pressure_space": "CG",
    "pressure_degree": 1,
}   

class NetworkPoisson:
    """
    Bilinear forms a and L for the Poisson equation on the network

    (kappa \partial_s u, \partial_s v) + (beta u, \partial_s v) = (f,v)
    """

    def __init__(self, G, f=Constant(0), p_bc=Constant(0), kappa=Constant(1), degree=1):

        # Graph on which the model lives
        self.G = G

        # Model parameters
        self.f = f
        self.p_bc = p_bc
        self.kappa = kappa

        self.V = FunctionSpace(G.global_mesh, "CG", degree)

    def a_form(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        G = self.G

        a = self.kappa * inner(G.dds(u), G.dds(v)) * dx

        return a

    def L_form(self):
        v = TestFunction(self.V)
        return self.f * v * dx

    def get_bc(self):
        bc = DirichletBC(self.V, self.p_bc, "on_boundary")
        return bc



class PrimalMixedHydraulicNetwork:
    """
    Bilinear forms a and L for the hydraulic equations
            Res*q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous pressure
    """

    def __init__(self, G, f=Constant(0), p_bc=Constant(0), Res=Constant(1), degree=1):

        # Graph on which the model lives
        self.G = G

        # Model parameters
        self.f = f
        self.p_bc = p_bc
        self.Res = Res

        self.W = [FunctionSpace(G.global_mesh, "DG", degree-1),
                  FunctionSpace(G.global_mesh, "CG", degree)]


        self.qp = list(map(TrialFunction, self.W))
        self.vphi = list(map(TestFunction, self.W))
        
        self.num_flux_spaces = 1

    def a_form(self):
        
        G = self.G
        q, p = self.qp
        v, phi = self.vphi

        a = [[0 for i in range(0, len(self.qp))] for j in range(0, len(self.qp))]
        
        a[0][0] += self.Res * inner(q, v) * dx
        a[0][1] += inner(G.dds(p), v) * dx
        a[1][0] += inner(G.dds(phi), q) * dx

        return a


    def L_form(self):
        v, phi = self.vphi
        L = [0, 0]
        L[0] = Constant(0)*v*dx
        L[1] = self.f*phi*dx
        
        return L
    
    def get_bc(self):
        bcs = [[], [DirichletBC(self.W[1], self.p_bc, "on_boundary")]]
        return bcs
    
    
    def solve(self):
        
        W = self.W
        a = self.a_form()
        L = self.L_form()

        W_bcs = self.get_bc()
        
        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, W_bcs)
        A, b = map(ii_convert, (A, b))

        
        qp = ii_Function(W)
        solver = LUSolver(A, "mumps")
        solver.solve(qp.vector(), b)
        
        
        return qp






class MixedHydraulicNetwork:
    """
    Bilinear forms a and L for the dual mixed form of the hydraulic equations
            Res*q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous
    normal stress

    Args:
        G (FenicsGraph): Network domain
        Res (dict): dictionary with edge->resistance
        f (df.expr): source term
        p_bc (df.expr): pressure boundary condition
    """

    def __init__(self, G, f=Constant(0), p_bc=Constant(0), space=RT):
        """
        Set up function spaces and store model parameters f and ns
        """

        # Graph on which the model lives
        self.G = G

        # Model parameters

        self.f = f
        self.p_bc = p_bc

        # Setup function spaces:
        # - flux space on each segment, ordered by the edge list
        # - pressure space on the full mesh
        # - real space on each bifurcation
        submeshes = list(nx.get_edge_attributes(G, "submesh").values())

        P2s = [
            FunctionSpace(msh, space["flux_space"], space["flux_degree"])
            for msh in submeshes
        ]
        P1s = [
            FunctionSpace(
                G.global_mesh, space["pressure_space"], space["pressure_degree"]
            )
        ]
        LMs = [FunctionSpace(G.global_mesh, "R", 0) for b in G.bifurcation_ixs]

        ### Function spaces
        W = P2s + P1s + LMs
        self.W = W

        self.meshes = submeshes + [G.global_mesh] * (
            G.num_bifurcations + 1
        )  # associated meshes

        # Trial and test functions
        self.qp = list(map(TrialFunction, W))
        self.vphi = list(map(TestFunction, W))

    def diag_a_form_on_edges(self, a=None):
        """
        Add edge contributions to the bilinear form
        """

        if not a:
            a = self.init_a_form()

        qp, vphi = self.qp, self.vphi
        G = self.G

        # split out the components
        n_edges = G.num_edges

        qs, vs = qp[:n_edges], vphi[:n_edges]

        # edge contributions to form
        for i, e in enumerate(G.edges):
            dx_edge = Measure("dx", domain=G.edges[e]["submesh"])
            
            try: 
                res = self.G.edges()[e]["Res"]
            except KeyError:
                res = Constant(1)

            a[i][i] += res * qs[i] * vs[i] * dx_edge

        return a

    def offdiag_a_form_on_edges(self, a=None):
        """
        Add edge contributions to the bilinear form
        """

        if not a:
            a = self.init_a_form()

        qp, vphi = self.qp, self.vphi
        G = self.G

        # split out the components
        n_edges = G.num_edges

        qs, vs = qp[:n_edges], vphi[:n_edges]
        p, phi = qp[n_edges], vphi[n_edges]

        submeshes = list(nx.get_edge_attributes(G, "submesh").values())
        ps = [Restriction(p, msh) for msh in submeshes]
        phis = [Restriction(phi, msh) for msh in submeshes]

        # edge contributions to form
        for i, e in enumerate(G.edges):

            dx_edge = Measure("dx", domain=G.edges[e]["submesh"])

            a[n_edges][i] += +G.dds_i(qs[i], i) * phis[i] * dx_edge
            a[i][n_edges] += -ps[i] * G.dds_i(vs[i], i) * dx_edge

        return a

    def a_form_on_bifs(self, a=None):
        """
        Bifurcation point contributions to bilinear form a
        """

        if not a:
            a = self.init_a_form()

        qp, vphi = self.qp, self.vphi
        G = self.G

        # split out the components
        n_edges = G.num_edges

        qs, lams = qp[0:n_edges], qp[n_edges + 1 :]
        vs, xis = vphi[0:n_edges], vphi[n_edges + 1 :]

        edge_list = list(G.edges.keys())

        # bifurcation condition contributions to form
        for b_ix, b in enumerate(G.bifurcation_ixs):

            # make info dict of edges connected to this bifurcation
            conn_edges = {
                **{e: (1, BIF_IN, edge_list.index(e)) for e in G.in_edges(b)},
                **{e: (-1, BIF_OUT, edge_list.index(e)) for e in G.out_edges(b)},
            }

            for e in conn_edges:
                ds_edge = Measure(
                    "ds", domain=G.edges[e]["submesh"], subdomain_data=G.edges[e]["vf"]
                )

                sign, tag, e_ix = conn_edges[e]

                a[e_ix][n_edges + 1 + b_ix] += (
                    sign * vs[e_ix] * lams[b_ix] * ds_edge(tag)
                )
                a[n_edges + 1 + b_ix][e_ix] += (
                    sign * qs[e_ix] * xis[b_ix] * ds_edge(tag)
                )

        return a

    def init_a_form(self):
        """
        Init a
        """

        ## Init a as list of lists
        a = [[0 for i in range(0, len(self.qp))] for j in range(0, len(self.qp))]

        # Init zero diagonal elements (for shape info)
        for i, msh in enumerate(self.meshes):
            a[i][i] += (
                Constant(0) * self.qp[i] * self.vphi[i] * Measure("dx", domain=msh)
            )

        return a

    def a_form(self):
        """
        The bilinear form

        Args:
            Res (dict): dictionary with edge->resistance
        """

        a = self.init_a_form()
        a = self.diag_a_form_on_edges(a)
        a = self.offdiag_a_form_on_edges(a)
        a = self.a_form_on_bifs(a)

        return a

    def init_L_form(self):
        """
        Init L
        """

        L = [0 for i in range(0, len(self.vphi))]

        # Init zero diagonal elements (for shape info)
        for i, msh in enumerate(self.meshes):
            dx = Measure("dx", domain=msh)
            L[i] += Constant(0) * self.vphi[i] * dx

        return L

    def L_form(self):
        """
        The right-hand side linear form
        """

        L = self.init_L_form()

        vphi = self.vphi

        # split out the components
        n_edges = self.G.num_edges
        vs, phi, xis = vphi[0:n_edges], vphi[n_edges], vphi[n_edges + 1 :]

        submeshes = list(nx.get_edge_attributes(self.G, "submesh").values())
        phis = [Restriction(phi, msh) for msh in submeshes]
        fs = [project(self.f, FunctionSpace(msh, 'CG', 1)) for msh in submeshes]
        
        # Assemble edge contributions to a and L
        for i, e in enumerate(self.G.edges):
            ds_edge = Measure(
                "ds",
                domain=self.G.edges[e]["submesh"],
                subdomain_data=self.G.edges[e]["vf"],
            )
            dx_edge = Measure("dx", domain=self.G.edges[e]["submesh"])

            L[i] += self.p_bc * vs[i] * ds_edge(BOUN_OUT) - self.p_bc * vs[i] * ds_edge(BOUN_IN)
            L[n_edges] += fs[i] * phis[i] * dx_edge

        for i in range(0, len(self.G.bifurcation_ixs)):
            L[n_edges + 1 + i] += Constant(0) * xis[i] * dx

        return L
    
    def solve(self):
        
        W = self.W
        a = self.a_form()
        L = self.L_form()

        A, b = map(ii_assemble, (a, L))
        A, b = map(ii_convert, (A, b))

        qp = ii_Function(W)
        solver = LUSolver(A, "mumps")
        solver.solve(qp.vector(), b)
        
        return qp





class NetworkStokes(MixedHydraulicNetwork):
    """
    Bilinear forms a and L for the hydraulic equations
            R*q + d^2/ds^2 q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous
    normal stress

    Args:
        G (FenicsGraph): Network domain
        Res (dict): dictionary with edge->resistance
        nu (df.expr): viscosity
        f (df.expr): source term
        p_bc (df.expr): pressure boundary condition
    """

    def __init__(self, G, mu=Constant(1), f=Constant(0), p_bc=Constant(0), space=TH):
        self.mu = mu
        super().__init__(G, f, p_bc, space)

    def diag_a_form_on_edges(self, a=None):
        """
        The bilinear form
        """

        if not a:
            a = self.init_a_form()

        qp, vphi = self.qp, self.vphi
        G = self.G

        # split out the components
        n_edges = G.num_edges

        qs, vs = qp[0:n_edges], vphi[0:n_edges]

        # edge contributions to form
        for i, e in enumerate(G.edges):

            dx_edge = Measure("dx", domain=G.edges[e]["submesh"])
            
            try: 
                Res = self.G.edges()[e]["Res"]
                Ainv = self.G.edges()[e]["Ainv"]
            except KeyError:
                Res = Constant(1)
                Ainv = Constant(1)

            a[i][i] += Res * qs[i] * vs[i] * dx_edge
            a[i][i] += self.mu * Ainv * G.dds_i(qs[i], i) * G.dds_i(vs[i], i) * dx_edge

        return a
