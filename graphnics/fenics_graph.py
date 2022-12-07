import networkx as nx
import numpy as np
from fenics import *
from xii import *

"""
The Graphnics class constructs fenics meshes from networkx directed graphs.
"""

# Marker tags for inward/outward pointing bifurcation nodes and boundary nodes
BIF_IN = 1
BIF_OUT = 2
BOUN_IN = 3
BOUN_OUT = 4


class FenicsGraph(nx.DiGraph):
    """
    Make fenics mesh from networkx directed graph

    Attributes:
        global_mesh (df.mesh): mesh for the entire graph
        edges[i].mesh (df.mesh): submesh for edge i
        global_tangent (df.function): tangent vector for the global mesh, points along edge
        dds (function): derivative d/ds along edge
        mf (df.function): 1d meshfunction that maps cell->edge number
        vf (df.function): 0d meshfunction on  edges[i].mesh that stores bifurcation and boundary point data
        num_edges (int): number of edges in graph
    """

    def __init__(self):
        nx.DiGraph.__init__(self)

        self.global_mesh = None

    def make_mesh(self, n=1, store_mesh=True):
        """
        Makes a fenics mesh on the graph with 2^n cells on each edge

        Args:
            n (int): number of refinements
        Returns:
            mesh (df.mesh): the global mesh
        """

        # Store the coordinate dimensions
        geom_dim = len(self.nodes[1]["pos"])
        self.geom_dim = geom_dim

        self.num_edges = len(self.edges)

        # Make list of vertex coordinates and the cells connecting them
        vertex_coords = np.asarray([self.nodes[v]["pos"] for v in self.nodes()])
        cells_array = np.asarray([[u, v] for u, v in self.edges()])

        # We first make a mesh with 1 cell per edge
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, "interval", 1, geom_dim)
        editor.init_vertices(len(vertex_coords))
        editor.init_cells(len(cells_array))

        [editor.add_vertex(i, xi) for i, xi in enumerate(vertex_coords)]
        [editor.add_cell(i, cell.tolist()) for i, cell in enumerate(cells_array)]

        editor.close()

        # Make meshfunction containing edge ixs
        mf = MeshFunction("size_t", mesh, 1)
        mf.array()[:] = range(0, len(self.edges()))

        # Refine global mesh until desired resolution
        for i in range(0, n):
            mesh = refine(mesh)
            mf = adapt(mf, mesh)

        if not store_mesh:  # then our job is done
            return mesh


        # Store refined global mesh and refined mesh function marking branches
        self.global_mesh = mesh
        self.mf = mf

        # Make and store one submesh for each edge
        for i, (u, v) in enumerate(self.edges):
            self.edges[u, v]["submesh"] = EmbeddedMesh(mf, i)
            
        
        # Give each edge a Meshfunction that marks the vertex if its a boundary node
        # or a bifurcation node
        # A bifurcation node is tagged BIF_IN if the edge points into it or BIF_OUT if the edge points out of it
        # A boundary node is tagged BOUN_IN if the edge points into it or BOUN_OUT if the edge points out of it

        # Initialize meshfunction for each edge
        for e in self.edges():
            msh = self.edges[e]["submesh"]
            vf = MeshFunction("size_t", msh, 0, 0)
            self.edges[e]["vf"] = vf

        # Make list of bifurcation nodes (connected to two or more edges)
        # and boundary nodes (connected to one edge)
        bifurcation_ixs = []
        boundary_ixs = []
        for v in self.nodes():

            num_conn_edges = len(self.in_edges(v)) + len(self.out_edges(v))

            if num_conn_edges == 1:
                boundary_ixs.append(v)
            elif num_conn_edges > 1:
                bifurcation_ixs.append(v)
            elif num_conn_edges == 0:
                print(f"Node {v} in G is lonely (i.e. unconnected)")

        # Store these as global variables
        self.bifurcation_ixs = bifurcation_ixs
        self.num_bifurcations = len(bifurcation_ixs)
        self.boundary_ixs = boundary_ixs

        # Mark the meshfunction belonging to each edge with
        # - BIF_IN/BIF_OUT if the vertex belongs to an edge that goes in/out of the bif node
        self.mark_edge_vfs(self.bifurcation_ixs, BIF_IN, BIF_OUT)
        # - BOUN_OUT/BOUN_IN if the vertex is an inlet or outlet point
        self.mark_edge_vfs(self.boundary_ixs, BOUN_OUT, BOUN_IN)
        # Note: We need to reverse our notation here so that BOUN_OUT is applied to
        # edges going into a boundary node (which is actually an outlet)

        # Compute tangent vectors
        self.assign_tangents()


    def compute_edge_lengths(self):
        """
        Compute and store the length of each edge
        """

        for e in self.edges():
            v1, v2 = e
            dist = np.linalg.norm(
                np.asarray(self.nodes()[v2]["pos"])
                - np.asarray(self.nodes()[v1]["pos"])
            )
            self.edges()[e]["length"] = dist

    def assign_tangents(self):
        """
        Assign a tangent vector list to each edge in the graph
        The tangent vector lists are stored
            * for each edge in G.edges[i]['tangent']
            * as a lookup dictionary in G.tangents
            * as a fenics function in self.global_tangent
        """

        for u, v in self.edges():
            tangent = np.asarray(self.nodes[v]["pos"]) - np.asarray(
                self.nodes[u]["pos"], dtype=np.float64
            )
            tangent_norm = np.linalg.norm(tangent)
            tangent_norm_inv = 1.0 / tangent_norm
            tangent *= tangent_norm_inv
            self.edges[u, v]["tangent"] = tangent
        self.tangents = list(nx.get_edge_attributes(self, "tangent").items())

        tangent = TangentFunction(self, degree=1)
        tangent_i = interpolate(
            tangent, VectorFunctionSpace(self.global_mesh, "DG", 0, self.geom_dim)
        )
        self.global_tangent = tangent_i

    def dds(self, f):
        """
        function for derivative df/ds along graph
        """
        return dot(grad(f), self.global_tangent)

    def dds_i(self, f, i):
        """
        function for derivative df/ds along graph on branch i
        """
        tangent = self.tangents[i][1]
        return dot(grad(f), Constant(tangent))

    def get_num_inlets_outlets(self):
        num_inlets, num_outlets = 0, 0

        for e in self.edges():
            vf_vals = self.edges[e]["vf"].array()

            num_inlets += len(list(np.where(vf_vals == BOUN_IN)[0]))
            num_outlets += len(list(np.where(vf_vals == BOUN_OUT)[0]))

        return num_inlets, num_outlets


    def mark_edge_vfs(self, node_ixs, tag_in, tag_out):
        """ Mark the meshfunction belonging to the edges submesh 
        with tag_in if the edge goes into a node in node_ixs
        and tag_out if the edge goes out of a node in node_ixs

        Args:
            node_ixs (list): _description_
            edges (func): _description_
            tag (int): corresponding tag 
        """
        for n in node_ixs:

            for e in self.in_edges(n):
                msh = self.edges[e]["submesh"]
                vf = self.edges[e]["vf"]

                node_ix_in_submesh = np.where(
                    (msh.coordinates() == self.nodes[n]["pos"]).all(axis=1)
                )[0]
                if len(node_ix_in_submesh) > 0:
                    vf.array()[node_ix_in_submesh[0]] = tag_in


            for e in self.out_edges(n):
                msh = self.edges[e]["submesh"]
                vf = self.edges[e]["vf"]

                node_ix_in_submesh = np.where(
                    (msh.coordinates() == self.nodes[n]["pos"]).all(axis=1)
                )[0]
                if len(node_ix_in_submesh) > 0:
                    vf.array()[node_ix_in_submesh[0]] = tag_out


class GlobalFlux(UserExpression):
    """
    Evaluates P2 flux on each edge
    """

    def __init__(self, G, qs, **kwargs):
        """
        Args:
            G (nx.graph): Network graph
            qs (list): list of fluxes on each edge in the branch
        """

        self.G = G
        self.qs = qs
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        edge = self.G.mf[cell.index]
        tangent = self.G.tangents[edge][1]
        values[0] = self.qs[edge](x) * tangent[0]
        values[1] = self.qs[edge](x) * tangent[1]
        if self.G.geom_dim == 3:
            values[2] = self.qs[edge](x) * tangent[2]

    def value_shape(self):
        return (self.G.geom_dim,)


class GlobalCrossectionFlux(UserExpression):
    """
    Evaluates P2 flux on each edge
    """

    def __init__(self, G, qs, **kwargs):
        """
        Args:
            G (nx.graph): Network graph
            qs (list): list of fluxes on each edge in the branch
        """

        self.G = G
        self.qs = qs
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        edge = self.G.mf[cell.index]
        values[0] = self.qs[edge](x)
    
    def value_shape(self):
        return ()


class TangentFunction(UserExpression):
    """
    Tangent expression for graph G, which is
    constructed from G.tangents
    """

    def __init__(self, G, degree, **kwargs):
        """
        Args:
            G (nx.graph): Network graph
            degree (int): degree of resulting expression
        """
        self.G = G
        self.degree = degree
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        edge = self.G.mf[cell.index]
        values[0] = self.G.tangents[edge][1][0]
        values[1] = self.G.tangents[edge][1][1]
        if self.G.geom_dim == 3:
            values[2] = self.G.tangents[edge][1][2]

    def value_shape(self):
        return (self.G.geom_dim,)


def copy_from_nx_graph(G_nx):
    """
    Return deep copy of nx.Graph as FenicsGraph
    Args:
        G_nx (nx.Graph): graph to be coped
    Returns:
        G (FenicsGraph): fenics type graph with nodes and egdes from G_nx
    """

    G = FenicsGraph()
    G.graph.update(G_nx.graph)
    
    # copy nodes and edges
    G.add_nodes_from((n, d.copy()) for n, d in G_nx._node.items())
    for u, v in G_nx.edges():
        G.add_edge(u, v)
    
    # copy node attributes
    for n in G.nodes():
        node_dict = G_nx.nodes()[n]
        for key in node_dict:
            G.nodes()[n][key] = G_nx.nodes()[n][key]
            
    # copy edge attributes
    for e in G.edges():
        edge_dict = G_nx.edges()[e]
        for key in edge_dict:
            G.edges()[e][key] = G_nx.edges()[e][key]
        

    return G
