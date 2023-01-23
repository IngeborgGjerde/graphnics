import networkx as nx
from .fenics_graph import *
from vtk import *
import os

'''
Overloaded File class for writing .vtp files for functions defined on the graph
This allows for using the TubeFilter in paraview

TODO: Allow for writing time-dependent functions
'''

class TubeRadius(UserExpression):
    '''
    Expression that evaluates FenicsGraph 'radius' at each point
    '''
    
    def __init__(self, G, **kwargs):
        self.G = G
        super().__init__(**kwargs)
    def eval_cell(self, value, x, cell):
        edge_ix = self.G.mf[cell.index]
        edge = list(self.G.edges())[edge_ix]
        value[0] = self.G.edges()[edge]['radius']


class TubeFile(File):
    def __init__(self, G, fname, **kwargs):
        """
        .vtp file with network function and radius, made for TubeFilter in paraview
        
        Args:
            G (FenicsGraph): graph with mesh
            fname (str): location and name for file
            
        Usage:
        >> G = make_Y_bifurcation(dim=3)
        >> V = FunctionSpace(G.global_mesh, 'CG', 1)
        >> f_i = interpolate(Expression('x[0]', degree=2))
        >> f_i.rename('f', '0.0')
        >> TubeFile(G, 'test.vtp') << f_i  
        """
        
        f_name, f_ext = os.path.splitext(fname)
        assert f_ext == '.vtp', 'TubeFile must have .vtp file ending'
        
        self.fname = fname
        self.G = G
        
        
    def __lshift__(self, func):
        """
        Write function to .vtp file
        
        Args:
            - func: function to plot
            - radius (function): network radius
        """
        
        assert self.G.geom_dim==3, f'Coordinates are {self.G.geom_dim}d, they need to be 3d'
        assert len(nx.get_edge_attributes(self.G, 'radius'))>0, 'Graph must have radius attribute'
        
        radius = TubeRadius(self.G, degree=2)
        radius_i = interpolate(radius, FunctionSpace(self.G.global_mesh, 'CG', 1))
        
        
        # Store points in vtkPoints
        coords = self.G.global_mesh.coordinates()
        points = vtkPoints()
        for c in coords:
            points.InsertNextPoint(list(c))

        # Store edges in cell array
        lines = vtkCellArray()
        edge_to_vertices = self.G.global_mesh.cells()

        for vs in edge_to_vertices: 
            line = vtkLine()
            line.GetPointIds().SetId(0, vs[0])
            line.GetPointIds().SetId(1, vs[1])
            lines.InsertNextCell(line)

        # Create a polydata to store 1d mesh in
        linesPolyData = vtkPolyData()
        linesPolyData.SetPoints(points)
        linesPolyData.SetLines(lines)


        # Write data from associated function
        data = vtkDoubleArray()
        data.SetName(func.name())
        data.SetNumberOfComponents(1)
        
        # store value of function at each coordinates
        for c in coords:
            data.InsertNextTuple([func(c)])

        linesPolyData.GetPointData().AddArray(data)

        
        # Write radius data
        data = vtkDoubleArray()
        data.SetName('radius')
        data.SetNumberOfComponents(1)
        
        # store value of function at each coordinates
        for c in coords:
            data.InsertNextTuple([radius_i(c)])

        linesPolyData.GetPointData().AddArray(data)

        # Write to file
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(self.fname)
        writer.SetInputData(linesPolyData)
        writer.Update()
        writer.Write()
        
    