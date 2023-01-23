import networkx as nx
from .fenics_graph import *
from vtk import *
import os

'''
Overloaded File class for writing .vtp files for functions defined on the graph
This allows for using the TubeFilter in paraview

TODO: Allow for writing time-dependent functions
'''

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
        >> radius_i = interpolate(Expression('x[1]+0.1*x[0]', degree=2), V)
        >> f_i = interpolate(Expression('x[0]', degree=2))
        >> f_i.rename('f', '0.0')
        >> TubeFile(G, 'test.vtp') << (val, radius_i)    
        """
        
        f_name, f_ext = os.path.splitext(fname)
        assert f_ext == '.vtp', 'TubeFile must have .vtp file ending'
        
        self.fname = fname
        self.G = G
        
        
    def __lshift__(self, func_and_radius):
        """
        Write function to .vtp file
        
        Args:
            func_and_radius (tuple):
                - func: function to plot
                - radius (function): network radius
        """
        
        func, radius = func_and_radius
        
        assert self.G.geom_dim==3, f'Coordinates are {self.G.geom_dim}d, they need to be 3d'
        
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
            data.InsertNextTuple([func(c)])

        linesPolyData.GetPointData().AddArray(data)

        # Write to file
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(self.fname)
        writer.SetInputData(linesPolyData)
        writer.Update()
        writer.Write()