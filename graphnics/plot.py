import networkx as nx
from .fenics_graph import *
from vtk import *

'''
Write .vtp files for functions defined on the graph
'''


def write_vtp(G, functions=[], fname='plot.vtp'):
    '''
    Write file for plotting graph mesh and associated scalar fields in paraview
    
    Args:
        G (FenicsGraph): graph
        functions (list of tuples): fenics functions to plot and their name
        fname (str): file name for plot
        
    Example:
        >> G = make_double_Y_bifurcation(dim=3)
        >> radius = Expression('x[1]+0.1*x[0]', degree=2)
        >> val = Expression('x[0]', degree=2)
        >> write_vtp(G, functions=[(radius, 'radius'), (val, 'val')])
        
    The function values are assigned at each vertex.
    '''
    
    # Store points in vtkPoints
    coords = G.global_mesh.coordinates()
    points = vtkPoints()
    for c in coords:
        points.InsertNextPoint(list(c))

    # Store edges in cell array
    lines = vtkCellArray()
    edge_to_vertices = G.global_mesh.cells()

    for vs in edge_to_vertices: 
        line = vtkLine()
        line.GetPointIds().SetId(0, vs[0])
        line.GetPointIds().SetId(1, vs[1])
        lines.InsertNextCell(line)

    # Create a polydata to store 1d mesh in
    linesPolyData = vtkPolyData()
    linesPolyData.SetPoints(points)
    linesPolyData.SetLines(lines)


    # Add data from associated functions
    for func, name in functions:
        data = vtkDoubleArray()
        data.SetName(name)
        data.SetNumberOfComponents(1)

        # Store value of function at each coordinates
        for c in coords:
            data.InsertNextTuple([func(c)])

        linesPolyData.GetPointData().AddArray(data)


    # Write to file
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(linesPolyData)
    writer.Update()
    writer.Write()

