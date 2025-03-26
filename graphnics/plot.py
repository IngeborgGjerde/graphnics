'''
Copyright (C) 2022-2023 by Ingeborg Gjerde

This file is a part of the graphnics project (https://arxiv.org/abs/2212.02916)

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.
'''

import networkx as nx
from .fenics_graph import *
from vtk import *
import os

'''
Overloaded File class for writing .vtp files for functions defined on the graph
This allows for using the TubeFilter in paraview
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
            G (FenicsGraph): graph with mesh and radius attribute on edges
            fname (str): location and name for file
            
        Example:
        >> G = Y_bifurcation(dim=3)
        >> for e in G.edges():
        >>      G.edges()[e]['radius'] = 1
        >>  V = FunctionSpace(G.mesh, 'CG', 1)
        >>  f_i = interpolate(Expression('x[0]', degree=2), V)
        >>  f_i.rename('f', '0.0')
        >>  TubeFile(G, 'test.pvd') << f_i  
        """
        super().__init__(fname)
        f_name, f_ext = os.path.splitext(fname)
        assert f_ext == '.pvd', 'TubeFile must have .pvd file ending'
        
        self.fname = f_name
        self.G = G
        
        assert self.G.geom_dim==3, f'Coordinates are {self.G.geom_dim}d, they need to be 3d'
        assert len(nx.get_edge_attributes(self.G, 'radius'))>0, 'Graph must have radius attribute'
        
        # Make pvd file containing header and footer
        pvdfile = open(fname, "w")
        pvdfile.write(pvd_header + pvd_footer) 
        pvdfile.close()
        
        
        
    def __lshift__(self, func):
        """
        Write function to .vtp file
        
        Args:
            - func: tuple with function and time step
            If only function is given, time step is set to 0 
        """
        
        if type(func) is tuple:
            func, i = func
        else:
            i = 0    

        #radius = TubeRadius(self.G, degree=2)
        #radius = interpolate(radius, FunctionSpace(self.G.mesh, 'CG', 1))
        
        radius_dict = nx.get_edge_attributes(self.G, 'radius')
        mesh0, foo = self.G.get_mesh(0)
        DG = FunctionSpace(mesh0, 'DG', 0)
        radius = Function(DG)
        radius.vector()[:] = list(radius_dict.values())
        radius.set_allow_extrapolation(True)
                
        ### Write vtp file for this time step
        
        # Store points in vtkPoints
        coords = self.G.mesh.coordinates()
        points = vtkPoints()
        for c in coords:
            points.InsertNextPoint(list(c))

        # Store edges in cell array
        lines = vtkCellArray()
        edge_to_vertices = self.G.mesh.cells()

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
            data.InsertNextTuple([radius(c)])

        linesPolyData.GetPointData().AddArray(data)

        # Write to file
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(f"{self.fname}{int(i):06d}.vtp")
        writer.SetInputData(linesPolyData)
        writer.Update()
        writer.Write()
        
        
        ### Update pvd file with vtp file for this time step
        pvdfile = open(self.fname+ ".pvd", "r")
        content = pvdfile.read().splitlines()
        
        # add entry before footer
        short_fname = self.fname.split('/')[-1]
        
        pvd_entry = f"<DataSet timestep=\"{i}\" part=\"0\" file=\"{short_fname}{int(i):06d}.vtp\" />"
        updated_content = content[:-2] + [pvd_entry] + content[-2:] 
        updated_content = "\n".join(updated_content)# convert to string
        
        pvdfile = open(self.fname + '.pvd', "w")
        pvdfile.write(updated_content)
        pvdfile.close()
        
        
    
pvd_header = """<?xml version=\"1.0\"?>
<VTKFile type="Collection" version=\"0.1\">
  <Collection>\n"""

pvd_footer= """</Collection>
</VTKFile>"""
