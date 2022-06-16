from fenics import *

def read_h5_mesh(loc):
    meshf = HDF5File(MPI.comm_world, loc, "r")
    mesh = Mesh()
    meshf.read(mesh, varname, False)
    meshf.close()  
    return mesh

def read_h5_func(W, loc, varname):
    upf = HDF5File(W.mesh().mpi_comm(), loc, "r")
    up = Function(W)
    upf.read(up, varname)
    upf.close()
    return up


def write_HDF5file(var, mesh, fname, varname):
    file = HDF5File(mesh.mpi_comm(), fname, "w")
    file.write(var, varname)
    file.close()