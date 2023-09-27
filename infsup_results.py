
from graphnics import *
import sys
sys.path.append('../other/')
from infsup_eigvals import *
sys.path.append('../graphnics/data')
from generate_arterial_tree import *
import imp
import infsup_eigvals



imp.reload(infsup_eigvals)


print('n_bifurcations \\ N_refs \\\\')
import tabulate

history_lmin, history_lmax, history_cond = [], [], []
for N in [3, 4, 5, 6, 7, 8]:
    # G = make_arterial_tree(N, uniform_lengths=False)
    #G = honeycomb(N, N)
    G = line_graph(N)   # What happens with CG1 x DG0 a la Hdiv x L2?
    # G = YY_bifurcation(N)
    strr = ''

    lmins, lmaxs, conds = [G.num_bifurcations], [G.num_bifurcations], [G.num_bifurcations]
    ns = [1]
    for n in ns:
    
        G.make_mesh(n)
        G.make_submeshes()
        G.compute_vertex_degrees()

        mesh = G.mesh
        x = mesh.coordinates()
        xmin, ymin = np.min(x, axis=0)[:2]
        xmax, ymax = np.max(x, axis=0)[:2]

        print(f'[{xmin}, {xmax}] x [{ymin}, {ymax}]')

        if mesh.geometry().dim() == 3:
            shift = np.array([[xmin, ymin, 0]])
        else:
            shift = np.array([[xmin, ymin]])
        x -= shift

        box_length = max(xmax - xmin, ymax - ymin)
        x /= box_length

        x = G.mesh.coordinates()
        xmin, ymin = np.min(x, axis=0)[:2]
        xmax, ymax = np.max(x, axis=0)[:2]            
        print(f'[{xmin}, {xmax}] x [{ymin}, {ymax}]')
        
        length = interpolate(Constant(1), FunctionSpace(G.mesh, 'CG', 1))
        length = norm(length)**2

        # length = sum(c.volume() for c in cells(G.mesh))
        we = Constant(10/length)
        wv = Constant(length**0.5*we)

        we = Constant(1)
        wv = Constant(1)
        model = infsup_eigvals.MixedHydraulicNetwork_EP(G)

        print(f'N = {N} n = {n} {G.mesh.num_cells()} {len(model.W)}')
        # print([Wi.ufl_element() for Wi in model.W])
        # print(G.bifurcation_ixs)
        # File(f'fooN{N}n{n}.pvd') << G.mesh
        # model = infsup_eigvals.HydraulicNetwork_EP(G)
        assert G.num_bifurcations > 0

        
        a = model.a_form()
        A = ii_assemble(a)

        blocks = A.blocks

        for row in blocks[-G.num_bifurcations:]:
            for col, mat in enumerate(row):
                if not isinstance(mat, int):
                    print(col, ii_convert(mat).array())
        
        a_r = model.b_form_eigval(w_e=we, w_v=wv)
        # a_r = model.b_form_eigval()
        
        B = ii_assemble(a_r)

        # from IPython import embed
        # embed()
        
        W_bcs = model.get_bc()
        A, _ = apply_bc(A, b=None, bcs=W_bcs)

        # from IPython import embed
        # embed()
        
        B, _ = apply_bc(B, b=None, bcs=W_bcs)                

        A, B = map(ii_convert, (A, B))
        
        lmin, lmax = solve_infsup_eigproblem(A, B, model)
        
        lmins.append(lmin)
        lmaxs.append(lmax)
        conds.append(lmax/abs(lmin))
    history_lmin.append(lmins)
    history_lmax.append(lmaxs)
    history_cond.append(conds)
    
    print(tabulate.tabulate(history_lmin, headers=['Nb'] + list(map(str, ns))))
    print(tabulate.tabulate(history_lmax, headers=['Nb'] + list(map(str, ns))))
    print(tabulate.tabulate(history_cond, headers=['Nb'] + list(map(str, ns))))        


