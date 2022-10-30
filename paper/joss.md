---
title: 'Graphnics: Combining networkx and FEniCS to solve network models'
tags:
  - Python
  - networks
  - medicine
authors:
  - name: Ingeborg Gjerde
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Ingeborg Gjerde, Simula Research Laboratory
   index: 1
date: December 2022
bibliography: paper.bib
link-citations: true
---

# Summary

Network models facilitate inexpensive simulations, but require careful
handling of bifurcation conditions. In this software we combine
[networkx]{.smallcaps} [@hagberg2008exploring] and [FEniCS]{.smallcaps}
[@AlnaesBlechta2015a] into the [graphnics]{.smallcaps} library, which
offers

-   A `FenicsGraph` class built on top of the [networkx]{.smallcaps}
    `DiGraph` class, that constructs a global mesh for a network and
    provides [fenics]{.smallcaps} mesh functions describing how they
    relate to the graph structure.

-   Example models showing how the `FenicsGraph` class can be used to
    assemble and solve different network flow models.

The example models are implemented using
[fenics]{.smallcaps}$\_$[ii]{.smallcaps} [@kuchta2021assembly] and the
mixed-dimensional branch of [FEniCS]{.smallcaps}
[@daversincatty2019abstractions]. Only minor adaptions to the code are
necessary to convert between the two, as the assembly uses the block
structure of the problem.


# Statement of need
[FEniCS]{.smallcaps} [@AlnaesBlechta2015a] provides high-level
functionality for specifying variational forms. This allows the user to
focus on the model they are solving rather than implementational
details. A key component of network models is the correct handling of
bifurcation conditions, one being that there should be no jump in the
cross-section flux (i.e. conversation of mass at junctions).
[FEniCS]{.smallcaps} implicitly assumes that each vertex is connected to
two cells. At bifurcation vertices in a network, however, the vertex is
connected to three or more cells. Thus one cannot use the jump operator
currently offered in [FEniCS]{.smallcaps}. The manual assembly of the
jump terms is highly prone to errors once the network becomes
non-trivial.

In this software we extend the `DiGraph` class offered by
[networkx]{.smallcaps} so that it (i) creates a global mesh of the
network and (ii) creates data structures describing how this mesh is
connected to the graph structure of the problem. In addition to this we
provide convenience functions that then it make it straightforward to
assemble and solve network flow models using the finite element method.

# Mathematics

Let $\mathsf{G}=(\mathsf{V}, \mathsf{E})$ denote a graph $\mathsf{G}$
with edges $\mathsf{E}$ and vertices $\mathsf{V}$. We let $\Lambda_i$ be
the geometrical domain associated with the edge $e_i$.

We want to solve flow models on networks, the simplest example being the
hydraulic network model 
$$\begin{aligned}
    \mathcal{R} Q_i + \partial_s P_i &= 0 \text{ on } \Lambda_i && \text{(constitutive equation on edge)} \\ 
    \partial_s Q_i &= f \text{ on } \Lambda_i  && \text{(conservation of mass on edge)}\\ 
    [[Q]]_b &= 0 \text{ for } b \in \mathsf{B}  && \text{(conservation of mass at bifurcation)}
\end{aligned}$$
where $Q_i$ denotes the cross-section flux along an edge, $P_i$ denotes
the pressure on an edge, $\mathcal{R}$ denotes the flow resistance and
$\partial_s$ denotes the spatial derivative along the (one-dimensional)
edge. Further 
$$\begin{aligned}
[Q]_b =  \sum_{i \in \mathsf{E}_{in}(b)} Q_i(b) - \sum_{i \in \mathsf{E}_{out}(b)} Q_i(b)
\end{aligned}$$
is used to denote the jump in cross-section flux over a bifurcation. To
close the system we further assume the pressure is continuous over each
bifurcation point.

The variational form associated with this model reads: Find $Q \in V$
and $(P, \lambda) \in M$ such that
$$\begin{aligned}
    a(Q,\psi)+b(\psi,(P, \lambda)) &= 0 \\
    b((\phi, \xi), Q) &= (f, \phi)
\end{aligned}$$ 
for all $\psi \in V$ and $(\phi, \xi) \in M$, where 
$$\begin{aligned}
    a(Q, \psi) &= \sum_{i=1}^n (\mathcal{R} Q, \Psi)_{\Lambda_i}. \\
    b(\psi, (P, \lambda)) &= -(\partial_s \psi, P) + \sum_{b \in \mathsf{B}} [\Psi]_b \lambda_b.
\end{aligned}$$
Here
$\lambda=(\lambda_1, \lambda_2, \lambda_3, ..., \lambda_m) \in \mathbf{R}^{\vert \mathsf{B} \vert}$
is a Lagrange multiplier used to impose conservation of mass at each bifurcation point.

# Software

The `FenicsGraph` class 
-----------------------

The main component of [graphnics]{.smallcaps} is the `FenicsGraph`
class, which inherts from the [networkx]{.smallcaps} `DiGraph` class.
The `FenicsGraph` class provides a function for meshing the network;
meshfunctions are used to relate the graph structure to the cells and
vertices in the mesh. Tangent vectors $\boldsymbol{\tau}_i$ are computed
for each edge and stored as edge attributes for the network. This is
then used in a convenience class function **dds$\_$i** which returns the
spatial derivative
$\partial_s f_i = \nabla f_i \cdot \boldsymbol{\tau}_i$ on the edge.

Network models
--------------

[Graphnics]{.smallcaps} can further be used to create and solve network flow model.
If the problem is posed in terms of global variables this can be done using standard methods in
[FEniCS]{.smallcaps}; this is demoed in the NetworkPoisson model. For
problems that are posed in terms of edge and vertex variables, the edge
and vertex iterators in [Networkx]{.smallcaps} are used to assemble
their contributions to the block matrix. This is demoed in the
HydraulicNetwork and NetworkStokes models.

<p float="left">
<img src="pial_pressure.png" alt="drawing" width="420"/>
<img src="pial_flux.png" alt="drawing" width="400"/>
</p>
<figcaption> Figure 1: Simulation of fluid flow in the pial blood vessel network of a rodent [@topological-kleinfeld]. The network consists of 417 edges and 389 vertices, of which 320 are bifurcation points. The pressure (a) and cross-section flux (b) were computed using the hydraulic network model. A linear pressure drop was ascribed from inlet to outlets using

 </figcaption>


Acknowledgments
===============

We thank Miroslav Kuchta, Cecile Daversin-Catty and Jørgen Dokken for
their input on the implementation, and Pablo Blinder and David Kleinfeld
for sharing data for the pial vasculature of rodents.

# Acknowledgements

We thank Miroslav Kuchta, Cecile Daversin-Catty and Jørgen Dokken for
their input on the implementation and Pablo Blinder and David Kleinfeld
for sharing data.

## References
