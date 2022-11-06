[![Pytest](https://github.com/IngeborgGjerde/graphnics/actions/workflows/main.yml/badge.svg)](https://github.com/IngeborgGjerde/graphnics/actions/workflows/main.yml)
[![Docker image](https://github.com/IngeborgGjerde/graphnics/actions/workflows/docker-image.yml/badge.svg)](https://github.com/IngeborgGjerde/graphnics/actions/workflows/docker-image.yml)

# Implementing network models in FEniCS

The `graphnics` library solves network models using the finite element method. This is facilitated via the `FenicsGraph` class which is an extension of the `DiGraph` class in `networkx`. 
 

[<img alt="alt_text" width="300px" src="pial-network.png" />]([https://www.google.com/](https://github.com/IngeborgGjerde/fenics-networks/pial-network.png?raw=true))

## Demos

- [Introduction to `Graphnics`](https://github.com/IngeborgGjerde/graphnics/blob/main/demo/Graphnics%20demo.ipynb).
- [Solving coupled 3d-1d network models](https://github.com/IngeborgGjerde/graphnics/blob/main/demo/Coupled%201d-3d.ipynb) using `fenics_ii`

## Installation
We use `networkx` combined with the mixed-dimensional library [`fenics_ii`](https://github.com/MiroK/fenics_ii) created by Miroslav Kuchta. 

The environment is provided as a docker container. The container can be built and run locally by executing

```shell
git clone https://github.com/IngeborgGjerde/fenics-networks/
cd fenics-networks
docker build --no-cache -t fenics-networks .
docker run --name networkfenics -v "$(pwd):/home/fenics/shared" -d -p 127.0.0.1:8888:8888 fenics-networks 'jupyter-notebook --ip=0.0.0.0'
```

You can then enter the container by running 
```shell
docker exec -it networkfenics /bin/bash
```
To connect it to a jupyter notebook, run
```shell
docker logs networkfenics
```
and enter the html-links it provides in your browser.

## Paper 

This code is currently being prepared for submission to JOSS, titled:

*Graphnics: Combining networkx and FEniCS to solve network models* by Ingeborg Gjerde

The paper draft can be found [here.](https://github.com/IngeborgGjerde/graphnics/blob/main/paper/joss.md)
