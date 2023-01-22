[![Pytest](https://github.com/IngeborgGjerde/graphnics/actions/workflows/main.yml/badge.svg)](https://github.com/IngeborgGjerde/graphnics/actions/workflows/main.yml)
[![Docker image](https://github.com/IngeborgGjerde/graphnics/actions/workflows/docker-image.yml/badge.svg)](https://github.com/IngeborgGjerde/graphnics/actions/workflows/docker-image.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/IngeborgGjerde/graphnics/HEAD)

# Implementing network models in FEniCS

The `graphnics` library solves network models using the finite element method. This is facilitated via the `FenicsGraph` class which is an extension of the `DiGraph` class in `networkx`. 
 

[<img alt="alt_text" width="300px" src="pial-network.png" />]([https://www.google.com/](https://github.com/IngeborgGjerde/fenics-networks/pial-network.png?raw=true))

## Demos

- [Introduction to `Graphnics`](https://github.com/IngeborgGjerde/graphnics/blob/main/demo/Graphnics%20demo.ipynb).
- [Pulsatile flow in rat tumour vasculature](https://github.com/IngeborgGjerde/graphnics/blob/main/demo/Pulsatile%20flow%20on%20a%20real%20vascular%20network.ipynb)
- [Vasomotion in arterial trees](https://github.com/IngeborgGjerde/graphnics/blob/main/demo/Vasomotion%20in%20arterial%20trees.ipynb)
- [Solving coupled 3d-1d network models](https://github.com/IngeborgGjerde/graphnics/blob/main/demo/Coupled%201d-3d.ipynb) using `fenics_ii`

## Runtimes
The library `fenics_ii` has been used for the implementation as it provides rapid and robust assembly of mixed-dimensional problems. The runtimes for the most computationally expensive model is shown here: 
- [Profiling](https://github.com/IngeborgGjerde/graphnics/blob/main/demo/Tree%20profiling.ipynb)

## Installation
We use `networkx` combined with the mixed-dimensional library [`fenics_ii`](https://github.com/MiroK/fenics_ii) created by Miroslav Kuchta. 

The environment is provided as a docker container. The container can be built and run locally by executing

```shell
git clone https://github.com/IngeborgGjerde/graphnics/
cd graphnics/docker 
docker build --no-cache -t graphnics . # build docker image
cd ..

# make container
docker run --name graphnics-container -v "$(pwd):/home/fenics/shared" -d -p 127.0.0.1:8888:8888 graphnics 'jupyter-notebook --ip=0.0.0.0'
```

You can then enter the container by running 
```shell
docker exec -it graphnics-container /bin/bash
```

To connect it to a jupyter notebook, run
```shell
docker logs graphnics-container
```
and enter the html-links it provides in your browser.

## Citation

This code is currently being prepared for submission to JOSS, titled:

*Graphnics: Combining networkx and FEniCS to solve network models* by Ingeborg Gjerde

The paper draft can be found [here.](https://github.com/IngeborgGjerde/graphnics/blob/main/paper/joss.md)

You can cite the repo using arxiv preprint
```
@article{graphnics2022gjerde,
       author = {{Gjerde}, Ingeborg G.},
        title = "{Graphnics: Combining FEniCS and NetworkX to simulate flow in complex networks}",
      journal = {arXiv e-prints},
         year = 2022,
        month = dec,
archivePrefix = {arXiv},
       eprint = {2212.02916}}
```

