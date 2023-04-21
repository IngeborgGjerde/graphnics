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

## Dependencies and installation

The core functionality of `graphnics` is provided by the following libraries:
- [FEniCS](https://fenicsproject.org/) version 2019.1.0 or later
- [networkx](https://networkx.org/)  version 2.5 or later

Some of the network flow models require mixed-dimensional assembly, which is provided by the following libraries:
- [`fenics_ii`](https://github.com/MiroK/fenics_ii)
- ['cbc.block'](https://bitbucket.org/fenics-project/cbc.block/src/master/)

### Installation
Provided that these dependencies are installed, `graphnics` can be pip-installed via
```shell
git clone https://github.com/IngeborgGjerde/graphnics/ && cd graphnics
python3 -m pip install .
```

Should you have trouble installing these dependencies, we recommend you use the below docker image.

## Docker image
The full environment for `graphnics`, along with demos, is provided as a docker container. The container can be built and run locally by executing

```shell
git clone https://github.com/IngeborgGjerde/graphnics/
cd graphnics/docker 
docker build --no-cache -t graphnics . # build docker image
cd ..

# make container
docker run --name graphnics-container -v "$(pwd):/home/fenics/shared" -d -p 127.0.0.1:8888:8888 graphnics 'jupyter-notebook --ip=0.0.0.0'
```
The directory you run the above command from will then be shared with the docker container.

In order to run the jupyter notebook demos, execute 
```shell
docker logs graphnics-container
```
in the terminal. This will print a link to the jupyter notebook. Copy the link and paste it into your browser.


To run scripts natively in the container, you can enter the container by running
```shell
docker exec -it graphnics-container /bin/bash
```
and access the shared directory by running
```shell
cd shared
```


## Citation

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

