# Implementing network models in FEniCS

This is a repo for experimenting with using `networkx` to make graph meshes and solve equations on these in `FEniCS`. 

## Installation
We use `networkx` combined with 
- the mixed-dimensional branch of `FEniCS` 
- the mixed-dimensional library [`fenics_ii`](https://github.com/MiroK/fenics_ii)

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

[<img alt="alt_text" width="400px" src="pial-network.png" />]([https://www.google.com/](https://github.com/IngeborgGjerde/fenics-networks/pial-network.png?raw=true))
