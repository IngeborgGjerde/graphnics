# Builds a Docker image containing the mixed-dimensional features of FEniCS
# found in
#   - fenics_mixed_dimensional 
#   - fenics_ii 

FROM ceciledc/fenics_mixed_dimensional:21-06-22

USER root

RUN apt-get update && \
    apt-get -y install python3-h5py && \
    apt install -y libgl1-mesa-glx && \
    pip install --upgrade pip && \
    pip install networkx && \
    pip install pandas && \
    pip install vtk && \
    pip install tqdm && \
    pip install seaborn && \
    pip install --no-cache-dir notebook jupyterlab jupyterhub && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*- && \
    pip install -U setuptools

# remove the fenics-mixed-dim folder to avoid confusion
RUN rm -r demo/

# Get fenics_ii
RUN git clone https://github.com/MiroK/fenics_ii.git && \
    cd fenics_ii && \
    python3 -m pip install -e . && \
    cd ..

# cbc.block
RUN git clone https://bitbucket.org/fenics-apps/cbc.block && \
    cd cbc.block && \
    python3 -m pip install . && \
    cd ..
     
# fix decorator error by reinstalling scipy
RUN pip uninstall -y scipy && pip install scipy

