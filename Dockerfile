# Builds a Docker image containing the mixed-dimensional features of FEniCS
# found in
#   - fenics_mixed_dimensional 
#   - fenics_ii 

FROM ceciledc/fenics_mixed_dimensional:21-06-22

USER root

RUN apt-get -qq update && \
    apt-get clean && \
    apt-get -y install python3-h5py && \
    pip install --upgrade pip && \
    pip install networkx && \
    pip install pandas && \
    pip install tqdm && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*-

# Get fenics_ii
RUN git clone https://github.com/MiroK/fenics_ii.git && \
    cd fenics_ii && \
    python setup.py install --user && \
    cd ..

# cbc.block
RUN git clone https://bitbucket.org/fenics-apps/cbc.block && \
    cd cbc.block && \
    python setup.py install --user && \
    cd ..

# ulfy
RUN git clone https://github.com/MiroK/ulfy.git && \
    cd ulfy && \
    git checkout -b 42dfe51c821acffbccc0df26d7b9549a5cb949eb && \
    python setup.py install --user && \
    cd ..

# fix decorator error by reinstalling scipy
RUN pip uninstall -y scipy && pip install scipy
