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
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*- && \
    pip install -U setuptools
    
# Get fenics_ii
RUN git clone https://github.com/MiroK/fenics_ii.git && \
    cd fenics_ii && \
    python3 setup.py install --user && \
    cd ..

# cbc.block
RUN git clone https://bitbucket.org/fenics-apps/cbc.block && \
    cd cbc.block && \
    python3 setup.py install --user && \
    cd ..

# fix decorator error by reinstalling scipy
RUN pip uninstall -y scipy && pip install scipy

#  networkgen for creating arterial trees
RUN git clone https://gitlab.com/ValletAlexandra/NetworkGen
    #cd NetworkGen  && \
    ##source setup.rc  && \
    #cd ..
# the source.setup.rc fails in docker build so we do it manually later

RUN python3 -m pip install jupyter