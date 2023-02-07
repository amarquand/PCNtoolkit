# Installs software and cleans up unnecessary memory: the smaller, the less downtime
FROM continuumio/miniconda3:latest
RUN apt-get update
RUN apt-get install -y libhdf5-dev \
                        pkg-config \
                        gcc \
                        g++ \
                        zip

# Make sure we have the right version of numpy                         
RUN conda install numpy=1.21

# Install directly from GitHub (master branch)
#RUN pip install scikit-learn 
#RUN pip install pcntoolkit==0.26

# This is an alternative method that pulls from the dev branch
RUN wget https://github.com/amarquand/PCNtoolkit/archive/dev.zip 
RUN unzip dev.zip
RUN pip install scikit-learn
RUN cd PCNtoolkit-dev; pip install . ; cd ..

# Add command line links and clean up 
RUN ln -s /opt/conda/lib/python3.10/site-packages/pcntoolkit /opt/ptk
RUN chmod 755 /opt/ptk/normative.py
RUN chmod 755 /opt/ptk/normative_parallel.py
RUN chmod 755 /opt/ptk/trendsurf.py
RUN rm -rf PCNtoolkit-dev dev.zip
