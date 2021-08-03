FROM ubuntu:18.04

# set up conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update --fix-missing \
    && apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion libgdal-dev libyaml-dev \
    # [Optional] Add sudo support for the non-root user
    && apt-get install -y sudo \
    && apt-get clean
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b && \
    rm ~/miniconda.sh && \
    conda clean -tipsy && \
    sudo ln -s ~/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find ~/miniconda3/ -follow -type f -name '*.a' -delete && \
    find ~/miniconda3/ -follow -type f -name '*.js.map' -delete && \
    conda clean -afy
RUN conda install ipython

RUN pip install pytest

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive &&\
#     apt-get -y install --no-install-recommends \
#     build-essential \
#     python3-dev &&\
#     gdb &&\
#     ln -s /usr/bin/python3.6 /usr/bin/python
