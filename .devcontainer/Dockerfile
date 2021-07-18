FROM ubuntu:18.04

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive &&\
    apt-get -y install --no-install-recommends \
    build-essential \
    python3-dev &&\
    ln -s /usr/bin/python3.6 /usr/bin/python

ENV CPATH=/usr/include/python3.6
