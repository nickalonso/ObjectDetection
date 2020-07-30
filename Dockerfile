# Copyright (c) 2019 Kinetica DB Inc.
#
# Kinetica Machine Learning
# Kinetica Machine Learning BlackBox Container SDK
#
# for support, contact Saif Ahmed (support@kinetica.com)
#
FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

# These utilities are only for debugging
# These can safely be removed in PROD settings, if desired
RUN apt-get install -y git htop wget nano
RUN apt-get update && apt-get install -y \
 curl \
 htop \
 wget \
 nano \
 git \
 zip \
 build-essential \
 libcurl3-dev \
 libfreetype6-dev \
 libzmq3-dev \
 pkg-config \
 software-properties-common \
 swig \
 zlib1g-dev 
# libpng12-dev
# libstdc++6 

RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
	apt-get update -y && \
	apt-get upgrade -y && \
	apt-get dist-upgrade -y

RUN mkdir -p /opt/gpudb/kml/bbx
WORKDIR "/opt/gpudb/kml/bbx"

# Install Required Libraries and Dependencies
ADD requirements.txt  ./
RUN pip install -r requirements.txt

# Add Kinetica BlackBox SDK (currently v7.0.5b)
ADD bb_runner.sh ./
ADD sdk ./sdk

ADD bb_module_default.py ./
ADD frozen_inference_graph.pb ./
ADD visualization_utils.py ./
ADD research ./research
ADD frozen_inference_graph.pb ./
ADD pascal_label_map.pbtxt ./
# ADD LC_encoder.sav ./

RUN ["chmod", "+x",  "bb_runner.sh"]
ENTRYPOINT ["/opt/gpudb/kml/bbx/bb_runner.sh"]



# FROM nvidia/cuda:9.0-devel-ubuntu16.04
# FROM python:3.6
#FROM tensorflow/tensorflow:latest-devel-gpu-py3