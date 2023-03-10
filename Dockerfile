FROM tensorflow/tensorflow:1.12.0-gpu-py3
MAINTAINER Ankur Debnath, Cheif Architect

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=8.0" \
    LANG=C.UTF-8

EXPOSE 8000
RUN apt-get update && apt-get install -y apache2 apache2-dev vim \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/api/lists/*

RUN mkdir /core_lm
WORKDIR /core_lm
ADD . /core_lm

RUN pip3 install -r requirements.txt

RUN python3 model_utils.py 'threapymodel/top_model'

RUN /opt/conda/bin/mod_wsgi-express install-module
RUN mod_wsgi-express setup-server core_lm.wsgi --port=8000 \
    --user www-data --group www-data \
    --server-root=/etc/mod_wsgi-express-80
    
RUN python3 concatconfigfile.py
CMD /etc/mod_wsgi-express-80/apachectl start -D FOREGROUND