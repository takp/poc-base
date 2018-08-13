FROM ubuntu:18.04

RUN apt-get update \
&& apt-get install -y python3.6 \
&& apt-get install -y python3-pip \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN pwd
RUN ls -l ./
COPY requirements.txt /root
RUN pip3 install -r /root/requirements.txt

ENV POC_ROOT=/usr/local/poc_root
ENV POC_HOME=${POC_ROOT}/poc
ENV POC_USER=poc_user

ENV WEB_PATH=${POC_ROOT}/web
ENV CLI_PATH=${POC_ROOT}/cli
ENV STUB_PATH=${POC_ROOT}/stub

# Add linux user.
RUN useradd -ms /bin/bash -d ${POC_ROOT} -G sudo ${POC_USER}
RUN chown -R ${POC_USER}.${POC_USER} ${POC_ROOT}

COPY main.py ${POC_HOME}/main.py
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

USER ${POC_USER}
WORKDIR ${POC_HOME}
