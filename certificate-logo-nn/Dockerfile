FROM tensorflow/tensorflow:1.13.2-py3

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /

# Install system dependencies
RUN \
  apt-get -y update && \
  apt-get -y install --no-install-recommends libgl1 python3-opencv && \
  apt-get clean && \
  rm -Rf /var/cache/apt/archives /var/lib/apt/lists/*

# Install Python dependencies (except opencv, which takes a long time, and system-based is good enough)
RUN python3 -m pip install --upgrade --upgrade-strategy only-if-needed `cat /requirements.txt | grep -v opencv`

CMD /bin/bash
