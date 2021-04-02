from gcr.io/kaggle-gpu-images/python

ARG user
ARG uid
ARG group
ARG gid

RUN apt-get update \
  && apt-get install -y --no-install-recommends sudo \
  && apt-get clean

RUN pip install -U pip \
  && pip install -U pysen-ls pysen black isort mypy flake8 autopep8 

RUN groupadd -g $gid $group
RUN useradd -m -u $uid -g $group $user
RUN echo "$user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $user
RUN python -m venv --system-site-package ~/.venv \
    && echo "source ~/.venv/bin/activate" >> ~/.bashrc \
    && echo "set -o vi" >> ~/.bashrc
