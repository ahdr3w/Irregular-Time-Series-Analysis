FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN apt-get update && apt-get install apt-file -y && apt-file update
RUN apt-get install -y git build-essential curl wget software-properties-common zip unzip vim

RUN pip install jupyterlab jupyterlab-optuna

EXPOSE 6006
EXPOSE 8888

COPY requirements.txt /home/requirements.txt
WORKDIR /home/

RUN pip install -r requirements.txt

RUN groupadd -r -g 1001 awesome && \
    useradd -r -u 1000 -g awesome -m -d /home/kid -s /bin/bash kid && \
    mkdir -p /home/kid/.local/share/jupyter && \
    chown -R kid:awesome /home/kid/.local
USER root
# CMD ["jupyter lab --ip 0.0.0.0 --no-browser --allow-root & tensorboard --logdir ./kid/work/Reguformer/notebooks/ --host 0.0.0.0 --port 6006"]
