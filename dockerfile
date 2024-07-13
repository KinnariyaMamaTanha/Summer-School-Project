ARG PYTORCH_VERSION="2.3.1"
ARG CUDA_VERSION="12.1"
ARG CUDNN_VERSION="8"
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime
COPY *.py /workspace/
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ --upgrade pip \
    && pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/ \
    && pip install debugpy soundata matplotlib pandas scikit-learn \
    && python /workspace/DataDownloader.py
