# Ludwig development environment based on official Ludwig docker image
ARG BASE_IMAGE_TAG=master
# set base using ludwig master branch
FROM ludwigai/ludwig-ray:${BASE_IMAGE_TAG}

# setup for local ludwig install
WORKDIR /app
COPY . /app

# TODO: do we need this?
#ENV HOROVOD_WITHOUT_TENSORFLOW=1
#ENV HOROVOD_WITHOUT_MPI=1
#ENV HOROVOD_WITH_PYTORCH=1
#ENV HOROVOD_WITHOUT_MXNET=1
#ENV MAKEFLAGS="-j 12"

# RUN pip install -e .[test]

###
## install other required tools
###
RUN pip install \
    jupyterlab nbconvert ipywidgets \
    mkdocs'<1.3' \
    mkdocs-material mike Markdown'>=3.0.1' \
    pydot pydotplus graphviz \
    matplotlib IProgress \
    nltk dill \
    pympler pre-commit

## install pycharm related test dependencies
RUN pip install pytest-pycharm pytest-xdist pytest-cov

## install graph vizualization
RUN pip install -U git+https://github.com/szagoruyko/pytorchviz.git@master
#
## install torchtest packages
RUN pip install --upgrade torchtest "torchmetrics>=0.11"
#
## hugging face dataset package
RUN pip install datasets

# set working dir for the project folder
WORKDIR /opt/project
