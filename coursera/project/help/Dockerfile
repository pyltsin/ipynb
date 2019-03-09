FROM continuumio/anaconda

LABEL maintainer="Vlad"

RUN conda install -c statsmodels statsmodels=0.8.0
RUN conda install seaborn

RUN echo "alias py_kaggle=\"jupyter notebook --no-browser --ip="*" --notebook-dir=/tmp/working\"">>.bashrc 
RUN echo "alias TB=\"'tensorboard --logdir=t_boardlog/ &\"" >> ~/.bashrc 
