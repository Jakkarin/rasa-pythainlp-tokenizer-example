FROM tensorflow/tensorflow:2.1.1

RUN pip3 install -U pip && pip3 install rasa==1.10.8 pythainlp===2.2.2

CMD ["/bin/bash"]