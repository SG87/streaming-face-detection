FROM ubuntu:16.04

ENV openvino_version=p_2018.5.455
RUN apt-get update && apt-get -y upgrade && apt-get autoremove

#Install needed dependences
RUN apt-get install -y --no-install-recommends \
        build-essential \
        cpio \
        curl \
        git \
        lsb-release \
        pciutils \
        python3.5 \
        python3.5-dev \
        python3-pip \
        python3-setuptools \
        sudo \
        wget

RUN mkdir /openvino \
 && cd /openvino \
 && wget http://registrationcenter-download.intel.com/akdlm/irc_nas/15078/l_openvino_toolkit_${openvino_version}.tgz \
 && tar -xf l_openvino_toolkit* \
 && mv l_openvino_toolkit_${openvino_version}/* .


#ADD l_openvino_toolkit* /openvino/

ARG INSTALL_DIR=/opt/intel/computer_vision_sdk


# installing OpenVINO dependencies
RUN cd /openvino/ && \
    ./install_cv_sdk_dependencies.sh

RUN git clone https://github.com/jeffbass/imagezmq.git

RUN pip3 install numpy opencv-python

# installing OpenVINO itself
RUN cd /openvino/ && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh --silent silent.cfg

# Model Optimizer
RUN cd $INSTALL_DIR/deployment_tools/model_optimizer/install_prerequisites && \
    ./install_prerequisites.sh

# clean up
RUN apt autoremove -y && \
    rm -rf /openvino /var/lib/apt/lists/*

RUN /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh"

RUN echo "source $INSTALL_DIR/bin/setupvars.sh" >> /root/.bashrc

RUN pip3 install flask

RUN mkdir /OpenVinoApps

COPY ./OpenVinoApps/* /OpenVinoApps/

EXPOSE 5000

CMD ["python3", "/OpenVinoApps/face_detection_flask.py"]
