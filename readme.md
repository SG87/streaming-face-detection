# Streaming Face Detection

## Getting started

    git clone https://github.com/SG87/streaming-face-detection.git
    
    cd streaming-face-detetion
    
    docker build . -t streaming-face-detection
    
    
## Running the container
    
    docker run -p 5000:5000 streaming-face-detection
    
## To dos

- Currently the app only supports detection on a video file. Support for an IP-camera and device camera should be added.
- Currently the app only supports detection on CPU. GPU, MYRIAD, ... should be added