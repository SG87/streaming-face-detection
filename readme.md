# Streaming Face Detection

Application that detects faces on on a video/camera stream. Next to detection the app is able to blur faces and to detect sentiment.


## Getting started

    git clone https://github.com/SG87/streaming-face-detection.git
    
    cd streaming-face-detetion
    
    docker build . -t streaming-face-detection
    
    
## Running the container
    
    docker run -p 5000:5000 streaming-face-detection
    
## View the app

Browse to: *IP*:5000/video_feed

Standard the endpoint will show the initial video with face **recognition**.
To add blurring or sentiment detection: add the blur parameter as follows:
- For a flurring of the faces: **blur=blurred**
- For a black rectangle over the faces: **blur=black**
- For the face of Donald Trump over the faces: **blur=trump**
- For sentiment detection: **blur=sentiment**

Example: *IP*:5000/video_feed?blur=blurred

## To dos

- Currently the app only supports detection on a video file. Support for an IP-camera and device camera should be added.
- Currently the app only supports detection on CPU. GPU, MYRIAD, ... should be added