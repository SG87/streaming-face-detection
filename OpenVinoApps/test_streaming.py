
# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--prototxt", required=True,
#                help="path to Caffe 'deploy' prototxt file")
#ap.add_argument("-m", "--model", required=True,
#                help="path to Caffe pre-trained model")
#ap.add_argument("-c", "--confidence", type=float, default=0.2,
#                help="minimum probability to filter weak detections")
#ap.add_argument("-mW", "--montageW", required=True, type=int,
#                help="montage frame width")
#ap.add_argument("-mH", "--montageH", required=True, type=int,
#                help="montage frame height")
#args = vars(ap.parse_args())


import cv2
import imagezmq


image_hub = imagezmq.ImageHub()
while True:  # show streamed images until Ctrl-C
    rpi_name, image = image_hub.recv_image()
    cv2.imshow(rpi_name, image)  # 1 window for each RPi
    cv2.waitKey(1)
    image_hub.send_reply(b'OK')