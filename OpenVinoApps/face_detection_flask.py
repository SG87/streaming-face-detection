from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin
from flask import Flask, request, Response
from models import EmotionClassifier
from utils import load_emojis
from utils import emoji_overlay, Location


app = Flask(__name__)

def detect(input="./video.mp4", blur=None):
    trump = cv2.imread("./trump.png", cv2.IMREAD_UNCHANGED)

    emojis = load_emojis("./emojis/")
    emotion_classifier = EmotionClassifier(model_xml="./emotion_recognition/FP32/em.xml",
                                           model_bin="./emotion_recognition/FP32/em.bin",
                                           device="CPU",
                                           cpu_extension="/opt/intel/computer_vision_sdk_2018.5.455/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so",
                                           emotion_label_list=["neutral", "happy", "sad", "surprise", "anger"])

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    model_xml = "./face_detection/FP32/fd.xml"
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device="CPU", plugin_dirs=None)
    plugin.add_cpu_extension(
        "/opt/intel/computer_vision_sdk_2018.5.455/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so")
    # Read IR
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    supported_layers = plugin.get_supported_layers(net)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the plugin for specified device {}:\n {}".format(plugin.device,
                                                                                                          ', '.join(
                                                                                                              not_supported_layers)))
        log.error(
            "Please try to specify cpu extensions library path in demo's command line parameters using -l or --cpu_extension command line argument")
        sys.exit(1)
    assert len(net.inputs.keys()) == 1, "Demo supports only single input topologies"
    assert len(net.outputs) == 1, "Demo supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape

    if input == 'cam':
        input_stream = 0
    else:
        input_stream = input
        assert os.path.isfile(input), "Specified input file doesn't exist"

    labels_map = None

    cap = cv2.VideoCapture(input_stream)

    cur_request_id = 0
    next_request_id = 1

    log.info("Starting inference in async mode...")
    log.info("To switch between sync and async modes press Tab button")
    log.info("To stop the demo execution press Esc button")
    is_async_mode = True
    render_time = 0
    ret, frame = cap.read()
    while cap.isOpened():
        process_start = time.time()

        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()
        if not ret:
            break
        initial_w = cap.get(3)
        initial_h = cap.get(4)
        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = time.time()
        if is_async_mode:
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
        else:
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > 0.5:
                    xmin = int(obj[3] * initial_w)
                    ymin = int(obj[4] * initial_h)
                    xmax = int(obj[5] * initial_w)
                    ymax = int(obj[6] * initial_h)
                    class_id = int(obj[1])
                    # Draw box and label\class_id
                    color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))

                    if blur == "black":
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, -1)
                    elif blur == "blurred":
                        face = frame[ymin:ymax, xmin:xmax]
                        face = cv2.GaussianBlur(face, (23, 23), 30)
                        frame[ymin:ymax, xmin:xmax] = face
                    elif blur == "emotion":
                        face = frame[ymin:ymax, xmin:xmax]
                        emotion = emotion_classifier.predict(face)
                        #print (emotion)
                        face_location = Location(xmin,ymin,xmax-xmin,ymax-ymin)
                        emoji_overlay(emojis[emotion], frame, face_location)
                        #draw_bounding_box(face,img)]
                    elif blur == "trump":
                        face = frame[ymin:ymax, xmin:xmax]
                        #print (emotion)
                        face_location = Location(xmin,ymin,xmax-xmin,ymax-ymin)
                        emoji_overlay(trump, frame, face_location)

                    else:
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    det_label = labels_map[class_id] if labels_map else str(class_id)
                    cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)


            # Draw performance stats
            inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
                "Inference time: {:.3f} ms".format(det_time * 1000)
            render_time_message = "OpenCV processing time: {:.3f} ms".format(process_time * 1000)
            async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
                "Async mode is off. Processing request {}".format(cur_request_id)

            # cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, async_mode_message, (10, int(initial_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)

        process_end = time.time()
        process_time = process_end - process_start

        if is_async_mode:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')

            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame



    cv2.destroyAllWindows()
    del exec_net
    del plugin

@app.route('/video_feed')
def video_feed():
    try:
        blur = request.args.get('blur')
    except:
        blur = None
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(detect(blur=blur), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=6000)
