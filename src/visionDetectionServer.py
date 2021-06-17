from proto.python.message_detection_pb2 import *
from proto.python.message_ROI_pb2 import *
import cv2
import socket
import numpy as np
import _thread

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from tool.utils import *

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class graphicServer:

    def __init__(self, address, port, engine_path, IN_IMAGE_H, IN_IMAGE_W):

        self.address = (address, port)
        self.mySocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.mySocket.bind(self.address)
        self.detectMsg = message_detection()
        self.ROIMsg = message_ROI()
        self.engine_path = engine_path
        self.image_h = IN_IMAGE_H
        self.image_w = IN_IMAGE_W
        self.ed = sl.Camera()


    def init_zed2(self):
        # Create a InitParameters object and set configuration parameters
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        self.init_params.camera_fps = 30  # Set fps at 30

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

    def get_engine(self):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(self.engine_path))
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    def allocate_buffers(self, engine, batch_size):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:

            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            dims = engine.get_binding_shape(binding)

            # in case batch dimension is -1 (dynamic)
            if dims[0] < 0:
                size *= -1

            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def doInference(self, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def doDetect(self, context, buffers, image_src, image_size, num_classes):
        IN_IMAGE_H, IN_IMAGE_W = image_size

        # Input
        resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        img_in = np.ascontiguousarray(img_in)

        inputs, outputs, bindings, stream = buffers
        inputs[0].host = img_in

        trt_outputs = self.doInference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)


        trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
        trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)

        boxes = post_processing(img_in, 0.4, 0.6, trt_outputs)

        return boxes

    def handelROI(self, image_src):
        min_x = self.image_w
        max_x = 0
        min_y = self.image_h
        max_y = 0
        for point in self.ROIMsg.ROI_points:
            if(min_x > point.x):min_x = point.x
            if(max_x < point.x):max_x = point.x
            if(min_y > point.y):min_y = point.y
            if(max_y < point.y):max_y = point.y
        return image_src[min_y:max_y,min_x,max_x]

    def start(self):
        _thread.start_new_thread(self.run,(1,))

    def run(self,i):
        with self.get_engine() as engine, engine.create_execution_context() as context:
            while True:
                self.mySocket.listen(5)
                clientSocket, addr = self.mySocket.accept()
                len_data = clientSocket.recv(1024)
                data = clientSocket.recv(int(len_data.decode('utf-8')))
                self.ROIMsg.ParseFromString(data)

                buffers = self.allocate_buffers(1)
                context.set_binding_shape(0, (1, 3, self.image_h, self.image_w))
                image = sl.Mat()
                self.runtime_parameters = sl.RuntimeParameters()

                # Grab an image, a RuntimeParameters object must be given to grab()
                if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(image, sl.VIEW.LEFT)
                image_src = image.get_data()

                image_src = self.handelROI(image_src)

                num_classes = 2
                boxes = self.doDetect(context, buffers, image_src, (self.image_h, self.image_w), num_classes)
                namesfile = 'data/names'
                class_names = load_class_names(namesfile)
                self.makeMsg(boxes, class_names)
                try:
                    self.sendMsg(clientSocket)
                except:
                    pass

    def makeMsg(self, boxes, class_names):
        if(len(boxes)==0):
            self.detectMsg.result = "noDefect"
        else:
            for box in boxes[0]:
                if(class_names[box[6]]=="break"):
                    self.detectMsg.result = "break"
                    return
            self.detectMsg.result = "crack"

    def sendMsg(self,clientSocket):
        msgData = self.detectMsg.SerializeToString()
        length = len(msgData)
        strLength = str(length)
        clientSocket.send(strLength.encode('utf-8'))
        clientSocket.send(msgData)