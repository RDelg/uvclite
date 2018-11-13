import uvclite
from time import sleep
import cv2
import numpy as np

def raw_to_8bit(data):
   cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
   np.right_shift(data, 8, data)
   return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

uvc_frame_format = uvclite.libuvc.uvc_frame_format
capturing = True

with uvclite.UVCContext() as context:
    device = context.find_device(0x1e4e, 0x0100) # finds PureThermal cam
    device.open()
    print("Device open")
    device.set_stream_format(uvc_frame_format.UVC_FRAME_FORMAT_Y16, 160, 120, 9)
    device.start_streaming()
    print("Streaming...")
    #sleep(1) # get_frame() raises a timeout error the first time it is called if it run immediately afet start streaming.
    while capturing:
        try:
            frame = device.get_frame()
            if(frame.size != (2*160*120)): # sometimes the webcam doesn't return a fullframe
                continue
            data = np.frombuffer(frame.data, dtype=np.uint16).reshape(120, 160)
            data = cv2.resize(data, (640, 480))
            img = raw_to_8bit(data)
            cv2.imshow('PureThermal', img)
            if cv2.waitKey(25) & 0xFF  == ord('q'):
               break
        except uvclite.UVCError as e:
            print(e)
            if(e[1] == 110):
                continue
            else:
                break

    device.stop_streaming()
    device.close()
