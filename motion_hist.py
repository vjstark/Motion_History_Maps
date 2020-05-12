import cv2
import numpy as np
import os
from extract import findAOIs

username = "admin"
password2 = "Iconsense1234"
ip = '192.168.1.3'
port = "554"
port2 = "7159"
url = "rtsp://{}:{}@{}:{}/cam/realmonitor?channel=1&subtype=1".format(username, password2, ip, port)
print(url)

path = "ROI/frames"
path3 = "ROI/output"


cap = cv2.VideoCapture(url)
MHI_DURATION = 4 * 28
DEFAULT_THRESHOLD = 32


def slice_image_stack(stack, rect):
    sliced_stack = []
    for frame in stack:
        tmp = frame[rect[0], rect[1]]
        sliced_stack.append(tmp)

    return sliced_stack

def stream_video():

    ret, frame = cap.read()
    print(frame.shape)
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    timestamp = 0
    c = 0;
    recent_frame_stack = []


    while True:

        c = c + 1
        ret, frame = cap.read()
        if not ret:
            break
        recent_frame_stack.append(frame)

        frame_diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
        timestamp += 1

        # update motion history
        cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)

        # normalize motion''[=]\]\[] history
        mh = np.array((np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)))

        #cv2.imshow('motempl', mh)

        if timestamp % MHI_DURATION == 0:
            fr = "frame" + str(c)
            cv2.imwrite(os.path.join(path, fr + ".jpg"), mh)


            acceptedRects = findAOIs(mh)

            if len(acceptedRects) > 0:
                sequence_folder = 'sequence' + str(timestamp)
                seq_path = os.path.join(path3, sequence_folder)
                os.makedirs(seq_path)

            for i in range(len(acceptedRects)):
                #roi_name = "ROI{}".format(i)
                roi_name = f'ROI{i}'
                roi_path = os.path.join(seq_path, roi_name)
                os.makedirs(os.path.join(roi_path))

                rect = acceptedRects[i]
                cropped_frames = slice_image_stack(recent_frame_stack, rect)
                cn = 0
                for cropped_frame in cropped_frames:
                    cn += 1
                    tfr = fr + str(i) + str(cn) + ".jpg"
                    cv2.imwrite(os.path.join(roi_path, tfr), cropped_frame)

            recent_frame_stack = []

        prev_frame = frame.copy()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    # check_fps()
    stream_video()
