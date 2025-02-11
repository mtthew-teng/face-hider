import os
import cv2
import mediapipe as mp
import argparse
import numpy as np
import pyvirtualcam
import time

C922_WIDTH = 1280
C922_HEIGHT = 720
C922_FPS = 60

def process_img(img, face_detection):
    H, W, _ = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            w = max(1, min(w, W - x1))
            h = max(1, min(h, H - y1))

            # print(f"Face detected at: x1={x1}, y1={y1}, w={w}, h={h}")
            
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (50, 50))

    return img

args = argparse.ArgumentParser()
args.add_argument("--mode", default='virtualcam')
args.add_argument("--filePath", default=None)
args = args.parse_args()

output_dir =  './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.1) as face_detection:
    if args.mode in ["image"]:
        img = cv2.imread(args.filePath)
        

        img = process_img(img, face_detection)

        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

    elif args.mode in ["video"]:
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver)  < 3 :
            fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else :
            fps = cap.get(cv2.CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'), 
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       fps,
                                       (frame.shape[1], frame.shape[0]))

        while ret:
            frame = process_img(frame, face_detection)
            
            output_video.write(frame)

            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode in ["webcam"]:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, C922_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C922_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, C922_FPS)  # May not work on all cameras

        prev = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            time_elapsed = time.time() - prev
            if time_elapsed >= (1.0 / C922_FPS):
                prev = time.time()  # Reset time

                frame = process_img(frame, face_detection)
                cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    elif args.mode in ["virtualcam"]:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, C922_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C922_HEIGHT)

        fmt = pyvirtualcam.PixelFormat.BGR
        with pyvirtualcam.Camera(width=C922_WIDTH, height=C922_HEIGHT, fps=C922_FPS, fmt=fmt) as cam:
            # print(f"Using virtual camera: {cam.device}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                frame = process_img(frame, face_detection)
                cam.send(frame)
                cam.sleep_until_next_frame()

        cap.release()
        cv2.destroyAllWindows()
