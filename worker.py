import numpy as np
import time
import cv2
import sys
from openalpr import Alpr
import data
import threading
from threading import Thread
debug = 0
result_image = None
def print_progress(count, total, frame_time, suffix='Processing Frame'):
    time_left = total*frame_time - count*frame_time
    m, s = divmod(time_left, 60)
    h, m = divmod(m, 60)
    sys.stdout.write('%s [%s of %s] Approx. %sh %sm %ss left\r' % (suffix, count, total, int(h), int(m), int(s)))
    sys.stdout.flush()

def Init_alpr():
    alpr = Alpr("us", data.config_location, data.runtime_data)
    if not alpr.is_loaded():
        print "Error Loading ALPR"
        sys.exit(1)
    alpr.set_top_n(1)
    alpr.set_default_region("va")
    return alpr

def Init_cv2():
    face_cascade = cv2.CascadeClassifier(data.face_cascade)
    profile_cascade = cv2.CascadeClassifier(data.profile_cascade)
    cap = cv2.VideoCapture(data.input_video)

    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    vout = cv2.VideoWriter()
    frame_size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), \
                  int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    print "Frame size: {0}\nFPS: {1}\nFrame Count: {2}".format(frame_size, fps, length)
    success = vout.open(data.output_video, fourcc, fps, frame_size, True)
    return cap, vout, face_cascade, profile_cascade, length

def BlurFrameFaces(objects, frame, result_image):
    for(x, y, w, h) in objects:
        sub_face = frame[y:y+h, x:x+w]
        sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)
        result_image[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
    return result_image

def BlurFramePlates(results, frame, result_image):
    for plate in results['results']:
        #x,y coordinates of opposite corners of license plate
        x1 = plate['coordinates'][0]['x']
        y1 = plate['coordinates'][0]['y']
        x3 = plate['coordinates'][2]['x']
        y3 = plate['coordinates'][2]['y']
        if x1 is not None and y1 is not None and x3 is not None and y3 is not None:
            sub_plate = frame[y1:y1+(y3-y1), x1:x1+(x3-x1)]
            sub_plate = cv2.GaussianBlur(sub_plate, (23, 23), 30)
            result_image[y1:y1+sub_plate.shape[0], x1:x1+sub_plate.shape[1]] = sub_plate
    return result_image

def Face_t(frame, face_cascade, result_image):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(75, 75))
    with threading.RLock():
        result_image = BlurFrameFaces(faces, frame, result_image)

def Plate_t(frame, alpr, result_image):
    #get data about license plates in the frame (specifically the coordiantes)
    ret, enc = cv2.imencode("*.bmp", frame)
    plate_results = alpr.recognize_array(bytes(bytearray(enc)))
    with threading.RLock():
        result_image = BlurFramePlates(plate_results, frame, result_image)

def BlurVideo(cap, vout, face_cascade, profile_cascade, alpr, length):
    i = 0
    print_progress(i, length, 0)
    iter_time = time.time()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            result_image = frame.copy()

            face_thread = Thread(target = Face_t, args=(frame, face_cascade, result_image))
            plate_thread = Thread(target = Plate_t, args=(frame, alpr, result_image))
            face_thread.start()
            plate_thread.start()
            face_thread.join()
            plate_thread.join()
            vout.write(result_image)
            i+=1
            if i is 1:
                frame_time = time.time() - iter_time
            print_progress(i, length, frame_time)
        else:
            break
        if debug is 1:
            cv2.imshow('Video', result_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print "Error..."

def Cleanup(cap, vout, alpr):
    alpr.unload()
    cap.release()
    vout.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if debug is 1:
        print "Debug mode is on"
    alpr = Init_alpr()
    cap, vout, face_cascade, profile_cascade, length = Init_cv2()
    process_thread = Thread(target = BlurVideo, args=(cap, vout, face_cascade, profile_cascade, alpr, length, ))
    destroy_thread = Thread(target = Cleanup, args=(cap, vout, alpr, ))
    process_thread.start()
    process_thread.join()
    destroy_thread.start()
    destroy_thread.join()
