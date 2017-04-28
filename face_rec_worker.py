import cv2
import data
import face_recognition as fr
import os
class Image:
    def __init__(self, encoding, classification, name):
        self.encoding = encoding
        self.classification = classification
        self.name = name
        self.color = data.yellow if "murder" in self.classification else data.red if "sex" in self.classification else data.blue if "arson" in self.classification else (0, 0, 0)
        self.text_color = data.black if self.color is data.yellow else data.white
class Face:
    def __init__(self, name, color, text_color):
        self.name = name
        self.color = color
        self.text_color = text_color

def Init_cv2():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    vout = cv2.VideoWriter()
    frame_size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), \
                  int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print "Frame size: {0}\nFPS: {1}".format(frame_size, fps)
    success = vout.open(data.output_video, fourcc, fps, frame_size, True)
    return cap, vout

def getImageObjects():
    encoded_images = []
    for subdir, dirs, imgs in os.walk("./photodb"):
        for img in imgs:
            face_img = fr.load_image_file(os.path.join(subdir, img))
            image = Image(fr.face_encodings(face_img)[0], os.path.join(subdir, img), os.path.splitext(img)[0])
            encoded_images.append(image)
    return encoded_images

def drawKey(frame):
    classifications = ["Sexual Offender", "Murderer", "Arsonist"]
    colors = [data.red, data.yellow, data.blue]
    text_colors = [data.white, data.black, data.white]
    for i, classification, color, text_color in zip(range(len(classifications)), classifications, colors, text_colors):
        cv2.rectangle(frame, (5, 10 + (30 * i)), (275, 38 + (30 * i)), color, cv2.cv.CV_FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, classification, (6, 35 + (30 * i)), font, 1.0, text_color, 1)
    return frame

def find_faces(small_frame, encoded_images):
    #find all faces in the frame
    face_locations = fr.face_locations(small_frame)
    face_encodings = fr.face_encodings(small_frame, face_locations)

    #loop through each face in frame
    faces=[]
    for face_encoding in face_encodings:
        match = fr.compare_faces([e.encoding for e in encoded_images], face_encoding)
        name = "Unknown"
        color = data.black
        text_color = data.white
        for i in range(len(match)):
            if match[i]:
                color = encoded_images[i].color
                name = encoded_images[i].name
                text_color = encoded_images[i].text_color
        faces.append(Face(name, color, text_color))
    return faces, face_locations

def drawOnFrame(face_locations, faces, frame):
    for(top, right, bottom, left), face in zip(face_locations, faces):
        #resize frame
        top*=4
        right*=4
        bottom*=4
        left*=4
        if face.name is "Unknown": # block out face if not in database
            cv2.rectangle(frame, (left, top), (right, bottom), data.black, cv2.cv.CV_FILLED)
        else: # Draw a box around the face otherwise with name and color classification
            cv2.rectangle(frame, (left, top), (right, bottom), face.color, 3)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), face.color, cv2.cv.CV_FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, face.name, (left + 6, bottom - 6), font, 1.0, face.text_color, 1)
    return frame

def BlurVideo(cap, vout):
    encoded_images = getImageObjects()
    process_frame = True
    while (cap.isOpened()):
        ret, frame = cap.read()
        #resize the frame by 1/4 for faster recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # draw key at top left
        frame = drawKey(frame)
        # process every other frame for performace improvements
        if process_frame:
            faces, face_locations = find_faces(small_frame, encoded_images)
        process_frame = not process_frame

        # draw square around faces
        frame = drawOnFrame(face_locations, faces, frame)

        #vout.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print "Error..."

def Cleanup(cap, vout):
    cap.release()
    vout.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cap, vout = Init_cv2()
    BlurVideo(cap, vout)
    Cleanup(cap, vout)

# def recognizeFaces(top, right, bottom, left, face_encoding, encoded_images, frame):
#     #see if the face is a match for the known faces
#     match = fr.compare_faces([e.encoding for e in encoded_images], face_encoding)
#     name = "Unknown"
#     color = data.black
#     text_color = data.white
#     for i in range(len(match)):
#         if match[i]:
#             color = encoded_images[i].color
#             name = encoded_images[i].name
#             text_color = encoded_images[i].text_color
#     #resize frame
#     top*=4
#     right*=4
#     bottom*=4
#     left*=4
#     if name is "Unknown": # block out face if not in database
#         cv2.rectangle(frame, (left, top), (right, bottom), data.black, cv2.cv.CV_FILLED)
#     else: # Draw a box around the face otherwise with name and color classification
#         cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
#
#     # Draw a label with a name below the face
#     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.cv.CV_FILLED)
#     font = cv2.FONT_HERSHEY_DUPLEX
#     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, text_color, 1)
#     return frame
