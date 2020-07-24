from cv2 import cv2
from imutils.video import FileVideoStream
import dlib
import os
import glob
import numpy
import math


imageCounter = 0
scaleFactor = 1.2
minNeighbors = 3
minSize = (50, 50)
cascadePath = "../haarcascade_frontalface_alt.xml"
predictorPath = "../shape_predictor_5_face_landmarks.dat"
faceRecognitionModelPath = "../dlib_face_recognition_resnet_model_v1.dat"
savedImagesPath = os.getcwd() + "/saved_images"
detectedFacesPath = savedImagesPath + "/detected_faces"


if not os.path.exists("saved_images"):
    os.mkdir("saved_images")

os.chdir(savedImagesPath)

if not os.path.exists("detected_faces"):
    os.mkdir("detected_faces")


def detectFaces(filePath):
    cascadeClassifier = cv2.CascadeClassifier(cascadePath)
    video = FileVideoStream(filePath).start()
    global imageCounter

    while video.more():
        frame = video.read()
        keyPress = cv2.waitKey(1)

        if keyPress & 0xFF == ord('c'):
            print("Stopping the detection...")
            break
        if keyPress & 0xFF == ord('s'):
            print("Saving the image...")
            imageCounter += 1
            cv2.imwrite('image #' + str(imageCounter) + '.jpg', frame)
 
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detectedFaces = cascadeClassifier.detectMultiScale(grayFrame, scaleFactor = scaleFactor, minNeighbors = minNeighbors, minSize = minSize)

        if len(detectedFaces) > 0:
            for (x, y, w, h) in detectedFaces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face detection', frame)
        
    video.stop()


def compareFaces(filePath):
    faceDetector = dlib.get_frontal_face_detector()
    shapePredictor = dlib.shape_predictor(predictorPath)
    faceRecognitionModel = dlib.face_recognition_model_v1(faceRecognitionModelPath)
    faceDescriptorsForComparison = []
    faceMatchThreshold = 0.6
    faceNumber = 0

    if(filePath == ""):
        os.chdir(detectedFacesPath)

        for f in glob.glob(os.path.join(savedImagesPath, "*.jpg")):
            print("File: {}".format(f))
            image = dlib.load_rgb_image(f)
            detectedFaces = faceDetector(image, 1)
            print("Faces detected: {}".format(len(detectedFaces)))

            for face in detectedFaces:
                croppedImage = image[face.top():face.top() + abs(face.top()-face.bottom()), face.left():face.left() + abs(face.left()-face.right())]
                cv2.imwrite('face #' + str(faceNumber+1) + '.png', croppedImage)
                faceNumber += 1
                shape = shapePredictor(image, face)
                faceDescriptor = faceRecognitionModel.compute_face_descriptor(image, shape)
                faceDescriptorsForComparison.append(numpy.asarray(faceDescriptor))

        os.chdir(savedImagesPath)
    else:
        os.chdir(filePath)
        if not os.path.exists("detected_faces"):
            os.mkdir("detected_faces")
        os.chdir(filePath + "/detected_faces")

        for f in glob.glob(os.path.join(filePath, "*.jpg")):
            print("File: {}".format(f))
            image = dlib.load_rgb_image(f)
            detectedFaces = faceDetector(image, 1)
            print("Faces detected: {}".format(len(detectedFaces)))

            for face in detectedFaces:
                croppedImage = image[face.top():face.top() + abs(face.top()-face.bottom()), face.left():face.left() + abs(face.left()-face.right())]
                cv2.imwrite('face #' + str(faceNumber+1) + '.png', croppedImage)
                faceNumber += 1
                shape = shapePredictor(image, face)
                faceDescriptor = faceRecognitionModel.compute_face_descriptor(image, shape)
                faceDescriptorsForComparison.append(numpy.asarray(faceDescriptor))
        
        os.chdir(savedImagesPath)

    for i in range(0, len(faceDescriptorsForComparison)):
        euclideanDistance = numpy.linalg.norm(faceDescriptorsForComparison[0] - faceDescriptorsForComparison[i])

        if euclideanDistance > faceMatchThreshold:
            if(euclideanDistance >= 1.0):
                print("The distance between face #1 and face #{} is {} and they are {}% similar.".format(i+1, euclideanDistance, 0))
            else:
                rangeValue = (1.0 - faceMatchThreshold)
                linearValue = (1.0 - euclideanDistance) / (rangeValue * 2.0)
                similarityPercentage = linearValue * 100
                print("The distance between face #1 and face #{} is {} and they are {}% similar.".format(i+1, euclideanDistance, round(similarityPercentage, 2)))
        else:
            rangeValue = faceMatchThreshold
            linearValue = 1.0 - (euclideanDistance / (rangeValue * 2.0))
            linearValue += ((1.0 - linearValue) * math.pow((linearValue - 0.5) * 2, 0.2))
            similarityPercentage = linearValue * 100
            print("The distance between face #1 and face #{} is {} and they are {}% similar.".format(i+1, euclideanDistance, round(similarityPercentage, 2)))


cv2.destroyAllWindows()