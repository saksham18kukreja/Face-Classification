from cv2 import imread
from cv2 import imshow
from cv2 import CascadeClassifier
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import rectangle
from cv2 import VideoCapture
import cv2
from numpy import asarray,expand_dims
from PIL import Image
from keras.models import load_model

#using my webcam as the primary source of image
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
camera = VideoCapture(0)

#function to extract face from the webcam frame
def extract_face_webcam(frame,bound_box,required_size=(160,160)):
    pixel = asarray(frame)
    x1,y1,w1,h1 = bound_box
    x1,y1 = abs(x1),abs(y1)
    x2,y2 = x1+w1,y1+h1
    face = pixel[y1:y2,x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# Create Embeddings for the detected face 
def embed_face_webcam(model,face):
    face = face.astype('float32')
    mean,std = face.mean(),face.std()
    face = (face-mean)/std
    sample = expand_dims(face,axis=0)
    embed_face = model.predict(sample)
    return embed_face

# Predicting the detected face by using the SVC classifier named model_classify_face
def predict_face_webcam(model,face):
    face = in_encoder.transform(face)
    yhat_class = model.predict(face)
    yhat_class_prob = model.predict_proba(face)
    class_index = yhat_class[0]
    probability = yhat_class_prob[0,class_index]*100
    pred_name = label.inverse_transform([yhat_class])
    return pred_name,probability
    
# Load the CNN model to make Face Embeddings
model_embed = load_model('facenet_keras.h5')


# Use OpenCv to capture frame from the webcam feed and predicting the detected face
while True:
    (_,frame) = camera.read()
    
    bbox = classifier.detectMultiScale(frame,1.1,5,minSize = (30,30))
    
    for box in bbox:
        face = extract_face_webcam(frame,box)
        embed_face = embed_face_webcam(model_embed,face)
        name,probability = predict_face_webcam(model_classify_face,embed_face)
        printname = str(name[0])
        printprob = float(probability)      
        x,y,w,h = box
        x2,y2 = x+w,y+h
        rectangle(frame,(x,y),(x2,y2),(0,255,0),2)
        if printprob >= 90: 
            cv2.putText(frame,'{0}'.format(printname),(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)   
        else:
            cv2.putText(frame,'Face Not-Identified',(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    imshow('face detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('n'):
          break
          
       
camera.release()
cv2.destroyAllWindows()    
