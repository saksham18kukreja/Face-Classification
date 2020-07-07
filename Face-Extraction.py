from PIL import Image
from os import listdir
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from numpy import savez_compressed

def extract_face(filename,required_size=(160,160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixel = asarray(image)
    model = MTCNN()
    result = model.detect_faces(pixel)
    x1,y1,w1,h1 = result[0]['box']
    x1,y1 = abs(x1),abs(y1)
    x2,y2 = x1+w1,y1+h1
    face = pixel[y1:y2,x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def load_faces(directory):
    faces = list()
    for image in listdir(directory):
        path = directory + image
        face = extract_face(path)
        faces.append(face)  
    return faces
   
   
def load_dataset(directory):
    X,y = list(),list()
    for subdirectory in listdir(directory):
        path = directory + subdirectory + '/'
        face_array = load_faces(path)
        label = [subdirectory for _ in range(len(face_array))]
        print('the face has been extracted {0}'.format(subdirectory))
        X.extend(face_array)
        y.extend(label)  
    return asarray(X),asarray(y)   
    
trainX,trainy = load_dataset('faces/')
print(trainX.shape,trainy.shape)  

savez_compressed('extracted_faces.npz',trainX,trainy)
