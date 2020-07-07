from numpy import expand_dims
from keras.models import load_model
from numpy import load

def get_embeddings(model,face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean,std = face_pixels.mean(),face_pixels.std()
    face_pixels = (face_pixels-mean)/std
    samples = expand_dims(face_pixels,axis=0)
    yhat = model.predict(samples)
    return yhat[0]
    
model_embed = load_model('facenet_keras.h5')
data = load('Face-Extracted.npz')
trainX,trainy = data['arr_0'],data['arr_1']
print('model loaded')
print('faces loaded')

newtrainX = list()
for face in trainX:
    embed = get_embeddings(model_embed,face)
    newtrainX.append(embed)
newtrainX = asarray(newtrainX)
print('face embedded')
print(newtrainX.shape)
print(trainy.shape)

savez_compressed('face-detection-embedding.npz',newtrainX,trainy)    
