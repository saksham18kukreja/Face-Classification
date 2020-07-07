from numpy import load
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

data = load('face-detection-embedding.npz')
trainX,trainy = data['arr_0'],data['arr_1']

in_encoder = Normalizer()
trainX = in_encoder.fit_transform(trainX)

label = LabelEncoder()
trainy = label.fit_transform(trainy)

model_classify_face = SVC(kernel='linear',probability=True)
model_classify_face.fit(trainX,trainy)

pred_train = model_classify_face.predict(trainX)
score_train = accuracy_score(trainy,pred_train)
print(score_train)

