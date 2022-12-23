from Face-Embedding import face_embedding
from Face-Extraction import face_extraction
from Final-Model import final_model
from numpy import load
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer



if __name__ == "__main__":

	extraction = face_extraction()
	embedd = face_embedding()
	model = final_model()


	extraction.extract_faces_from_training()
	embedd.save_embeddings()


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
	print("training score is : ", score_train)

	final_model.run_model(model_classify_face)