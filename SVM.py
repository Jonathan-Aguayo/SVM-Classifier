from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('WebAgg')
import os
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


print("starting...")
dir = '/home/jonathan/cmpe188/SVM-Classifier/cmpe188_train/cmpe188_train' 

categories = ['buildings', 'street', 'glacier','forest', 'mountain', 'sea']
param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000] ,'gamma': [0.0001, 0.001, 0.01, 0.1, 1], 'kernel': ['rbf', 'linear', 'sigmoid', 'poly']}
data = []
svc = svm.SVC()
model = GridSearchCV(svc, param_grid)

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        imagePath= os.path.join(path, img)
        objectImage = cv2.imread(imagePath, 0)
        objectImage = cv2.resize(objectImage, (32,32))
        finalImage = np.array(objectImage).flatten()
        data.append([finalImage, label])

features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

       
   
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size= 0.15)
scalar = StandardScaler()
scalar.fit_transform(xtrain)
xtrain = scalar.transform(xtrain)
xtest = scalar.transform(xtest)

pca = PCA(.60)
pca.fit(xtrain)
print(pca.n_components_)


print('loading model...')
#model.fit(xtrain, ytrain)

#Saving the model to our directory so we dont have to train it every time this program runs
# As os 10/21/21 8:30 PM. The model was trained using test size of 0.98, so it may not be very accurate
#pick = open('nonPCATestSize0.15.wav', 'wb')
#pickle.dump(model, pick)
#pick.close()

# opening model again later
pick = open('/home/jonathan/cmpe188/SVM-Classifier/model.sav', 'rb')
model = pickle.load(pick)
pick.close()

prediction = model.predict(xtest)
Accuracy = model.score (xtest, ytest)
print('Prediction is: ' + str(categories[prediction[0]]))
print('Accuracy: ' + str(Accuracy))
plt.imshow(xtest[0].reshape(32,32), cmap='gray')
plt.show()




