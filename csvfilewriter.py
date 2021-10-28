import csv
import cv2
import os
import pickle
import numpy as np

#categories to choose from
categories = ['buildings', 'street', 'glacier','forest', 'mountain', 'sea']
#file name of the csv file to write to 
filename = 'classifier.csv'
#dir of photos
dir = '/home/jonathan/cmpe188/SVM-Classifier/cmpe188_test'
#load the model 
pick = open('/home/jonathan/cmpe188/SVM-Classifier/model.sav', "rb")
model = pickle.load(pick)
pick.close()

#write the header row
fields=['ID', "Category"]
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

print("Starting loop...")
#loop through all pictures and guess their category and write it to 
for img in os.listdir(dir):
    imagePath= os.path.join(dir, img)
    objectImage = cv2.imread(imagePath, 0)
    objectImage = cv2.resize(objectImage, (150,150))
    finalImage = np.array(objectImage).flatten()
    prediction  = model.predict([finalImage])
    row = [img.replace(".jpg", ""), categories[prediction[0]]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)

print("Done")


