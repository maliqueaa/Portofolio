import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import os

def extractColorhist(image):
    hist = cv.calcHist([image], [0,1,2], None,[8,8,8],[0,256,0,256,0,256])
    hist = cv.normalize(hist,hist).flatten()
    return hist

ripedClass = []
unripedClass = []
testData = []

dataset_dir = 'Images/testData/'
gambar_files = os.listdir(dataset_dir)

for file in gambar_files:
    imgPath = os.path.join(dataset_dir, file)
    img = cv.imread(imgPath)
    
    if img is None:
        print("Gagal membaca gambar:", file)
        continue
    
    testData.append(img)

dataset_dir = 'Images/RIPED/'
gambar_files = os.listdir(dataset_dir)

for file in gambar_files:
    imgPath = os.path.join(dataset_dir, file)
    img = cv.imread(imgPath)
    
    if img is None:
        print("Gagal membaca gambar:", file)
        continue
    
    ripedClass.append(img)
    
dataset_dir = 'Images/UNRIPED/'
gambar_files = os.listdir(dataset_dir)

for file in gambar_files:
    imgPath = os.path.join(dataset_dir, file)
    img = cv.imread(imgPath)
    
    if img is None:
        print("Gagal membaca gambar:", file)
        continue
    
    unripedClass.append(img)
    
class1Feature = [extractColorhist(img) for img in ripedClass]
class2Feature = [extractColorhist(img) for img in unripedClass]
testFeature = [extractColorhist(img) for img in testData]

trainData = np.vstack((class1Feature,class2Feature))
responses = np.array([1] * len(class1Feature) + [2] * len(class2Feature))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(trainData,responses)

predictKNN = knn.predict(testFeature)

classLabel = ['Matang', 'Mentah']

fig, axs = plt.subplots(5, 2, figsize=(10, 20))
for ax, img, predictKNN in zip(axs.flatten(), testData, predictKNN):
    image = img
    label = classLabel[int(predictKNN) - 1]
    ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax.set_title("Tomat: {}\n".format(label))
    ax.axis('off')
    
plt.tight_layout()
plt.show()

scoresKNN = cross_val_score(knn, trainData, responses, cv=10).mean()*100
print("akurasi KNN: {:.2f}%".format(scoresKNN))
