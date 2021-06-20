import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path= './img/'
onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]


#create arrray for training data and labels
Training_Data,Labels=[],[]

#open traing image in our datapath
#create numpy array for traing data
for i, files in enumerate(onlyfiles):
    image_path= data_path + onlyfiles[i]
    images=cv2.imread(image_path ,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)
    
    
Labels =np.asarray(Labels, dtype=np.int32)

#model =cv2.face_LBPHFaceRecognizer.create()
model=cv2.face.LBPHFaceRecognizer_create()
Model.train(np.asarray(Training_Data),np.asarray(Labels))

print("Model trained successfully")
