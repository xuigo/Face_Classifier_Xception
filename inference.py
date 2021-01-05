import cv2
import os
import tqdm
from keras.models import load_model
import numpy as np

from utils import preprocess_input,load_image

class inference:
    def __init__(self,model_path,threshold=0.5):
        self.model_path=model_path
        self.threshold=threshold
        self.restore()
        self.size=self.classifier.input_shape[1:3]

    def restore(self):        
        self.classifier=load_model(self.model_path,compile=False)

    def run(self,src):
        filepath,filename=os.path.split(src)
        gray_image = load_image(src, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')
        gray_face = cv2.resize(gray_image, self.size)
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        predict=self.classifier.predict(gray_face)
        predict=np.squeeze(predict)
        predict=np.around(predict,decimals=2)
        predict=[0 if p < self.threshold else 1 for p in predict]
        return predict
    
if __name__=='__main__':
    model_path='./weights/celeba_mini_XCEPTION.42-0.89.hdf5'
    image_dir='./test'
    xception=inference(model_path,threshold=0.5)
    for file in tqdm.tqdm(os.listdir(image_dir)):
        image_path=os.path.join(image_dir,file)
        pred=xception.run(image_path)
        #print(pred)

    
