import numpy as np
from random import shuffle
import os
import cv2
import sys

from keras.preprocessing import image

def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)


class DataManager(object):
    """ """
    def __init__(self,dataset_path=None, image_size=(48, 48)):
        self.dataset_path = dataset_path
        self.image_size = image_size

    def get_data(self):
        #ground_truth_data=self._load_data()  #single label
        ground_truth_data=self._load_data()  #multi label
        return ground_truth_data

    def _load_celeba(self):
        attr_txt='./list_attr_celeba.txt'
        with open(attr_txt,'r') as f:
            lines=f.readlines()
        
        files,labels=[],[]
        for line in lines[2:]:
            line_split=line.strip().split(' ')
            filename=line_split[0]
            label=[0 if int(line_) ==-1 else 1 for line_ in line_split[1:] if line_ !='']
            fullfile=os.path.join(self.dataset_path,filename)
            if os.path.isfile(fullfile):
                files.append(fullfile)
                labels.append(label)
        del lines
        assert len(files)==len(labels)
        images_arr=[]
        labels_arr=[]
        y_size, x_size = self.image_size

        for idx,file in enumerate(files):
            sys.stdout.write('Loading data: {} / {} \r'.format(idx,len(files)))
            sys.stdout.flush()
            image_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            images_arr.append(image_array)
            labels_arr.append(labels[idx])
        
        shuffle_idx=[i for i in range(len(images_arr))]
        np.random.shuffle(shuffle_idx)
        images_arr=np.array(images_arr)
        labels_arr=np.array(labels_arr)
        images_arr=images_arr[shuffle_idx]
        labels_arr=labels_arr[shuffle_idx]   
        images_arr = np.expand_dims(images_arr, -1)
        return images_arr, labels_arr

    def _load_data(self):
        if isinstance(self.dataset_path,list):
            self.data_dir=self.dataset_path
        elif isinstance(self.dataset_path,str):
            self.data_dir=[os.path.join(self.dataset_path,folder) for folder in os.listdir(self.dataset_path)]
        else:
            raise ValueError('[!] Incorrect dataPath!') 
        y_size, x_size = self.image_size
        images=[]
        labels=[]
        num_classes=len(self.data_dir)
        for idx,sub_dir in enumerate(self.data_dir):
            label=np.zeros(shape=(num_classes))
            label[idx]=1
            for file in os.listdir(sub_dir):
                image_array = cv2.imread(os.path.join(sub_dir,file), cv2.IMREAD_GRAYSCALE)
                image_array = cv2.resize(image_array, (y_size, x_size))
                images.append(image_array)
                labels.append(label)
        assert len(images)==len(labels)
        shuffle_idx=[i for i in range(len(images))]
        np.random.shuffle(shuffle_idx)
        images=np.array(images)
        labels=np.array(labels)
        images=images[shuffle_idx]
        labels=labels[shuffle_idx]   
        images = np.expand_dims(images, -1)
        return images, labels

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

if __name__=='__main__':
    dm=DataManager('/home/xsh/workspace/mydata/img_align_celeba_128')
    dm._load_celeba()