from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from models import mini_XCEPTION

from utils import DataManager
from utils import split_data
from utils import preprocess_input

# parameters
batch_size = 32
num_epochs = 10000
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 3
patience = 50
base_path = './trained_models/'
task_name='age'

dataset_path=['/home/xsh/workspace/mydata/attribute_data/age_kids_256','/home/xsh/workspace/mydata/attribute_data/age_adults_256','/home/xsh/workspace/mydata/attribute_data/age_elders_256'] #list or str
#dataset_path='/home/xsh/workspace/mydata/img_align_celeba_128'
# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)
# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
# model.compile(optimizer='adam', loss='categorical_crossentropy',
#               metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

# callbacks
log_file_path = base_path + task_name + '_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,patience=int(patience/4), verbose=1)
trained_models_path = base_path + task_name + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]
# loading dataset
data_loader = DataManager(dataset_path, image_size=input_shape[:2])
faces, labels = data_loader.get_data()
faces = preprocess_input(faces)
num_samples, num_classes = labels.shape
train_data, val_data = split_data(faces, labels, validation_split)
train_faces, train_labels = train_data
model.fit_generator(data_generator.flow(train_faces, train_labels,batch_size), steps_per_epoch=len(train_faces) // batch_size, epochs=num_epochs, verbose=1, callbacks=callbacks, validation_data=val_data)



