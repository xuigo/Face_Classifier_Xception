## Face_Classifier_Xception

Tensorflow Implementation of Face Classification using Xception in dataset celeba.

### Installation requirements

We use GPU and CPU respectively for face parsing, both of which can work normally. And all the dependence has been package to **requirements.txt**. and you can use it to recreate the environment

```shell
pip install -r requirements.txt 
```

### Inference

```python
model_path=''
image_dir=''
threshold=  #[0.-1.]
```

In inference stage, there are three Parameters you need to modified, **model_path**: the pretrained model file path; **image_dir:** test image dir, in which all the image file will be predict and return the result; **threshold:** this value is used to judge the coffidence of attribute, in this case we use the same value defalut is 0.5, and you can adjust it in each attribute so that it can get a better result. 

Before inference, we assume that you have downloaded the pretrained model and place it in a correct place, then just running:

```shell
python inference.py
```

### Train

In order to train the model in other attributes, we also released the train script, and you can easily to start your tast with a  little configurations.

**base_path**: is a dir, which will save the training log and model file;

**dataset_path**: could be str or a list. If str: different attribute files should be placed in different sub-folders; If list, each element is the folder path of every attribute file paths.

**num_classes**: the number of attributes.

and so on ... (details can be found in train.py) 

if all the hyperparameters and data is correct, just run:

```shell
python train.py
```

the classifier result in celeba  as follows:

| ATTR     | Clock<br />Shadow      | Eyebrows      | Attractive   | EyeBags              | Bald         | Bangs               | BigLips              | BigNose              |
| -------- | ---------------------- | ------------- | ------------ | -------------------- | ------------ | ------------------- | -------------------- | -------------------- |
| **ACC**  | 0.9                    | 0.82          | 0.8          | 0.76                 | 0.99         | 0.96                | 0.73                 | 0.75                 |
| **ATTR** | **BlackHair**          | **BlondHair** | **Blurry**   | **Brown<br />Hair**  | **Eyebrows** | **Chubby**          | **Double<br />Chin** | **EyeGlasses**       |
| **ACC**  | 0.86                   | 0.95          | 0.96         | 0.83                 | 0.89         | 0.94                | 0.95                 | 0.99                 |
| **ATTR** | **Goatee**             | **GrayHair**  | **Makeup**   | **Cheek<br />bones** | **Male**     | **Mouth<br />Open** | **Mustache**         | **Narrow<br />Eyes** |
| **ACC**  | 0.96                   | 0.97          | 0.91         | 0.87                 | 0.98         | 0.92                | 0.96                 | 0.9                  |
| **ATTR** | **NoBeard**            | **OvalFace**  | **PaleSkin** | **Pointy<br />Nose** | **Hairline** | **Cheeks**          | **Sideburns**        | **Smile**            |
| **ACC**  | 0.93                   | 0.73          | 0.96         | 0.76                 | 0.93         | 0.94                | 0.96                 | 0.92                 |
| **ATTR** | **Straight<br />Hair** | **WavyHair**  | **Earrings** | **Hat**              | **Lipstick** | **Necklace**        | **Necktie**          | **Young**            |
| **ACC**  | 0.81                   | 0.81          | 0.85         | 0.99                 | 0.93         | 0.87                | 0.94                 | 0.87                 |





