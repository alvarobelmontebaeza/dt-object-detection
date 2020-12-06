# dt-object-detection

This package performs the task of object detection in the Duckietown simulator. The objects detected are:
    1) Duckies
    2) Cones
    3) Trucks
    4) Buses

## Generating the dataset

To create your own dataset, you just have to execute the data_collection/data_collection.py script. This will
generate a 2000 samples dataset, where each samples contains a 224x224 RGB image, a numpy array with the boxes 
localizing the objetcts detected in that image in the format 'boxes[i] = [xmin,xmax,ymin,ymax]', and a numpy array of labels where labels[i] is the class for the object detected in boxes[i].

## Training your model

To train the model, go to the 'model' directory, and in there execute the train.py script. This will begin the procedure of training a R-CNN with your previously generated dataset for detecting the above mentioned classes. After the training is completed, the resulting model parameters will be stored in weights/model.pt

## Evaluating your model

First, in the root directory of the package, you have to create a Docker image with the following command:

docker build -f Dockerfile -t dt-object-detection .

Then, go to the eval/ directory, and execute one of the following commands, depending on you having a CUDA enabled GPU:

make eval-cpu SUB=dt-object-detection:latest / make eval-gpu SUB=dt-object-detection:latest

This will begin the evaluation of your model using the data in eval/dataset directory.