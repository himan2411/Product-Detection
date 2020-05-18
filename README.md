# Product-Detection
Product recognition and detection from shelf images

This approach of ibject detection is inspired from https://arxiv.org/pdf/1810.01733.pdf on product recognition on store shelves.

And the data used here is [Toward Retail Product Recognition on Grocery Shelves](https://pdfs.semanticscholar.org/280e/57ea3e882f82a60065fedde058ce00769c06.pdf).

Intail steps for data cleaning includes:

1.Rotating few images as they were arbitarily rotated in both train and test folder.
2.Create pandas dataframe for train and test images with names and shelf_id ('Which to be honest was not used').
3.Create dataframe with bounding box as column in product dataframe for images taken from ProductImagesFromShelves folder.
4.Create directory structure and download necessary modules for training like tensorflow, object_detection etc.
$ brew install protobuf
$ cd <path_to_your_tensorflow_installation>
$ git clone https://github.com/tensorflow/models.git
$ cd <path_to_your_tensorflow_installation>/models/research/
$ protoc object_detection/protos/*.proto --python_out=.
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

5.Create labelmap
item {
    id: 1
    name: 'Product'
}
Since we are using only one category.


NOTE: Since the cooradinates are already available its better not to resize image as coordinates of product might change a little.

Steps for creating detector

1. Create tf_record for each image from train folder.
Since we need to train as well as validate our model, the data set will be kept into training (train.record) and validation sets (test.record)

To create tf_record from images I've used https://gist.github.com/iKhushPatel/5614a36f26cf6459cc49c8248e8b5b48.

$ python research/object_detection/dataset_tools/create_tf_record.py

2. Download pretrained model. Here I've used ssd_mobilenet_v2_coco.
3. Modify config file for model at models/research/object_detection/samples/configs
4. Create fine_tune_checkpoint that tells the model which checkpoint file to use. Set this to checkpoints/model.ckpt
5. Put train and test tf_ercord files in models folder.
6. Training
$ python research/object_detection/train.py --logtostderr --train_dir=train --                                                 pipeline_config_path=ssd_mobilenet_v2_coco.config
7. Fine tune the model from checkpoints.
8.Export model
$ python research/object_detection/export_inference_graph.py \    
--input_type image_tensor \    
--pipeline_config_path ssd_mobilenet_v2_coco.config \    
--trained_checkpoint_prefix  train/model.ckpt-<the_highest_checkpoint_number> \    
--output_directory fine_tuned_model

Steps for product detection and evaluation

1. Provide model(after training), checkpoint(from frozen graph) and label path (created in 1st part).
2. Give path to test images folder.
3. Create tensorflow graph for image tensor and tensor dict .
4. Using these tensors create detection graph and pass image (as np array) through the graph.
5. For all the detected anchor boxes if score is 0.5 then only use box and classify as TP rest are FP.
6. For each step of recall ue maximum of precision to get meanAveragePrecision. More on which is available https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173.
7. Create dict with required format and dump it as json.
