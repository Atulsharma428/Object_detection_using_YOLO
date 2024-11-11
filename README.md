# Object Detection using YOLO

This project demonstrates how to use YOLO (You Only Look Once) for object detection. The model is implemented to detect objects from images and videos with high accuracy and speed.

## Table of Contents
- [Dataset](#dataset)
- [Usage](#usage)
- [Workflow of the model](workflow)
- [YOLO versions](#versions)
- [Training](#training)
- [Results](#results)
  
## Installation
Dataset
We are using a custom dataset for this object detection task. The dataset includes various object classes and annotations in YOLO format. You can also modify the dataset in the sample.yaml file to suit your project.

## Dataset Setup
To set up your dataset, place your images and label files into the appropriate directories as outlined in the sample.yaml file.
```yaml
train: path/to/train/images
val: path/to/val/images
nc: number_of_classes
names: ['class1', 'class2', ...]
```
## Usage
Inference on Videos
To perform inference on a video, use the predict_video.py script. The script allows you to detect objects in a video stream and save the output with bounding boxes.
```python
predict_video.py --source path_to_video --weights path_to_model_weights --output output_path
```

## Workflow of the model
![Model workflow](https://github.com/user-attachments/assets/e14ea41c-5a21-49b2-b4f4-998abbceec8d)



## YOLO versions
### Yolov3
[Visit documentation](https://docs.ultralytics.com/models/yolov5/#performance-metrics)

Flowchart of Yolov3
<img src="https://github.com/user-attachments/assets/33944fa3-d06b-4f1e-b4cf-67c46bcd3102" alt="V3" width="200"/>

### Yolov5

[Visit documentation](https://docs.ultralytics.com/models/yolov3/#supported-tasks-and-modes)

Flowchart of Yolov5
![V5](https://github.com/user-attachments/assets/9508a8bb-f476-42a2-beb7-c6cb61be4c01)

### Yolov8
[Visit documentation](https://docs.ultralytics.com/models/yolov8/)

Flowchart of Yolov8
![V8](https://github.com/user-attachments/assets/1f04c712-f7ca-48b7-b319-3b2c332b1116)



## Training
The YOLO model can be trained on custom datasets. Follow the training steps outlined in the yolo_aplaca.ipynb notebook for training the model. Make sure to update the dataset paths in the configuration file before running the training.

# Example for starting training
```bash
python train.py --img 640 --batch 16 --epochs 50 --data sample.yaml --cfg yolov5l.yaml --weights yolov5l.pt
```
## Results
Once the model is trained, you can visualize the predictions using the predict_video.py script. Sample results are shown below: 
![alpaca_result](https://github.com/user-attachments/assets/ffa56319-c854-4c1b-a84f-ce248b869bfd)

![Construction_tools_dataset_result](https://github.com/user-attachments/assets/1504b947-a86d-410f-8ae9-04076910c7b8)

![Welding_dataset_result](https://github.com/user-attachments/assets/95b505a0-c280-4dc0-a032-61a75285d1cb)




