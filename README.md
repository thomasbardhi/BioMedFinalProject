#Final Project

## Libraries Used

* Cellpose: This is a deep learning tool for image segmentation. It helps to segment cells in images, which is crucial for our project.
* Normal: This is used for image normalization which is a pre-processing step.
Segmentation: This is used for image segmentation purposes. This is the process of partitioning an image into multiple segments.
* Detectron: Facebook AI's software system that implements state-of-the-art object detection algorithms.
* Matplotlib: This library was used for visualization purposes.
* Imwrite: This function was used for saving the segmented images.
* Hungarian Algorithm (scipy.optimize.linear_sum_assignment): This is used for tracking objects across different frames in a sequence of images.
* LableMe:
 The COCO (Common Objects in Context) dataset format is a widely used format for image segmentation datasets, and many image segmentation models and tools support this format. We used this with detectron

## Key Commands Used
Cellpose:
```python
model = models.Cellpose(gpu=True,model_type='nuclei')
channels = [0,0]
masks, flows, styles, diams = model.eval(img, diameter=30, channels=channels)
```
Segmentation:
```python
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Perform the watershed
labels = watershed(-distance, markers, mask=image)
```
Used Repo(Connor323/Cancer-Cell-Tracking):
See code for the functions we used


## Contributing
Sources: xingyizhou/CenterTrack & Connor323/Cancer-Cell-Tracking

* Preprocessing.py - Tom and Joe : segments a list of images using watershed
* mmdetection_custom_coco.ipynb - Tom and Joe : fed some samples and attempted to do object detection(failed)
* cellpose_segmentation_tracking_with_hungarian_algorithm.py - Tom and Joe: used cellpose to segement and tracked the movement
* UsingNormal Segmentation_Tracking With Hungarian Matching.ipynb - Tom and Joe: code with the segmentation done, used more advanced techniques and used the same matching trying to see if the results were better
* mask_rcnn_r50_fpn.py - Tom and Joe: this the code for the detectron2 model
* comparing_preprocessing_and_deep_learning.py - Tom and Joe : compared the preprocessing and cellpose so see if both produced good fits, got a coeff of .5113
