# Customer interactions with store layout.

Customer interaction monitor implemented with YOLOv4, DeepSort, and TensorFlow. YOLOv4 is a state of the art algorithm that uses deep convolutional neural networks to perform object detections. We can take the output of YOLOv4 feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker and use the tracks to identify customer movement.

## Demo of Customer interaction monitor
<p align="center"><img src="data/helpers/demo.gif"\></p>
This was tested on NVIDIA GeForce RTX 2060 

Check the video in outputs/ to see the output in full resolution.

## Getting Started
To get started, install the proper dependencies either via Anaconda. I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version for you. I also recommend using a GPU for a smooth framerate.

### Conda 

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```
## Installing Tesseract (On windows)
Tesseract is being used to extract date-time informatiom from the video. Head over to the download page https://github.com/UB-Mannheim/tesseract/wiki. This was tested with the 64bit version of the installer. Download the installer and install tesseract. Make sure you remember the install location while installing. The default location should be like this ```C:\Program Files\Tesseract-OCR```. 
If your download location is the same as mentioned before, you can proceed to the next step. If you installed it in a custom location,
head over to the file named customer_detection.py and change line 27 
```bash
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
``` 
to the value of your custom location. For example if the custom loction is ```C:\Users\Public\Documents\Tesseract-OCR``` then the variable becomes 
```bash
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Public\Documents\Tesseract-OCR\tesseract.exe'
```
## Downloading YOLOv4 Pre-trained Weights
Our object tracker uses YOLOv4 to make the object detections, which deep sort then uses to track. There exists an official pre-trained YOLOv4 object detector model that is able to detect 80 classes. For easy demo purposes we will use the pre-trained weights for our tracker.
Download pre-trained yolov4 and yolov4-tiny tensorflow model: https://drive.google.com/file/d/1r0aHB-dVZb_tTR0NMwFLatAfEp8OmHj-/view?usp=sharing

Extract the file named YOLOTFModel.zip which contains YOLOv4 and YOLOv4-tiny tensorflow weights and model.
Copy the two folders named yolov4-416 and yolov4-tiny-416 into the directory named checkpoints.

## The directory tree should be like this
```
customer_detection
???   README.md
???   customer_detection.py
|   ...
???
????????????checpoints
???   ????????????yolov4-416
|   |   |   assets
|   |   |   variables
|   |   |   saved_model.pb
???   ???
???   ????????????yolov4-tiny-416
???       ???   assets
???       ???   variables
???       ???   saved_model.pb
???   
????????????core
|   ???   ...
????????????data
|   |   ...
...
```

## Running the Customer Detection monitor using YOLOv4
```bash
# Run yolov4 deep sort customer detection monitor on video (It will take some time to execute)
python customer_detection.py --video ./data/video/Part1.mp4 --output ./outputs/demo.avi --model yolov4

# Run yolov4 deep sort customer detection monitor on webcam (set video flag to 0)
python customer_detection.py --video 0 --output ./outputs/webcam.avi --model yolov4
```
The output flag allows you to save the resulting video of the object tracker running so that you can view it again later. Video will be saved to the path that you set. (outputs folder is where it will be if you run the above command!)
## Running the Customer Detection Monitor with YOLOv4-Tiny
The following commands will allow you to run yolov4-tiny model. Yolov4-tiny allows you to obtain a higher speed (FPS) for the tracker at a slight cost to accuracy. Make sure that you have downloaded the tiny weights file and added it to the checkpoints folder for this to work!
```
# Run yolov4-tiny object tracker
python customer_detection.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny
```

## Resulting Video/In out data
As mentioned above, the resulting video will save to wherever you set the ``--output`` command line flag path to. I always set it to save to the 'outputs' folder. You can also change the type of video saved by adjusting the ``--output_format`` flag, by default it is set to AVI codec which is XVID.
The .csv data is stored in the outputs folder by default as well named ```data.csv```.

## Command Line Args Reference

```bash
 customer_detection.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/test.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf)
    (default: tf)
  --model: yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.50)
  --dont_show: dont show video output
    (default: False)
  --info: print detailed info about tracked objects
    (default: False)
```

### References  

   Huge thanks goes to TheAIGuy, hunglc007 and nwojke for creating the backbones of YOLOv4 and DeepSort:
  * [Object tracker using TF, YOLOv4 and DeepSORT](https://github.com/theAIGuysCode/yolov4-deepsort)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
