# ImageAnalysis

A 3D point cloud reconstruction pipeline using RGB-D images, Open3D, and GeoCalib for camera intrinsic estimation.

## Requirements

- Python 3.8+

## Setup

### **1. Clone the repository**
git clone https://github.com/robGogogo/ImageAnalysis.git
cd ImageAnalysis

### **2. Install dependencies**
pip install -r requirements.txt

### **3. Install GeoCalib**
git clone https://github.com/cvg/GeoCalib.git

### **4. Download NYU Depth Labeled Dataset V2**
**I.**

Go to
```
https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
```
Select 
```
Labeled dataset (~2.8 GB)
```
and download

**II.**

Add 
```
nyu_depth_v2_labeled
```
to folder
```
training_dataset
````

**III.**

Delete 
```
placeholder.txt
```

### **5. Extract NYU Dataset**
python utils/extraction/extract.py

  ||
  ||
------
\    /
 \  /
 
extracted_dataset
     | depths
     | images

## **Project Structure**

```
ImageAnalysis/
├── calibration/
├── depth_model/
├── edge_model/
├── visualization_open3d/
├── extracted_dataset/
├── src/
├── toolbox/
├── training_dataset/
├── utils/
├── main.py
└── requirements.txt
```
