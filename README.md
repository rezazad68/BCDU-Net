# [Attention LSTM U-net: Medical Image Segmentation ](https://www.miccai2019.org/)


Deep Auto-Encoder-Decoder Network for Medical Image Segmentation with State of the Art results on Skin lesion segmentation and Retina Blood vessel segmentation. This Method applies attentional LSTM layers in U-net structure to capture detail information. If this code helps with your research please consider citing following paper:
</br>
> [R. Azad](https://rezazad.000webhostapp.com/), [M. Asadi](http://ipl.ce.sharif.edu/members.html), [S. Kasaei](http://sharif.edu/~skasaei/), [Sergio Escalera](http://sergioescalera.com/organizer/) "Dynamic 3D Hand Gesture Recognition by Learning Weighted Depth Motion Maps", IEEE Transaction on CSVT, 2018, download [link](https://ieeexplore.ieee.org/document/8410578/).
## Updates
- September 2, 2017: First release (Complete implemenation for [MSR Action 3D](http://www.uow.edu.au/~wanqing/#MSRAction3DDatasets) data set)
- May 5, 2018: Complete implemenation for [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) data set added. Accuracy rate 75.16 and 68.66 with deep and non deep features achieved respectively. It is worth to mention that our method achieved highest performance on depth data (75.16))
- July 14, 2018: Paper [link](https://ieeexplore.ieee.org/document/8410578/) in IEEE Transaction on Circuits and Systems for Video Technology
## Prerequisties and Run
This code has been implemented in Matlab 2016a and tested in both Linux (ubuntu) and Windows 10, though should be compatible with any OS running Matlab. following Environement and Library needed to run the code:
- Matlab 2016
- [VL feat 0.9.20](http://www.vlfeat.org/)
## Run Demo
Run the `Main_MSRAction3D()` for both feature extraction and classification of dynamic 3D action. The `Main_MSRAction3D` uses `Step1_Extract_Featues` for extracting spatio-temporal features from different represantion of 3D video and `Step2_Description_Classification` for aggregating of descriptions and classification phase. These two functions can be use seperetely too. Function such as `Video Summarization()`, `Forward Bakward Motion()`, `Difference Forward Energy()`, `Temporal Sequence Generating()`, `Binary Weighted Mapping()`, and extracting `Regional LBP and HOG features()` has been implemented in 'Video_Analyser' class. the `Description_Classification class` contains functions that related to Vlad representation and dimension reduction phase.    
</br>
## Quick Overview
![Action and Hand Gesture Recognition](https://user-images.githubusercontent.com/20574689/29744825-1f43af08-8ac2-11e7-894e-2cb1b316185a.png)
=========
## Results
For evaluating the performance of the proposed method, three public data sets has been considered. In bellow, results of using three different strategies for 3D Action recognition demonstrated.
</br>
- Strategy 1 : Vlad Representation of Spatio-Temporal HOG Features from Different Representations
- Strategy 2 : Vlad Representation of Spatio-Temporal LBP Features from Different Representations 
- Strategy 3 : Vlad Representation of Spatio-Temporal HOG+LBP Features from Different Representations 

Data Set| Strategy 1 | Strategy 2| Strategy 3
------------ | -------------|----|----
[MSR Gesture 3D](http://www.uow.edu.au/~wanqing/#MSRAction3DDatasets) | 96.22| 96.52|98.05
[SKIG](http://lshao.staff.shef.ac.uk/data/SheffieldKinectGesture.htm) | 95.0|95.60|97.31
[MSR Action 3D](http://www.uow.edu.au/~wanqing/#MSRAction3DDatasets)|91.94|91.57|95.24
[NTU RGB+D](https://github.com/shahroudy/NTURGB-D)|-|-|75.16 deep

#### Effect of Choosing number of Visual Words on each data set has been illustrated in the followin table:
Selecting number of Visual Words on each data sets related to number of classes on each data set. In the following table these information has been evaluated. </br>

Number of Visual Words|25|30|40|50|70|100|128
---|---|---|---|---|---|---|---
[MSR Gesture 3D](http://www.uow.edu.au/~wanqing/#MSRAction3DDatasets) |98.05|97.50|97.50|96.94|96.66|96.38|96.38
[SKIG](http://lshao.staff.shef.ac.uk/data/SheffieldKinectGesture.htm) |97.13|97.22|96.67|96.48|96.76|96.30|96.02
[MSR Action 3D](http://www.uow.edu.au/~wanqing/#MSRAction3DDatasets) |92.31|93.04|93.04|94.14|95.24|93.77|93.77



#### Choosing appropriate number of PCA components
in the following table accuracy rate for choosing different amount of PCA components depicted. </br>

PCA Components|70|100|130|160|190|220|250
---|---|---|---|---|---|---|---
[MSR Gesture 3D](http://www.uow.edu.au/~wanqing/#MSRAction3DDatasets) |97.50|97.77|98.05|97.50|98.05|97.50|97.50
[SKIG](http://lshao.staff.shef.ac.uk/data/SheffieldKinectGesture.htm) |96.57|97.13|97.22|97.31|97.31|96.94|97.31
[MSR Action 3D](http://www.uow.edu.au/~wanqing/#MSRAction3DDatasets) |94.54|94.87|95.24|95.25|94.87|94.87|94.87

### Query
For any query please contact us for more information.

```python
razad@ce.sharif.edu

```
