# [Attention LSTM U-net: Medical Image Segmentation ](https://www.miccai2019.org/)


Deep auto-encoder-decoder network for medical image segmentation with state of the art results on skin lesion segmentation and retina blood vessel segmentation. This method applies attentional LSTM layers in U-net structure to capture details information. If this code helps with your research please consider citing the following paper:
</br>
> [R. Azad](https://scholar.google.com/citations?user=Qb5ildMAAAAJ&hl=en), [M. Asadi](http://ipl.ce.sharif.edu/members.html), [S. Kasaei](http://sharif.edu/~skasaei/), [Sergio Escalera](http://sergioescalera.com/organizer/) "Dynamic 3D Hand Gesture Recognition by Learning Weighted Depth Motion Maps", IEEE Transaction on CSVT, 2018, download [link](https://ieeexplore.ieee.org/document/8410578/).
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
![Retinal Blood Vessel Segmentation result 1](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/Figure_1.png)
=========
## Results
For evaluating the performance of the proposed method, Two challenging task in medical image segmentaion has been considered. In bellow, results of the proposed approach illustrated.
</br>

### Performance Evalution on the Skin Lesion Segmentation task

Methods | Year |F1-scores | Sensivity| Specificaty| Accuracy | AUC
------------ | -------------|----|-----------------|----|---- |---- 
Chen etc. all [Hybrid Features](https://link.springer.com/article/10.1007/s00138-014-0638-x)        |2014	  |	-       |0.7252	  |0.9798	  |0.9474	  |0.9648
Azzopardi  etc. all [Trainable COSFIRE filters ](https://www.sciencedirect.com/science/article/abs/pii/S1361841514001364)   |2015	  |	-       |0.7655	  |0.9704	  |0.9442	  |0.9614
Roychowdhury and etc. all [Three Stage Filtering](https://ieeexplore.ieee.org/document/6848752)|2016 	|	-       |0.7250	  |**0.9830**	  |0.9520	  |0.9620
Liskowsk  etc. all[Deep Model](https://ieeexplore.ieee.org/document/7440871)	  |2016	  |	-       |0.7763	  |0.9768	  |0.9495	  |0.9720
Qiaoliang  etc. all [Cross-Modality Learning Approach](https://ieeexplore.ieee.org/document/7161344)|2016	  |	-       |0.7569	  |0.9816	  |0.9527	  |0.9738
Ronneberger and etc. all [U-net](https://arxiv.org/abs/1505.04597)	     	    |2018   | 0.8142	|0.7537	  |0.9820	  |0.9531   |0.9755
Alom  etc. all [Recurrent Residual U-net](https://arxiv.org/abs/1802.06955)	|2018	  | 0.8149  |0.7726	  |0.9820	  |0.9553	  |0.9779
Oktay  etc. all [Attention U-net](https://arxiv.org/abs/1804.03999)	|2018	  | 0.8155	|0.7751	  |0.9816	  |0.9556	  |0.9782
Alom  etc. all [R2U-Net](https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf)	        |2018	  | 0.8171	|0.7792	  |0.9813	  |0.9556	  |0.9784
Azad etc. all [Proposed Attention LSTM-U-net](https://github.com/rezazad68/LSTM-U-net/edit/master/README.md)	  |2019 	| **0.8222**	|**0.8012**	  |0.9784	  |**0.9559**	  |**0.9787**


### Some of the Estimated masks for test data

![Retinal Blood Vessel Segmentation result 1](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/Figure_1.png)
![Retinal Blood Vessel Segmentation result 2](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/Figure_2.png)
![Retinal Blood Vessel Segmentation result 3](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/Figure_3.png)

# a

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
