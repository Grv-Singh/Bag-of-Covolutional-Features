### Live Interaction [![Run on Repl.it](https://repl.it/badge/github/Grv-Singh/Bag-of-Covolutional-Features)](https://repl.it/github/Grv-Singh/Bag-of-Covolutional-Features)

<a href="https://www.slideshare.net/GauravSingh1391/remote-sensing-image-scene-classification" target="_blank">Report</a>

### K Means Clustering
![](https://raw.githubusercontent.com/Grv-Singh/Bag-of-Covolutional-Features/master_new/Bag-of-Visual-Words-Python-master/figure_1.png)

### Frequency of Visual Words (e.g Tree, Car, Lake, Forest)
![](https://raw.githubusercontent.com/Grv-Singh/Bag-of-Covolutional-Features/master_new/Bag-of-Visual-Words-Python-master/vocab.png)

### Outputs
#### Fitness of iteration
![](https://github.com/Grv-Singh/Bag-of-Covolutional-Features/blob/master_new/Output%20Screenshots/Capture.JPG)
#### Count of Visual Word
![](https://github.com/Grv-Singh/Bag-of-Covolutional-Features/blob/master_new/Output%20Screenshots/threshholds.JPG)
#### Sample data
![](https://raw.githubusercontent.com/Grv-Singh/Bag-of-Covolutional-Features/master_new/Output%20Screenshots/gray.JPG)

This project proposed a simple and effective image feature representation method BoCF, for scene classification. Compared with traditional BoVW model in which the visual words are usually obtained by using handcrafted features. The later part of project proposes an application of Grey Wolf Optimizer (GWO) algorithm for satellite image segmentation. The original GWO has been correctly modified to work as an instinctive clustering algorithm. Further, a beneficial performance analysis was carried out by comparing the proposed method with the existing methods. Consequently, in the future work, we need to explore new methods and systems in which the combination of remote sensing data and information can be deployed to promote the state of the art of remote sensing image scene classification. Aim Our goal is to encourage the use of recent technologies like deep learning and recent nature inspired algorithms to detect more descriptive features from an image through remote sensing and to more accurate identification and classification of the images Classification of scenes is difficult if it contains blurry and noisy content. The two significant areas of scene classification problem are: learning and scenes models for formal categories. If the images are affected due to noise, poor quality, occlusion or background clutter, it becomes quite a challenge to classify an image. This difficult gets multiplied whenever an image consists of many objects. There has been a invariable raise in new classification algorithms, techniques Introduction Remote Sensing Image Scene Classification plays an essential role in a broad range of applications. In this project, we have presented a mechanism for remote sensing and image classification of large dataset image collections. . Bag of Visual Words (BoVW) model is used in first part of the project. However, the traditional BoVW model only captures the local patterns of images by utilizing local features. Then proposed Bag of Convolutional Features (BoCF) generates visual words from deep convolutional features using off- the-shelf convolutional neural networks. The further part of project proposes an application of Grey Wolf Optimizer (GWO) algorithm for satellite image segmentation. The original GWO has been suitably modified to work as an automatic clustering algorithm. Results Accuracies of BoVW, BoCF and GWO respectively.
* The accuracy of traditional BoVW method with dense SIFT is 41.72% under training ratio of 10% and 44.97% under the training ratio of 20%.
* The accuracy of BoCF method is almost doubled by 82.65% under training ratio of 10% and 84.32% under the training ratio of 20%.
* The accuracy of BoVW + GWO method is 78.70% under training ratio of 10% and 80.60% under the training ratio of 20%. Method
* Handcrafted Feature Learning: These methods mainly focus on using acceptable amount of engineering handiness and domain expertise to design various human engineering features.
* Unsupervised Feature Learning: Unsupervised feature learning aims to learn a set of basic functions (or filters) used for feature encoding, in which the input of the functions is a set of handcrafted features
* Deep Feature Learning Based Deep learning models that are composed of multiple processing layers can learn more powerful feature representations of data with multiple levels of abstraction.
