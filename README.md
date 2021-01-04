---
layout: post
title:  "Object Detection using Mask R-CNN"
date:   2020-12-25 12:00:00
---

# Object Detection

Task of finding different objects in images and classifying them. There are four main classes of problems in object detection:

![Fei-Fei Li Stanford Course — Detection And Segmentation](https://1.bp.blogspot.com/-rhPX8JpRpZM/X-FFuUHfvWI/AAAAAAAAJ3Q/QD1FzBV2pik4ftrpwR86BwE99OJ_qRU_QCLcBGAsYHQ/s0/img3.png)

***Figure 1**. Fei-Fei Li Stanford Course — Detection And Segmentation*

## Table of Content

1. R-CNN
2. Fast R-CNN
3. Faster R-CNN
4. Mask R-CNN
5. Anaconda Install and Setup for Mask R-CNN

## [Regions with CNN features (R-CNN)](https://arxiv.org/pdf/1311.2524.pdf) 

R-CNN (2014) was an early application of CNN to do object detection. Ross Girshick and his team found that using CNN to do object detection performs way better than using Histogram of Oriented Gradient (HOG) features which was a popular method prior to this success.

R-CNN works similar to how our brain might do object detection. It proposes a bunch of boxes in the image, specifically 2000 boxes, and see if any of them correspond to some object or image. It generates these regions called region proposals by using a process called selective search. At a high level, selective search looks at the image through windows of different sizes and it clusters together similar regions to create larger regions and identify objects based on color, texture, and intensity. Once the 2000 region proposals are selected, R-CNN warps the region to a standard square size (make all regions the same size maybe 256 x 256) and passes it through an AlexNet (Winner of ImageNet 2012) CNN to compute CNN features. The reason to warp the images to fixed shape is because it required by fully connected layers. On the final layer of the R-CNN, a Support Vector Machine (SVM) is used t classify if the region is an object, and if so which object is it. Now having found the object, we can tighten the box to get the true dimensions of the object. To get tighter bounding boxes around the objects, R-CNN run a linear regressor on the region proposals to generate a tighter bounding box coordinates. 

![R-CNN ](https://1.bp.blogspot.com/-I7EhelkrUr0/X-FFv69f3sI/AAAAAAAAJ3g/tUVPeQdpJv0WKsZ1hsUE5E5rBaNGgElKwCLcBGAsYHQ/s0/img6.png)

***Figure 2**. R-CNN*

### Drawbacks of R-CNN

1. Slow because it requires a forward pass of AlexNet for every region proposal. It could be 2000 forward passes if 2000 region proposals are used.

2. Train 3 different models separately: CNN to generate features, classifier that predicts the class, and the regression model to tighten the bounding box.

This makes the pipeline extremely hard to train.

### Summary of R-CNN

1. Generate a set of proposals for bounding boxes.

2. Run the bounding boxes through pre-trained Alexnet to extract features and input that to a Support Vector Machine (SVM) to classify what object is in the bounding box.
 
3. Run the box through a linear regression model to output tighter coordinate once the object in the box has been classified.

## Fast R-CNN

2015 - Implemented by the same author (Ross Girshick) to improve the speed of R-CNN and simplify it. He solved both the drawbacks of R-CNN.

### Solution - Region of Interest (ROI) Pooling

Girshick realized that the forward pass through AlexNet of the regions proposals images invariably overlapped causing it to run the same computation through the CNN again and again. His solution to this problem was to run the CNN once per image to generate a convolutional feature map. He also still produced the regions of interests using the selective search process and stored the coordinates of the top left and bottom right corners of the region. For every region proposal, we take the corresponding area from the convolutional feature map and scales it using max-pooling to produce fixed region proposals. This is known as Region of Interest Pooling algorithm. Here is a simple example of ROI Pooling.

![](https://1.bp.blogspot.com/-ZPSfNpf8IDg/X-H7GxPFkiI/AAAAAAAAJ44/E6yjqLewyW8LLeEUm9XXN9Qqo0HPxe5GwCLcBGAsYHQ/s1274/Screen%2BShot%2B2020-12-22%2Bat%2B8.56.16%2BAM.png)

[***Figure 3**. ROI Pooling*](https://deepsense.ai/region-of-interest-pooling-explained/)

Next pass these regions to the fully connected layer and run the Softmax classifier instead of SVM and also use the bounding box regressor to get tighter bounding boxes.

![Fast R-CNN](https://1.bp.blogspot.com/-3uK_kTwWpqQ/X-FFwtt6ReI/AAAAAAAAJ3k/xDPGoyzFamog9n3mwMzfV9wWLtG7GPiyACLcBGAsYHQ/s0/img7.jpeg)

***Figure 4**. Fast R-CNN*

This is much faster than the simple R-CNN since all it takes is a single pass through the image instead of 2000 passes of region proposals through AlexNet.

### Drawbacks

1. We are still using the slow selective search process to generate region proposals. This is the bottleneck in the process where you first have to generate bounding boxes in the image to detect objects.

## Faster R-CNN

R-CNN and Fast R-CNN both uses selective search to find out the region proposals. Selective search process is a slow and affects the performance of the network. [3] proposed Faster R-CNN, where instead of using selective search algorithm on the feature map to identify the region proposals, a separate network called regions proposal network (RPN) is used to predict the region proposals. 

The input image is feed in as input to a VGG or ImageNet CNN to generate the feature map. The map is inputed to the region proposal network which will generate the regions proposals and rest of the process stays same as Fast R-CNN.

![enter image description here](https://1.bp.blogspot.com/-x_AfOO7MuSs/X-FFu35KWII/AAAAAAAAJ3Y/czFnmkVPp6sOD2DznUvwKLzwXujySiUQACLcBGAsYHQ/s0/img4.jpeg)

***Figure 5**. Faster R-CNN*

### Inside the Regions Proposal Network (RPN)

RPN ranks region boxes called anchors. Higher rank anchors contain objects and lower rank are ignored. These boxes or anchors are not placed at each pixel location instead they are placed at a stride *s*. Stride is the number of pixels shift over input matrix or image. for example *s* = 16  in a *600 x 800* image we will have *(600/16) x (800/16) = 37 x 50 = 1850* anchor center locations as shown in the image below.

![enter image description here](https://1.bp.blogspot.com/-Tkcjc5nHrWA/X-VLmsrIWwI/AAAAAAAAJ5k/ohmXLLgpb-Q796GzkTE9Rd-ulcnC66wCgCLcBGAsYHQ/s320/rpn-anchors-centers.png)

[***Figure 6**. Anchor or box centers through the original image.*](https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/)

At each anchor center, we define three size of boxes *(128 x 128, 256 x 256, 512 x 512)* and use three aspect ratios *1:1, 1:2 and 2:1*. This will produce *9* anchors or boxes around each anchor center in the image.

![enter image description here](https://1.bp.blogspot.com/-gzS7_sEP_s4/X-VT27yD-dI/AAAAAAAAJ50/FK3sWm4hMP8gKDvTI2jU7P5Dpvlpvt1wgCLcBGAsYHQ/s0/anchors.png)

*[**Figure 7**. 9 Anchors or boxes around pixel location (320, 320).](https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8)*

In total, we will have *1850 x 9 = 16,650* anchors in the image. As you can see in the image below that using this approach you will be able to cover every corner of the image and look for objects.

![enter image description here](https://1.bp.blogspot.com/-me-hC5S4jFw/X-VXOgzPhGI/AAAAAAAAJ6E/AMJb_5bGXj8g4zyz95403z3wfymmAEZbACLcBGAsYHQ/s0/anchors-progress.png)

***Figure 8**. **Left**: Anchors, **Center**: Anchor for a single point, **Right**: All anchors*

RPN takes all these anchors as input and outputs a set of good region proposals where there might be some object.

These region proposal are different sizes. Next we will use ROI pooling to make all features the same size. This is essentially the same process as Fast R-CNN to produce the final results. 

## Mask R-CNN

Extended Faster R-CNN in 2017 to pixel level segmentation. In Mask R-CNN, a fully convolutional network (FCN) is added on top of the CNN features of Faster R-CNN to generate a mask or a segmentation output. Every pixel belonging to the object in the image is included in the mask output.

![enter image description here](https://1.bp.blogspot.com/-7asg0i3VKD0/X-VeFLLO_HI/AAAAAAAAJ6g/aWJUgwxgcisM6EGX2Nr-r1OfZg5fx1yGgCLcBGAsYHQ/s0/img5.png)

***Figure 9**. Mask R-CNN* 