# Ship-Detection-Project

Deep Learning models for detecting ships in satellite images ([Kaggle Airbus Competition])(https://github.com/arcrank/shipDetectionCS682)
This project is completed by [acrank](https://github.com/arcrank), [tana777](https://github.com/tana777) and myself. I was responsible for implementing the Mask-RCNN model and gathered our final submission file.

Using Python3 and usually jupyter notebooks

We explored Mask R-CNN and UNet model, both using a discriminator network to classify if an image has any ships at all first.

We have used some template code provided by the competition to get started, mostly related to visualizing images as well as loading the supplementary data in the training segmentation files.

At the end of the project, we implemented our model (Mask-RCNN) on out-of-box panama canal images (thanks [acrank](https://github.com/arcrank) for searching these test images) and obtain very impressive results! The results indicate that our model has a good generalizability.

![test1](https://github.com/ZTong1201/Ship-Detection-Project/blob/master/test_results/test1.png)
![test2](https://github.com/ZTong1201/Ship-Detection-Project/blob/master/test_results/test2.png)
![test4](https://github.com/ZTong1201/Ship-Detection-Project/blob/master/test_results/test4.png)
