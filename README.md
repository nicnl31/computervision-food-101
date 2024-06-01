# Computer Vision with the Food-101 Dataset

This repository is where I store the results of Transfer Learning + Fine-tuning for the Food-101 Dataset. The dataset can be downloaded from here: https://www.kaggle.com/datasets/dansbecker/food-101
It is also officially described in the paper by Bossard, Guillaumin & Gool. [1]

The `food-101` folder contains the metadata that lists the images in each of the train and test sets. Actual images are not uploaded due to their size. 

The project provides experiments in training and validating three state-of-the-art computer vision architectures: **Inception-ResNet-v2**, **MobileNetV3**, and **NASNet**. [3], [4], [6] Two phases of training are conducted: transfer learning and fine-tuning. The transfer learning phase takes said models with loaded pre-trained weights, replaces their head (i.e. the classifier) with a new head that is tailored to the number of classes for the target dataset, and trains its weights on a small portion of the target dataset, keeping the rest of the weights frozen. The fine-tuning phase then takes the best model in terms of validation loss, and gradually unfreezes more layers to tune to the specifics of the target dataset, where the full set will be used.

# The Food-101 Dataset

It is a collection of 101,000 images of 101 different classes of food, which was first proposed in Bossard, Guillaumin & Gool [1] as a standardised dataset for computer vision. For each food class, there are 750 training images and 250 testing images, and the authors of [1] states that only the testing set is manually cleaned, while the training set images are left in their raw form. Each of the original images has a different dimension, with the longest side’s length no more than 512 pixels.

<img width="800" alt="Screenshot 2024-05-31 at 5 20 44 PM" src="https://github.com/nicnl31/computervision-food-101/assets/86213993/0fec9bf9-df7b-4b34-af01-7837f3b914e5">

# Preparation
The following steps are taken to prepare for training:

## Image transformations
- Resizing each image to 224x224x3, since 224 is the minimum of the spatial dimension of ImageNet images used for training the original models, and 3 is the number of colour channels (red, green, blue).
- Converting all raw image data to PyTorch tensors.
- Rescale the tensors by using the per-channel means and standard deviations used in the ImageNet dataset, which are `[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]` respectively.

## Traning-validation Configurations
- A 10\% random sample of each set is taken for training all models in the transfer learning phase. This is due to constraints in training time, as using the whole dataset would take a lot of time. A similar approach to neural architecture search is employed by Zoph, Vasudevan, Shlens & Le, [2] where the ultimate model used for training on the ImageNet dataset is assembled via stacking multiple smaller architectures, whose individual configuration is searched by training on the much smaller CIFAR-10 dataset. This solution turns out to be more practical, while still being very effective in generalisation accuracy, as reported in [2]. Stratified sampling is used in order to respect the balance between the number of training examples per class.
- The training subsets (i.e. both the 10\% sample and the full set) are further split into a 70\% training set and a 30\% validation set, where the validation set is used for employing early stopping conditions and reducing the learning rate if training plateaus.
- Create DataLoader objects with a batch size of `128` for each of the training, validation, and testing sets.
- The maximum number of epochs to train is 300, with early stopping employed if the validation loss does not improve by at least the best validation loss plus a small delta of `1e-4` for the next 10 epochs.
- The initial learning rate is `1e-4`, with a ReduceLROnPlateau callback that reduces the learning rate to 0.1 times the current learning rate if the validation loss does not reduce to at least the best validation loss times `1 – 1e-4` for the next 3 epochs. The lower bound for the decrease is `1e-6`.
- The objective function to minimise is the Categorical Cross Entropy (CCE) loss, where $n$ is the number of training examples, $k$ is the number of classes, $p_{ij}$ is the probability of predicting class $j$ in the $i^{th}$ example, and $t_{ij}$ is an indicator if example $i$ is correctly classified as class $j$:
$$\text{CCE} = \frac1n \sum_{i=1}^n \sum_{j=1}^k t_{ij} \log(p_{ij})$$

# Performance analysis and possible future improvements

The best model in this phase is MobileNetV3, which achieves a `46.69%` out-of-sample top-1 accuracy when trained with 10\% of the dataset, with the lowest validation loss of `2.3328`. On further fine-tuning and using the full dataset for this model, the out-of-sample top-1 accuracy improves to `67.92%` with the lowest validation loss of `1.4674`.

<img width="281" alt="Screenshot 2024-05-31 at 6 14 51 PM" src="https://github.com/nicnl31/computervision-food-101/assets/86213993/c5793fce-577d-4ce9-b7cf-f2cafc33ee46">

<img width="841" alt="Screenshot 2024-05-31 at 6 15 16 PM" src="https://github.com/nicnl31/computervision-food-101/assets/86213993/7b22e942-af5e-444d-8c78-bc35b8dc741c">

MobileNetV3 experienced a significant boost in predictive performance when fine-tuned to the Food-101 dataset, compared to the transfer learning phase. This is due to there being more data available during fine-tuning, as well as more layers unfrozen to tune to the finer details of the target dataset. However, its top-1 accuracy is still subpar comparing to its top-1 accuracy of 75–76% on the original ImageNet dataset as reported in Howard et al. [3], suggesting further work to be done to tackle overfitting issues.

These results could be possibly improved via:
- Adjusting learning rates more proactively. For the experiments in this project, the learning rate is set quite low initially, and only decays at a 0.1 rate if training plateaus, whereas in the original paper the learning rate is much higher (0.1) with a tough decay of 0.01 strictly after every 3 epochs.
- Targeted unlocking of layers. This project used a somewhat arbitrary choice of layers for unfreezing in MobileNetV3 during fine-tuning, which are some of the layers towards the end. The fact that weights are tuned at less expressive layers of representation without e.g. residual connections [5] may have made fine-tuning more difficult in this instance. In the next experiments, layers at the beginning may be fine-tuned as well, e.g. the input convolutional layer which may be more sensitive to the specifics of the target dataset than is the case for subsequent layers.

# References
[1] L. Bossard, M. Guillaumin, and L. Gool, “Food-101 -Mining Discriminative Components with Random Forests.” Available: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food- 101/static/bossard_eccv14_food-101.pdf.

[2] B. Zoph, V. K. Vasudevan, J. Shlens, and Q. V. Le, “Learning Transferable Architectures for Scalable Image Recognition,” arXiv (Cornell University), Jul. 2017, doi: https://doi.org/10.48550/arxiv.1707.07012.

[3] A. Howard et al., “Searching for MobileNetV3,” arXiv.org, Nov. 20, 2019. https://arxiv.org/abs/1905.02244v5 (accessed Mar. 12, 2024).

[4] C. Szegedy et al., “Going deeper with convolutions,” 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1–9, 2015, doi: https://doi.org/10.1109/cvpr.2015.7298594.

[5] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” arXiv (Cornell University), Dec. 2015, doi: https://doi.org/10.48550/arxiv.1512.03385.

[6] C. Szegedy, S. Ioffe, V. Vanhoucke, and A. A. Alemi, “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,” arXiv (Cornell University), Feb. 2016, doi: https://doi.org/10.48550/arxiv.1602.07261.
