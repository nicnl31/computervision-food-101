# Computer Vision with the Food-101 Dataset

This repository is where I store the results of Transfer Learning + Fine-tuning for the Food-101 Dataset. It provides experiments in training and validating three state-of-the-art computer vision architectures: **Inception-ResNet-v2**, **MobileNetV3**, and **NASNet**. Two phases of training are conducted: transfer learning and fine-tuning. The transfer learning phase takes said models with loaded pre-trained weights, replaces their head (i.e. the classifier) with a new head that is tailored to the number of classes for the target dataset, and trains its weights on a small portion of the target dataset, keeping the rest of the weights frozen. The fine-tuning phase then takes the best model in terms of validation loss, and gradually unfreezes more layers to tune to the specifics of the target dataset, where the full set will be used.

<img width="800" alt="Screenshot 2024-05-31 at 5 20 44 PM" src="https://github.com/nicnl31/computervision-food-101/assets/86213993/0fec9bf9-df7b-4b34-af01-7837f3b914e5">

# Preparation
The following steps are taken to prepare for training:

## Image transformations
- Resizing each image to 224x224x3, since 224 is the minimum of the spatial dimension of ImageNet images used for training the original models, and 3 is the number of colour channels (red, green, blue).
- Converting all raw image data to PyTorch tensors.
- Rescale the tensors by using the per-channel means and standard deviations used in the ImageNet dataset, which are `[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]` respectively.

## Traning-validation Configurations
- A 10\% random sample of each set is taken for training all models in the transfer learning phase. This is due to constraints in training time, as using the whole dataset would take a lot of time. A similar approach to neural architecture search is employed by Zoph, Vasudevan, Shlens & Le, [8] where the ultimate model used for training on the ImageNet dataset is assembled via stacking multiple smaller architectures, whose individual configuration is searched by training on the much smaller CIFAR-10 dataset. This solution turns out to be more practical, while still being very effective in generalisation accuracy, as reported in [8]. Stratified sampling is used in order to respect the balance between the number of training examples per class.
- The training subsets (i.e. both the 10\% sample and the full set) are further split into a 70\% training set and a 30\% validation set, where the validation set is used for employing early stopping conditions and reducing the learning rate if training plateaus.
- Create DataLoader objects with a batch size of `128` for each of the training, validation, and testing sets.
- The maximum number of epochs to train is 300, with early stopping employed if the validation loss does not improve by at least the best validation loss plus a small delta of `1e-4` for the next 10 epochs.
- The initial learning rate is `1e-4`, with a ReduceLROnPlateau callback that reduces the learning rate to 0.1 times the current learning rate if the validation loss does not reduce to at least the best validation loss times `1 – 1e-4` for the next 3 epochs. The lower bound for the decrease is `1e-6`.
- The objective function to minimise is the Categorical Cross Entropy (CCE) loss, where $n$ is the number of training examples, $k$ is the number of classes, $p_{ij}$ is the probability of predicting class $j$ in the $i^{th}$ example, and $t_{ij}$ is an indicator if example $i$ is correctly classified as class $j$:
$$\text{CCE} = \frac1n \sum_{i=1}^n \sum_{j=1}^k t_{ij} \log(p_{ij})$$
