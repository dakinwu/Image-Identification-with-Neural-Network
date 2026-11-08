# Pneumonia Classification with AlexNet and VGG19

A TensorFlow/Keras-based image classification project that uses AlexNet and VGG19 to distinguish normal chest X-rays from images labeled as pneumonia.

## Overview

This project compares two convolutional neural networks (models that learn visual patterns from images) and explores how training choices affect classification performance.

Each input image is assigned to one of two dataset categories:

- `NORMAL`
- `PNEUMONIA`

The experiments compare model architectures, optimizers (methods that update model parameters during training), and batch sizes (the number of images processed per training step).

## Models

| Model | Implementation |
| --- | --- |
| AlexNet | Constructed layer by layer with convolutional and classification layers. |
| VGG19 | Uses an ImageNet-pretrained base with frozen layers and a custom classification head. |

The VGG19 setup uses transfer learning (reusing patterns learned from another image dataset).

## Dataset

The project uses the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

Download the dataset separately. The notebooks read image folders from the working directory:

```text
train/
├── NORMAL/
└── PNEUMONIA/

test/
├── NORMAL/
└── PNEUMONIA/
```

The Adam notebook creates a validation subset from `train/`.

## Workflow

1. Load images from the class folders.
2. Resize images and rescale pixel values.
3. Apply data augmentation (small image transformations that introduce training variation).
4. Build the AlexNet and VGG19 models.
5. Train the models under the selected settings.
6. Compare learning curves and test predictions.
7. Save model files, training histories, and figures.

## Repository Contents

| File | Description |
| --- | --- |
| `Alexnet-Adam+Vgg19-Adam.ipynb` | AlexNet and VGG19 experiments using Adam. |
| `Alexnet-SGD+Vgg19-RSMprop.ipynb` | Experiments using alternative optimizers. |
| `Alexnet-32+Vgg19-32.ipynb` | Batch-size experiments. |
| `Chest Image Identification.pdf` | Detailed project report. |
| `FINAL.pptx` | Project presentation. |

## Tools

- Python and Jupyter Notebook
- TensorFlow and Keras for neural network training
- NumPy and pandas for data handling
- Matplotlib and seaborn for visualization
- scikit-learn for confusion matrices (tables comparing predicted and actual labels)

## Getting Started

1. Obtain the repository files and download the dataset.
2. Arrange the image folders as shown above.
3. Prepare a Python environment with the required packages.
4. Create the output directories `models/`, `history/`, and `images/`.
5. Open the notebook for the experiment you want to inspect.
6. Review data paths and training settings before running the cells.

VGG19 may download its pretrained weights when first initialized.

## Evaluation and Reproducibility

The notebooks include training and validation curves, test accuracy, and confusion matrices.

The code uses older Keras/TensorFlow conventions and mixes imports from `keras` and `tf.keras`. Package compatibility, metric names, and checkpoint settings may need adjustments before rerunning.

This is an experimental classification study using dataset labels; clinical performance has not been established.
