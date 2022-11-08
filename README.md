# EE5907 CA2

Coursework for EE5907 Pattern Recognition CA2, AY 2022/2023 Semester 1.

The aim of this project is to construct a face recognition system via Principal Component Analysis (PCA), Linear Discriminative Analysis (LDA), Support Vector Machine (SVM) and Convolution Neural Network (CNN). PCA and LDA are used to perform data dimensionality reduction and visualization, in order to understand underlying data distribution. Then lower-dimensional data are classified based on the nearest neighbour classifier. In addition, SVM and CNN are also used to classify face images. PCA and LDA are from scratch, while the package `libsvm` is used for SVM models. CNN is builed based on `pytorch`.

The detail setting of the experiments can be found in the corresponding report, which is accessible [here]().

## Requirements


- Python 3.6
- matplotlib == 3.5.2
- numpy == 1.21.5
- pandas == 1.3.5
- libsvm == 3.23.0.4
- torch == 1.10.2


Dependencies can be installed using the following command:
```
pip install -r requirements.txt
```

## Data
The dataset used in this project contains the [CMU PIE dataset](https://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html) and the face photos taken by students themselves. There are in total of 68 different subjects and I selected the first 25 out of them. For each chosen subject, 70% for training and the remaining 30% for testing. Besides the CMU PIE images, there are 10 selfie photos used as samples after being converted to grey-scale images and resized into the same resolution (32 $\times$ 32)  with provided PIE image data in `./PIE/self`.


## Reproducibility

Commands for applying these methods:

### PCA
```
python main.py --model PCA
```

Corresponding results of PCA will be stored in the folder `./results`:

```
├── results
│   ├── pca_2d.png
│   ├── pca_3d.png
│   ├── pca_classification.csv
│   ├── pca_eigenfaces.png
```

### LDA
```
python main.py --model LDA
```

Corresponding results of LDA will be stored in the folder `./results`:
```
├── results
│   ├── lda_2d.png
│   ├── lda_3d.png
│   ├── lda_classification.csv
```
### SVM

```
python main.py --model SVM
```

Corresponding results of SVM will be stored in the folder `./results`:

```
├── results
│   ├── svm_classification.csv
```
### CNN

```
python main.py --model GMM
```

Corresponding results of CNN will be stored in the folder `./results`:
```
├── results
│   ├── cnn.pth
│   ├── cnn_fig.pdf
│   ├── cnn_loss.npy
│   ├── cnn_acc.npy
```