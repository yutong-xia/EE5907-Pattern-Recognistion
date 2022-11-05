# EE5907 CA2

Coursework for EE5907 Pattern Recognition CA2, AY 2022/2023 Semester 1.

The aim of this project is to construct a face recognition system via Principal Component Analysis (PCA), Linear Discriminative Analysis (LDA), Support Vector Machine (SVM) and Gaussian Mixture Model (GMM). Firstly, PCA is used to perform data dimensionality reduction and visualization, in order to understand underlying data distribution. Then two classification methods, LDA and SVM, are used to classify the face images. GMM is used to group the face images

The report of this project is accessible [here]().

## Usage

Commands for applying these methods:

### PCA

`python main.py --model PCA`

Corresponding results of PCA will be stored in the folder './results':

```
├── results
│   ├── pca_2d.png
│   ├── pca_3d.png
│   ├── pca_classification.csv
│   ├── pca_eigenfaces.png
```

### LDA

`python main.py --model LDA`

Corresponding results of LDA will be stored in the folder './results':
```
├── results
│   ├── lda_2d.png
│   ├── lda_3d.png
│   ├── lda_classification.csv
```
### SVM

`python main.py --model SVM`

Corresponding results of SVM will be stored in the folder './results':

```
├── results
│   ├── svm_classification.csv
```
### GMM

`python main.py --model GMM`

Corresponding results of GMM will be stored in the folder './results':

```
├── results
│   ├── gmm_classification.csv
```