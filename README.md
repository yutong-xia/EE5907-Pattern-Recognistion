# EE5907 CA2

Coursework for EE5907 Pattern Recognition CA2, AY 2022/2023 Semester 1.

The aim of this project is to construct a face recognition system via Principal Component Analysis (PCA), Linear Discriminative Analysis (LDA), Support Vector Machine (SVM) and Gaussian Mixture Model (GMM). Firstly, PCA is used to perform data dimensionality reduction and visualization, in order to understand underlying data distribution. Then two classification methods, LDA and SVM, are used to classify the face images. GMM is used to group the face images

The detail setting of the experiments can be found in the corresponding report, which is accessible [here]().

## Requirements


- Python 3.6
- matplotlib == 3.5.2
- numpy == 1.21.5
- pandas == 1.3.5
- libsvm == 3.23.0.4


Dependencies can be installed using the following command:
```
pip install -r requirements.txt
```

## Data
The dataset used in this project is [CMU PIE dataset](https://data.nvision2.eecs.yorku.ca/PIE_dataset/) and the face photos taken by the students themselves. There are in total of 68 different subjects and I selected the first 25 out of them. For each chosen subject, 70% for training and the remaining 30% for testing. Besides the CMU PIE images, there are 10 selfie photos used as samples after being converted to grey-scale images and resized into the same resolution (32 X 32) in './PIE/self'.


## Reproducibility

Commands for applying these methods:

### PCA
```
python main.py --model PCA
```

Corresponding results of PCA will be stored in the folder './results':

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

Corresponding results of LDA will be stored in the folder './results':
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

Corresponding results of SVM will be stored in the folder './results':

```
├── results
│   ├── svm_classification.csv
```
### GMM

```
python main.py --model GMM
```

Corresponding results of GMM will be stored in the folder './results':

```
├── results
│   ├── gmm_classification.csv
```