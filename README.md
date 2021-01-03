# regression

Task: build a regression model that works best for multiple dataset.
(see hw4_HyunJaePi.pdf for more detail)

Work flow:
1. test a base model performance -- w/o cleanup(e.g. outliers) and regularization
2. remove outliers and redundant features -- tested standardization(z-score), boxplot & 1.5IQR, SVM, and isolation forest
3. regularization -- tested Ridge(L2) and Lasso(L1)
4. optimize hyper-parameters
5. make a prediction on test datasets
