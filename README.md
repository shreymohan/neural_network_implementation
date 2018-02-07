# Neural Network Implementation

## Neural Network for Regression

* Here a simple neural network with one hidden layer is implemented for the purpose of regression.
* Thus, the cost function used is the simple mean squared error which is minimized using gradient descent algorithm.
* It is first trained using a single sample with 2 outputs and then tested with a sample to verify.
* To train/test using multiple samples, an additional loop is required inside both the functions which is quite straight forward.

## Neural Network for Classification

* Here a simple neural network with one hidden layer is implemented for the purpose of classification.
* Thus, the cost function used is the cross-entropy function which is again minimized using gradient descent algorithm.
* The train/test set is prepared from the Iris dataset, where we only take 2 kinds of flowers- versicolor and virginica removing the setosa kind, thus performing binary classification.
* Leave one out analysis is carried out where a false positive gives an error 1 whereas a true positive gives error 0.
* Total error is computed as the ratio of error by total number of samples which comes out to be 7%.
