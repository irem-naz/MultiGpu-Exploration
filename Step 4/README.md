# Multi-GPU Implementation for Cupy, with KNN Model and MNIST Dataset

After going through preliminary steps for framework and workload exploration, as well as simple mathematical and machine learning model implementation exploration, in this step the exploration for Multi-GPU utilization is conducted. 

The workload in this step is different than previous parts. MNIST dataset is used, through OpenML database, titled "mnist_784", version = 1.
```py
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
```
Moreover, to conduct multiclass classification, the Machine Learning model is changed to K-Nearest Neighbour. 

**KNN Pseudocode:**
    Load the training data.
    Predict a class value for new data:
        Calculate distance(X, Xi) from i=1,2,3,….,n.
        where X= new data point, Xi= training data, distance as per chosen distance metric, in here Euclidean Distance is used.
        Sort these distances in increasing order with corresponding train data.
        From this sorted list, select the top ‘K’ rows.
        From these K rows, each label gets a weighted vote according to how close they are to new data, to predict the new data's label.


