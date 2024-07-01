# Multi-GPU Implementation for Cupy, with KNN Model and MNIST Dataset

After going through preliminary steps for framework and workload exploration, as well as simple mathematical and machine learning model implementation exploration, in this step the exploration for Multi-GPU utilization is conducted. 

The workload in this step is different than previous parts. MNIST dataset is used, through OpenML database, titled "mnist_784", version = 1.
```python
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
```
Moreover, to conduct multiclass classification, the Machine Learning model is changed to K-Nearest Neighbour. 

**KNN Pseudocode (Initial):**

    Load the training data.
    Predict a class value for new data:
        Calculate distance(X, Xi) from i=1,2,3,….,n.
        where X= new data point, Xi= training data, distance as per chosen distance metric, in here Euclidean Distance is used.
        Sort these distances in increasing order with corresponding train data.
        From this sorted list, select the top ‘K’ rows.
        From these K rows, each label gets a weighted vote according to how close they are to new data, to predict the new data's label.


#### Splitting Data
The splitting data process is the only part of the code that uses non-Cupy library, to acquire a well-shuffled and split data to be used in the model. Indices for the split is acquired using _sklearn_.

```python
from sklearn.model_selection import train_test_split
def get_train_test_indices(n_samples, test_size=0.25, random_state=None):
    indices = np.arange(n_samples)
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    return train_indices, test_indices
```

Hence, despite having the indices array as non-Cupy, the split arrays for train and test groups are Cupy arrays.

#### Pure Cupy vs Euclidean Distance RawKernel on Single GPU Output, KNN model with MNIST dataset
**Pure Cupy**
```sh
$ python mnist_knn.py
Elapsed time: 28.71679949760437 seconds
Accuracy: 0.9721714285714286
```

**With Euclidean Distance as RawKernel**
```sh
$ python rawEucOnly.py
Elapsed time: 13.79976511001587 seconds
0.9721714285714286
```

#### Implementing Multi-GPU Commands
When Multi-GPU commands are used on the Euclidean Distance Raw Kernel is used with the KNN Pseudocode marked as Initial above, there is no overlapping between the execution of the 2 GPUs. In contrast, GPU 1 finishes executing on half of the dataset, and after GPU 2 starts working on the rest of the data. This can be observed in the following image from Nsight Systems:


This case happens despite actively sending the split dataset into separate GPUs, as well as using Streams from concurrency structures. Hence, at this point couple of actions are considered. 
1. To change the pseudocode to get through Euclidean Distance calculation at once, with no classification calculation coming in between to observe whether parallel execution is blocked due to Host being occupied with many calls from GPU 1.
     a. For this purpose the following Pseudocode is considered:
2. Have all of the data reside in Host (while data is acquired and split) and only send to 2 separate GPUs when they need to get processed.
3. TBC! 
