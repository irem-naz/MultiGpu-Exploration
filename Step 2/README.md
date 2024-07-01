# Logistic Regression and CuPy
As the goal of this project is to have control over kernel launch configurations while implementing ML models, the first candidate that allows such GPU acceleration and control is CuPy, which is the initial framework that is explored in this project as well. 

In this step, logistic regression is implemented in Cupy and Numpy, with increasing data sizes to observe the changes in the duration. Numpy is awefully stagnant and linearly increases in duration while CuPy has a more stable nearly non-changing time across increasing data sizes.

<p align="center">
  <img width="460" height="300" src="./elapsed_time_vs_data_size.png">
</p>
