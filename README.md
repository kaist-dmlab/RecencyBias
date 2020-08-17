## Carpe Diem: Seize the Samples Uncertain "at the Moment" for Adaptive Batch Selection

> __Publication__ </br>
> Song, H., Sundong, K., Kim, M., and Lee, J., "Carpe Diem: Seize the Samples Uncertain "at the Moment" for Adaptive Batch Selection," *In Proceedings of the 29th ACM International Conference on Information and Knowlege Management (CIKM)*, SOctober 2020, VirtualEvent, Ireland.

##  1. Requirement 
- Python 3
- tensorflow-gpu 2+
- tensorpack libracy //use "pip install tensorpack"

##  2. Description
- This Tutorial is to train DenseNet-25-12 in tensorflow-gpu environment.
- Please do not change the structure of directories:
	- Folder **_src_** provides all the code for evaluation with compared methods.
  	- Folder **_src/dataset_** contains a benchmark dataset (CIFAR-10). Due to the lack of space, the other data will be uploaded soon. Moreover, **_.bin_** format is used for the synthetic data because they can be loaded at once in main memory.

### 3. Tutorial for Evaluation.
- Training Configuration
	```python
	# All the hyperparameters of baseline methods were set to the same value described in our paper.
	# Source code provides a tutorial to train DensNet or ResNet using a simple command.
	```
	
- Necessary Parameters
	```python
	- 'gpu_id': gpu number which you want to use (only support single gpu).
	- 'data_name': {MNIST, CIFAR-10}. # others will be supported later
	- 'method_name': {Online Batch, Active Bias, Recency Bias}.
	- 'optimizer': {sgd, momentum}
	- 'log_dir': log directory to save (1) mini-batch loss/error, (2) training loss/error, and (3) test loss/error.
	```
- Running Command
	```python
	python main.py 'gpu_id' 'data_name' 'method_name' 'optimizer' 'log_dir'
	
    # e.g., train on CIFAR-10 using RecencyBias with sgd.
    # python main.py '0' 'CIFAR-10' 'Recency Bias' 'sgd' 'log-cifar-10'
	```	
- Detail of Log File
	```python
	# convergence_log.csv
    # format: epoch, elapsed time, lr, mini-batch loss, mini-batch error, trainng loss, 
    #         training error, test loss, test error
	```	
