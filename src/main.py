import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from reader import image_input_reader
from algorithm.onlinebatch import *
from algorithm.activebias import *
from algorithm.recencybias import *

def main():
    print("-------------------------------------------------------------------------")
    print("This Tutorial is to train DenseNet-25-12 in tensorflow-gpu environment.")
    print("\nDescription -----------------------------------------------------------")
    print("Supporting datasets: {CIFAR-10}. *** all the data sets will be added in the next version. ***")
    print("Available algorithms: {Online Batch, Active Bias, Recency Bias}")
    print("For Training, we used the same training schedule:")
    print("\tbatch = 128, warm-up = 10")
    print("\tFor CIFAR-10, learning rate = 0.1 (decayed 50% and 75% of total number of epochs), epochs = 100")
    print("------------------------------------------------------------------------")
    if len(sys.argv) != 6:
        print("------------------------------------------------------------------------")
        print("Run Cmd: python main.py  gpu_id  data  model_name  method_name  optimizer  log_dir")
        print("\nParamter description")
        print("gpu_id: gpu number which you want to use")
        print("data : {CIFAR-10}")
        print("method_name: {Online Batch, Active Bias, Recency Bias}")
        print("optimizer: {sgd, momentum}")
        print("log_dir: log directory to save mini-batch loss/acc, training loss/acc and test loss/acc")
        print("------------------------------------------------------------------------")
        sys.exit(-1)

    # For user parameters
    gpu_id = int(sys.argv[1])
    data = sys.argv[2]
    method_name = sys.argv[3]
    optimizer = sys.argv[4]
    log_dir = sys.argv[5]

    datapath = str(Path(os.path.dirname((os.path.abspath(__file__)))).parent) + "/src/dataset/" + data
    if os.path.exists(datapath):
        print("Dataset exists in ", datapath)
    else:
        print("Dataset doen't exist in ", datapath, ", please downloads and locates the data.")
        sys.exit(-1)

    # Default Training Configuration
    queue_size = 10
    warm_up = 10
    batch_size = 128
    epochs = [10, 100]
    se_s = [100.0, 1.0]

    # CIFAR-10
    num_train_images = 50000
    num_test_images = 10000
    total_epochs = 100
    model_name = "DenseNet-25-12"

    input_reader = image_input_reader.ImageReader(data, datapath, 5, num_train_images, num_test_images, 32, 32, 3, 10)
    lr_values = [0.1, 0.01, 0.001]
    lr_boundaries = [int(np.ceil(num_train_images / batch_size) * 50), int(np.ceil(num_train_images / batch_size) * 75)]

    if method_name == "Online Batch":
        online_batch(gpu_id, input_reader, model_name, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, warm_up, se_s,
                     epochs, log_dir=log_dir)

    elif method_name == "Active Bias":
        active_bias(gpu_id, input_reader, model_name, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, warm_up, log_dir=log_dir)

    elif method_name == "Recency Bias":
        recency_bias(gpu_id, input_reader, model_name, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, warm_up, queue_size,
                    se_s, epochs, log_dir=log_dir)


if __name__ == '__main__':
    print(sys.argv)
    main()
