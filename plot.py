import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

def make_graph():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default="./models/cifar10/")
    parser.add_argument('--save_dir', type=str, default="./plot/")
    parser.add_argument('--save_name', type=str, default="sgd_cifar10")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    assert os.path.exists(args.log_dir),\
        "Log path dones't exists, check it: %s" % args.log_dir

    train_log = os.path.join(args.log_dir, "train_log.txt")
    test_log = os.path.join(args.log_dir, "test_log.txt")
    # [epoch, total_epoch, batch, batch_num_per_epoch, loss]
    train_data = extract_data(train_log)
    # [epoch, total_epoch, accuracy]
    test_data = extract_data(test_log)
    
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(20,20))
    ax1.plot(train_data[:,0],train_data[:,-1])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Cross-Entropy Loss")
    ax1.set_yscale('log')
    
    ax2.plot(test_data[:,0], 100. - test_data[:,-1])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Error")
    ax2.set_yscale('log')
    
    image_name = os.path.join(args.save_dir, args.save_name+"_loss_error.jpg")
    plt.savefig(image_name)
    print "Create ",image_name

def extract_data(txt_file):
    data = []
    with open(txt_file,'r') as f:
        for line in f.readlines():
            string = line.strip()
            nums = re.findall(r"\d+\.?\d*",string)
            nums = map(float, nums)
            data.append(nums)
    return np.array(data).reshape((len(data),-1)).astype(np.float32)

if __name__ == "__main__":
    make_graph()
