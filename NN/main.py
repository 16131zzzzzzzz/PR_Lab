'''
env: python3.7.5
requirements: mindspore, numpy, matplotlib

usage: python main.py --device_target CPU/GPU/Ascend --mode load/train --net resnet50/lenet5
--device_target  according to the mindspore you use
--mode  use load when you want to skip the train step
--net  choose from the two nets.

Resnet takes more time to train while its accuracy is more than 70%
Accuracy of the Lenet fluctuate between 50% and 60% while it takes only a few minutes to train
'''

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore import context, Tensor, Model, load_checkpoint, load_param_into_net
from mindspore.train.callback import Callback, ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.dataset.vision import Inter
from mindspore.common.initializer import Normal
from mindspore.nn import Accuracy
import mindspore.ops as ops
from resnet50 import resnet50
from lenet5 import lenet5


# some changeable parameters
model_path = "./model"  # where to save model
mnist_path ="./cifar-10-binary/cifar-10-batches-bin"  # where the dataset is


# some input arguments
parser = argparse.ArgumentParser(description='PR_Lab3')
parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])
parser.add_argument('--mode', type=str, default="load", choices=['load','train'])
parser.add_argument('--net', type=str, default="resnet50", choices=['resnet50','lenet5'])

args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
mode = args.mode
net_arg = args.net

#definition of callbacks
class StepLossAccInfo(Callback):
    def __init__(self, model, eval_dataset, steps_loss, steps_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.steps_loss = steps_loss
        self.steps_eval = steps_eval

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = (cur_epoch-1)*1875 + cb_params.cur_step_num
        self.steps_loss["loss_value"].append(str(cb_params.net_outputs))
        self.steps_loss["step"].append(str(cur_step))
        if cur_step % 125 == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.steps_eval["step"].append(cur_step)
            self.steps_eval["acc"].append(acc["Accuracy"])

class CrossEntropyLoss(nn.Cell):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()
        self.mean = ops.ReduceMean()
        self.one_hot = ops.OneHot()
        self.one = Tensor(1.0, mstype.float32)
        self.zero = Tensor(0.0, mstype.float32)

    def construct(self, logits, label):
        label = self.one_hot(label, ops.shape(logits)[1], self.one, self.zero)
        loss_func = self.cross_entropy(logits, label)[0]
        loss_func = self.mean(loss_func, (-1,))
        return loss_func


# create a dataset of optional size
def create_dataset(sample_num, data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    # load dataset
    # mnist_ds = ds.MnistDataset(data_path)
    mnist_ds = ds.Cifar10Dataset(data_path, num_samples=sample_num, shuffle=True)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define resizers
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # define maps and enhance dataset
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # shuffle and batch
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)

    return mnist_ds


# the trainning and testing functions
steps_loss = {"step": [], "loss_value": []}
steps_eval = {"step": [], "acc": []}
def train_net(args, model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    # load dataset
    ds_train = create_dataset(50000, os.path.join(data_path, "train"), 32, repeat_size)
    ds_eval = create_dataset(10000, os.path.join(data_path, "test"))

    # save the network model and parameters for subsequence fine-tuning
    config_ck = CheckpointConfig(save_checkpoint_steps=375, keep_checkpoint_max=16)
    # group layers into an object with training and evaluation features
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_"+str(net_arg), directory=model_path, config=config_ck)

    # collect the steps,loss and accuracy information
    step_loss_acc_info = StepLossAccInfo(model ,ds_eval, steps_loss, steps_eval)

    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(per_print_times=1), step_loss_acc_info], dataset_sink_mode=False)

def test_net(network, model, data_path):
    ds_eval = create_dataset(10000, os.path.join(data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))


if __name__ == "__main__":
    # definition of the net
    net = resnet50(batch_size=32, num_classes=10) if net_arg == 'resnet50' else lenet5()
    # definition of loss function
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # definition of optimizer
    net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
    # model savers
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

    if mode == 'train':
        # train and evaluate the model in train mode
        train_epoch = 10
        dataset_size = 1
        model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
        train_net(args, model, train_epoch, mnist_path, dataset_size, ckpoint, False)
        test_net(net, model, mnist_path)

        # draw the step_loss chart
        steps = steps_loss["step"]
        loss_value = steps_loss["loss_value"]
        steps = list(map(int, steps))
        loss_value = list(map(float, loss_value))
        plt.plot(steps, loss_value, color="red")
        plt.xlabel("Steps")
        plt.ylabel("Loss_value")
        plt.title("Change chart of model loss value")
        plt.show()

    # load the model and evaluate, or you can just directly evaluate after training. The model loading step exists when you want to skip the trainning process and see the picture below.
    if net_arg == 'resnet50':
        load_checkpoint("checkpoint_resnet50_3-10_1562.ckpt", net=net)
    else:
        load_checkpoint("checkpoint_lenet5-10_1562.ckpt", net=net)
    net_loss = CrossEntropyLoss()
    model = Model(net, net_loss, metrics={"Accuracy": Accuracy()})
    ds_eval = create_dataset(100, os.path.join(mnist_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))

    # randomly choose 32 pictures from the testset and predict them. The blue ones are correctly classified while the red means incorrect.
    ds = create_dataset(32, os.path.join(mnist_path, "test"))
    ds_test = ds.create_dict_iterator()
    data = next(ds_test)

    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()

    output = model.predict(Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)

    images = np.add(images,1 * 0.1307 / 0.3081)
    images = np.multiply(images, 0.3081)

    index = 1
    for i in range(len(labels)):
        plt.subplot(4, 8, i+1)
        color = 'blue' if pred[i] == labels[i] else 'red'
        plt.title("pre:{}".format(pred[i]), color=color)
        img = np.squeeze(images[i]).transpose((1,2,0))
        plt.imshow(img)
        plt.axis("off")
        if color == 'red':
            index = 0
            print("Row {}, column {} is incorrectly identified as {}, the correct value should be {}".format(int(i/8)+1, i%8+1, pred[i], labels[i]))
    if index:
        print("All the figures in this group are predicted correctly!")
    plt.show()