import caffe

from caffe.proto import caffe_pb2
from pylab import *
from caffe import layers as L
from caffe import params as P

dataset_path = "/home/tictok/repository/cifar10/"
prototxt_path = "data/"
train_net_file = "auto_train00.prototxt"
test_net_file = "auto_test00.prototxt"
solver_file = "auto_solver.prototxt"


# net file generate
def net(datafile, mean_file, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(source=datafile, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                             transform_param=dict(scale=1.0 / 255.0, mean_file=mean_file))
    n.ip1 = L.InnerProduct(n.data, num_output=200, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    n.accu = L.Accuracy(n.ip2, n.label, include={'phase': caffe.TEST})
    return n.to_proto()


with open(prototxt_path + train_net_file, 'w') as f:
    f.write(str(net(dataset_path + 'cifar10_train_lmdb', dataset_path + 'mean.binaryproto', 200)))
with open(prototxt_path + test_net_file, 'w') as f:
    f.write(str(net(dataset_path + 'cifar10_test_lmdb', dataset_path + 'mean.binaryproto', 100)))

# solver file generate
s = caffe_pb2.SolverParameter()
s.train_net = prototxt_path + train_net_file
s.test_net.append(prototxt_path + test_net_file)
s.test_interval = 500
s.test_iter.append(100)
s.display = 500
s.max_iter = 10000
s.weight_decay = 0.005
s.base_lr = 0.1
s.lr_policy = "step"
s.gamma = 0.1
s.stepsize = 5000
s.solver_mode = caffe_pb2.SolverParameter.CPU

with open(prototxt_path + solver_file, 'w') as f:
    f.write(str(s))

# iter to calculate the model weight
solver = caffe.get_solver(prototxt_path + solver_file)
niter = 2001
train_loss = zeros(niter)
test_accu = zeros(niter)
for it in range(niter):
    solver.step(1)
    train_loss[it] = solver.net.blobs['loss'].data
    test_accu[it] = solver.test_nets[0].blobs['accu'].data

# output graph
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(arange(niter), test_accu, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
_.savefig(prototxt_path + "converge00.png")
