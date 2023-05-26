mkdir -p MNIST
wget -O ./MNIST/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -O ./MNIST/train-labels-idx1-ubyte.gz  http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -O ./MNIST/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -O ./MNIST/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

mkdir -p MNIST/train
mkdir -p MNIST/test