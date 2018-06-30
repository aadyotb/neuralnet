# NeuralNet

This is a library that can be used to implement a convolutional neural network for classification tasks. The implementation is optimized to run on the GPU and uses CUDA, cuBLAS, and cuDNN to perform computations related to neural network inference. The `neuralnet` executable trains a convolutional neural network on the MNIST dataset of handwritten digits, and evaluates its performance on the training subset.

A high level interface (similar to that of [Keras](https://keras.io/)) for constructing a linear feed-forward network is implemented in `model.hpp` and `model.cpp`. To add a `Layer`, a user should call the `add()` method of the `Model` and supply as arguments a string indicating the type of layer desired, and an array or vector of integers that specify parameters relevant to the `Layer`. Note that the shape of a single input minibatch (__N__umber of examples in minibatch, __C__hannels per example, __H__eight of example, __W__idth of example) must be given to the Model's constructor as NCHW.

Once a Model is constructed, a user can call the Model's `train()` method with a set of training data of the specified input shape, a set of corresponding labels with the specified output shape, the number of training examples being used, and the desired mini-batch size. Similar functions exist to return the Model's predictions (`predict()`) on a test set and to evaluate the Model's performance on a test set, given a set of ground truth labels (`evaluate()`).

The `Model` class provides an abstraction layer over a vector of generic `Layer` objects. Each `Layer` (implementations in `layers.hpp` and `layers.cpp`) is a subclass of an abstract Layer superclass and implements its own methods for a forward and backward pass on a minibatch of data. Notably, loss layers are implemented as subclasses of an abstract `Loss` class, which is in turn a subclass of the abstract `Layer` class. To add a new Layer, one need only implement an appropriate class in `layers.hpp` and `layers.cpp` (see below for more details) and modify
the Model's `add()` method to parse arguments specifying the desired layer and its parameters.

Each Layer maintains device pointers to its internal representation of (1) its input and output minibatches, (2) the gradients of the loss function with respect to them, and (3) the current weights and biases associated with that layer, and (4) the gradients of the weights and biases. Notably, the pointer to a layer's input minibatch (and its gradient) is the same as the pointer to the previous layer's output minibatch (and its gradient). See `layers.hpp` for more details.

In its constructor, every subclass of the Layer base class must (1) set the shape of its output minibatch of data, (2) set descriptors for other relevant quantities/characteristics of the layer (e.g. shape of weight matrix, descriptor of activation, descriptor of convolution operation, etc.) and (3) allocate buffers for weights, biases, and output minibatches.

Each Layer subclass must also specify how it interacts with a feedforward neural net by overriding the pure virtual methods `Layer::forward_pass()` (to do its computation and pass it to the next layer) and `Layer::backward_pass()` (to compute gradients and propagate them backwards to the previous layer). This design makes it simple to add support for new kinds layers.

Future releases will modernize the C++. In particular, some of the virtual methods will likely be replaced with template parameters, and less brittle memory structures (like STL smart pointers) will be introduced.
