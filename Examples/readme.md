# Examples

This example trains a multi-layer perceptron (MLP) on the MNIST dataset in a distributed fashion. 
The trained model is a handwritten digit classifier. To read more about the MNIST data set and
what you can do with it go [here](http://yann.lecun.com/exdb/mnist/).

This example was run on a single node and a 4 node cluster where each node is on its own logical computer, 
and each logical computer has 1 CPU (1 core) and 3.5 gigs of memory. Depending on your hardware 
you will need to modify the parameter [here](https://github.com/robertken/Distributed-Deep-Learning/blob/ad2b7b5ef53508322ac14f256a7dfef8a4f44267/Examples/MNIST_MLP.ecl#L47). This number should be maximized and is there to partition 
the data for the pyembed memory constraints.

 * First spray the prepared data, found in the MNIST directory onto your HPCC System
 	* Fixed record length of 785
 	* (the original MNIST data is included in MNIST_raw if for completeness)
  * Second simply run [this](MNIST_mlp.ecl)

