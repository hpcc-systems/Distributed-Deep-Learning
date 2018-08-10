IMPORT Python, STD, $;
IMPORT $.data_types as recType;
IMPORT $.MNIST_train as worker;
#option('outputLimit',2000);

//First distribute the training data. Testing data is used for model evaluation and is completed
//on a single node, i.e. no need to distribute the testing data.
trainingData := DISTRIBUTE(CHOOSEN(DATASET('~mnist::train', recType.mnist, THOR), 60000));
testingData := CHOOSEN(DATASET('~mnist::test', recType.mnist, THOR), 10000);

//Next, define a neural network, loss functions, and optimizer. This example uses Keras
//with a TensorFlow backend. See Keras.io for Keras documentation.
recType.modelRec createModel() := EMBED(Python)
		import time
		timer0 = time.time()
		from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
		from tensorflow.python.keras.models import Sequential, Model
		from tensorflow.python.keras import optimizers, losses

		from tensorflow.python.keras import backend as K
		K.clear_session() #required for use on THOR, hThor does not need it

		import cPickle as pickle

		num_classes = 10
		
		model = Sequential()
		model.add(Dense(100, activation='relu', input_shape=(784,)))
		model.add(Dropout(0.2))
		model.add(Dense(num_classes, activation='softmax'))

		model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
              metrics=['accuracy'])

		model_weights = model.get_weights()
		model_config = model.get_config()

		return (pickle.dumps(model_config, 0), pickle.dumps(model_weights, 0), '', timer0)
ENDEMBED;

//---- Partitions data and pyembed code for localized training -------------------------------
Marked := PROJECT(trainingData, TRANSFORM(recType.nodeMarked, SELF.node := Std.system.Thorlib.Node()+1, SELF:=LEFT), LOCAL);
GroupedData := GROUP(Marked, node, LOCAL);

//The "grp" attribute is dependant on the underlying hardware (memory) and the size of each row in the training dataset
subMarked := PROJECT(GroupedData, TRANSFORM(recType.subGroup, SELF.grp:=COUNTER DIV 300000, SELF:=LEFT), LOCAL);
subGroup := GROUP(subMarked, grp, LOCAL) : PERSIST('Sub group persist');

DATASET(recType.modelRec) epochStepInit(recType.modelRec model, INTEGER epoch):=function
     weightUpdates := ROLLUP(subGroup, GROUP, worker.init(model, ROWS(LEFT), LEFT.grp, 0, FALSE));
     return DATASET($.combineWeights(weightUpdates));
end;

DATASET(recType.modelRec) epochStep(recType.modelRec model, INTEGER epoch):=function
     weightUpdates := ROLLUP(subGroup, GROUP, worker.train(model, LEFT.grp, epoch, FALSE));
     return DATASET($.combineWeights(weightUpdates));
end;
//--------------------------------------------------------------------------------------------

//create a NN model
myModel := createModel();

//initilize the workers with training data and NN model
//This will complete 1 training epoch as well.
trainInit := epochStepInit(myModel, 0); 

//Train for some number of epochs on the training data.
//Will return a trained model.
numberOfTrainingEpochs := 1;
trainedModel := loop(trainInit, numberOfTrainingEpochs, epochStep(rows(left)[1], COUNTER));

//Evaluate the NN model on unseen testing data
//Returns training time, performanc metric, and number of training partitions used
modelEvaluation := $.evaluate_MNIST_mlp(trainedModel[1], testingData);
OUTPUT(modelEvaluation);

