IMPORT Python;
IMPORT $;
IMPORT $.data_types as recType;


recType.modelRec pythonFunction0(recType.modelRec modelInput, DATASET(recType.training) dataInput, UNSIGNED4 nodeId, INTEGER epoch, BOOLEAN lastEpoch) := EMBED(Python)
from uuid import getnode as get_mac
import pandas as pd
import numpy as np
import cPickle as pickle
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import utils
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K
K.clear_session()

model = Sequential.from_config(pickle.loads(modelInput[0]))
model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False), metrics=['accuracy'])
model.set_weights(pickle.loads(modelInput[1]))

#temp folder gets cleared before each workunit executes
ioname = 'distributed_worker_temp_node_' + str(nodeId) + '.h5'
if(epoch == 0):
	allData = [n for n in dataInput]
	df = pd.DataFrame(allData, columns=n._fields)
	df.to_hdf(ioname, 'train', mode = 'w', append=False) #make this true if the weight updates messes up NN performace
else:
	df = pd.read_hdf(ioname, 'train')

x_dat = df.ix[:, 1]
x_dat = np.asarray([np.frombuffer(x, dtype='B', count=-1, offset=0) for x in x_dat])
x_dat = x_dat.reshape(x_dat.shape[0], 784)
x_dat = x_dat.astype('float32')
x_dat /= 255

y_dat = df.ix[:, 0:1].values.astype('int32')
y_dat = utils.to_categorical(y_dat, 10)

model.fit(x_dat, y_dat, batch_size=128, epochs=100, shuffle=True, verbose=0, validation_split=0.2)
w0 = np.asarray(pickle.loads(modelInput[1]))
w1 = np.asarray(model.get_weights())
dw = w1 - w0

#just return the new weights, not the deltas
return (pickle.dumps(model.get_config(),0), modelInput[1], pickle.dumps(dw,0), modelInput[3])
ENDEMBED;

recType.modelRec pythonFunction(recType.modelRec modelInput, UNSIGNED4 nodeId, INTEGER epoch, BOOLEAN lastEpoch) := EMBED(Python)
from uuid import getnode as get_mac
import pandas as pd
import numpy as np
import cPickle as pickle
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import utils
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K
K.clear_session()

model = Sequential.from_config(pickle.loads(modelInput[0]))
model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False), metrics=['accuracy'])
model.set_weights(pickle.loads(modelInput[1]))

#temp folder gets cleared before each workunit executes
ioname = 'distributed_worker_temp_node_' + str(nodeId) + '.h5'
df = pd.read_hdf(ioname, 'train')

x_dat = df.ix[:, 1]
x_dat = np.asarray([np.frombuffer(x, dtype='B', count=-1, offset=0) for x in x_dat])
x_dat = x_dat.reshape(x_dat.shape[0], 784)
x_dat = x_dat.astype('float32')
x_dat /= 255

y_dat = df.ix[:, 0:1].values.astype('int32')
y_dat = utils.to_categorical(y_dat, 10)

model.fit(x_dat, y_dat, batch_size=128, epochs=1, shuffle=True, verbose=0, validation_split=0.2)
w0 = np.asarray(pickle.loads(modelInput[1]))
w1 = np.asarray(model.get_weights())
dw = w1 - w0

#just return the new weights, not the deltas
return (pickle.dumps(model.get_config(),0), modelInput[1], pickle.dumps(dw,0), modelInput[3])
ENDEMBED;

recType.modelRec runPy0(recType.modelRec model, DATASET(recType.nodeMarked) ds, UNSIGNED4 nodeId, INTEGER epoch, BOOLEAN lastEpoch) := TRANSFORM
  SELF := pythonFunction0(model, ds, nodeId, epoch, lastEpoch); //the python worker that needs distributing
END;

recType.modelRec runPy(recType.modelRec model, UNSIGNED4 nodeId, INTEGER epoch, BOOLEAN lastEpoch) := TRANSFORM
  SELF := pythonFunction(model, nodeId, epoch, lastEpoch); //the python worker that needs distributing
END;


EXPORT MNIST_train := MODULE
EXPORT init(recType.modelRec model, DATASET(recType.nodeMarked) ds, UNSIGNED4 nodeId, INTEGER epoch, BOOLEAN lastEpoch) := runPy0(model, ds, nodeId, epoch, lastEpoch);
EXPORT train(recType.modelRec model, UNSIGNED4 nodeId, INTEGER epoch, BOOLEAN lastEpoch) := runPy(model, nodeId, epoch, lastEpoch);
END;
