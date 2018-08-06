IMPORT Python;
IMPORT $.data_types as recType;

EXPORT recType.evaluationResultsRec evaluate_MNIST_mlp(recType.modelRec modelInput, DATASET(recType.training) dataInput) := EMBED(Python)
import time
timer1 = time.time()
timer0 = modelInput[3]
import cPickle as pickle
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras import utils
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses

from tensorflow.python.keras import backend as K

K.clear_session()  # this is because the python interpreter is kept alive for

model = Sequential.from_config(pickle.loads(modelInput[0]))
model.compile(loss=losses.categorical_crossentropy,optimizer=optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),metrics=['accuracy'])
model.set_weights(pickle.loads(modelInput[1]))

allData = [n for n in dataInput]
df = pd.DataFrame(allData, columns=n._fields)

d = np.array(df[:])

x_dat = df.ix[:, 1]
x_dat = np.asarray([np.frombuffer(x, dtype='B', count=-1, offset=0) for x in x_dat])
x_dat = x_dat.reshape(x_dat.shape[0], 784)
x_dat = x_dat.astype('float32')
x_dat /= 255

y_dat = df.ix[:, 0:1].values.astype('int32')
y_dat = utils.to_categorical(y_dat, 10)

history = model.evaluate(x_dat, y_dat, batch_size=200, verbose=0)

return (str(history), str(timer1-timer0), str(modelInput[2]))
ENDEMBED;