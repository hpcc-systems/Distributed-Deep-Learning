IMPORT Python;
IMPORT $.data_types as recType;


EXPORT recType.modelRec combineWeights(DATASET(recType.modelRec) modelInput) := EMBED(Python)
import cPickle as pickle
import numpy as np

#takes in model config, epoch starting weights, and weight deltas from workers
#return model config, sum deltas with starting and return as the new starting
#return empty string for the weight deltas

input = [(rec.modelconfig, rec.startingweights, rec.weightdelta, rec.timer) for rec in modelInput]

weights = [pickle.loads(rec[2]) for rec in input]
numberOfNodes = len(input)
summed = np.sum(weights, axis=0)
summed /= numberOfNodes

newW = pickle.loads(input[0][1])
newW += summed

return (input[0][0], pickle.dumps(newW, 0), str(numberOfNodes), input[0][3])
ENDEMBED;

