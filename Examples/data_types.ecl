/*

This file contians the record types for use in the distributed deep learning runtime.
Includes types for the MNIST data set for use in the examples.

*/

modelRec := RECORD
 STRING modelConfig;
 STRING startingWeights;
 STRING weightDelta;
 REAL timer;
END;

evaluationResultsRec := RECORD
STRING performanceMetric;
STRING timeToTrain;
STRING numPartitions;
END;


mnist_data_type := RECORD
 INTEGER1 label;
 DATA784 image;
END;

nodeMarked := RECORD(mnist_data_type)
  UNSIGNED4 node;
END;

subGroup := RECORD(nodeMarked)
	UNSIGNED grp;
END;


EXPORT data_types := MODULE

EXPORT modelRec := modelRec;
EXPORT mnist := mnist_data_type;
	EXPORT training := mnist_data_type; //data type of the training data
EXPORT nodeMarked := nodeMarked;
EXPORT subGroup := subGroup;
EXPORT evaluationResultsRec := evaluationResultsRec;

END;

