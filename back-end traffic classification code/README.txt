#python file
(1)Agg_RFmodelTrain_TCP.py、Agg_RFmodelTrain_UDP.py:  
  The offline construction of the classification model includes the clustering process of the original feature data and the training process of the Rf model.
(2)RFmodelTrainClass.py:
  Implementation of Rf model training, called by Agg_RFmodelTrain_TCP, Agg_RFmodelTrain_UDP.
(3)RFtest_Concept.py:
  Make predictions for RF models.
(4)BigSmallLabel.py:
  Implement the statistics of the elephant class. Modules called by other functions.

### How to use
(1) Run Agg_RFmodelTrain_TCP.py, Agg_RFmodelTrain_UDP.py to complete the construction of the classification model and get the model in .pkl format.
(2) Run RFtest_Concept.py to use the obtained classification model(.pkl file)to predict the classification of the test set.
