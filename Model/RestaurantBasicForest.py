#!/usr/bin/env python

import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier


def main():
	
	trainingFile = '/InputData/training-data.csv'
	testFile = '/TestingData/testing-data.csv'
	probOutFile = '/ModelOutput/Restaurant-prob-out.csv'
	predOutFile = '/ModelOutput/Restaurant-predict-out.csv'

	print('Traning data processing')
	Train_data = pd.read_csv(trainingFile, sep=',')
	Train_data_frame = pd.DataFrame(Train_data)
	
	#normalize birth_years to years of age
	Train_data_frame['birth_year'] = (Train_data_frame['birth_year'] - 2015) * -1
	#binarize franchise column
	Train_data_frame['franchise'] = np.where(Train_data_frame['franchise']=='t',1,0)

	train_columns = ['birth_year','franchise','price']
	
    #target and train data
	target = Train_data_frame['rating']
	train = Train_data_frame[train_columns]
    
    #unique values for price and ages
    
	Train_price_range = list(set(train['price']))
	Train_age_range = list(set(train['birth_year']))
	#TRAIN YOUR RANDOM FOREST CLASSIFIER
	print('Fitting the model')
	rf = RandomForestClassifier(n_estimators=150, min_samples_split=2, n_jobs=-1)
	rf.fit(train, target)

    #Test Data
	print('Processing the test data')
	Test_data = pd.read_csv(testFile)
	#data frame for test data
	Test_data_frame = pd.DataFrame(Test_data)
	#normalize the birth_years to years of age
	Test_data_frame['birth_year'] = (Test_data_frame['birth_year'] - 2015) * -1
	#binarize franchise column
	Test_data_frame['franchise'] = np.where(Test_data_frame['franchise']=='t',1,0)
	
	#RUN RANDOM FOREST ON TEST DATA TO GET PREDICTION PROBABILITIES
	print('Running the model against test data.')
	predProbs = rf.predict_proba(Test_data_frame)
	csv.writer(open(probOutFile, 'wb')).writerows(predProbs)
	
	#RUN RANDOM FOREST ON TEST DATA TO GET PREDICTIONS
	prediction = rf.predict(Test_data_frame)
	with open(predOutFile,'wb') as myfile:
		wrtr = csv.writer(myfile, delimiter = ',')
		for i in range(len(prediction)):
			wrtr.writerow([prediction[i]])
	myfile.close
	print('Modeling fitting and test complete. Please check outputs')

if __name__=="__main__":
	main()