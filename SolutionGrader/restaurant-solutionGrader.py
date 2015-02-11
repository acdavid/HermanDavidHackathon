#!/usr/bin/env python

from __future__ import division
import numpy as np
import csv
import scipy
import sys
import pandas as pd

def main():
    
	submittedFile = 'ModelOutput/Restaurant-predict-out.csv'
	answerFile = '/Solution/Restaurant-answerKey.csv'
	totalScore = 100
		
	#READ IN THE ANSWER FILE -- FIRST COLUMN THE CLASSIFICATION; OTHERS IRRELEVANT
	print('Inputting answer key.')
	data = pd.read_csv(answerFile,sep=',')
	answerKey = pd.DataFrame(data)
		
	#get the rating
	actual = answerKey['rating']
	
	#READ IN SUBMITTED FILE -- THIS SHOULD ONLY HAVE A SINGLE COLUMN WITH THE CLASSIFICATION
	print('Inputting submitted file.')
	readFile = pd.read_csv(submittedFile,sep=',', header=None)
	submitted_answers = pd.DataFrame(readFile)
		
	#ACTUALLY CALCULATE THE SCORE
	print('Comparing actuals and answers.')
	errors = 0
	
	for i in range(len(actual)):
		errors = errors + (submitted_answers[0][i] - actual[i])**2
	
	if errors > 0:
		print('Your score is: ' + str(round(totalScore*(len(actual) - errors) /len(actual))))+'%'
	else:
		print('You scored a 100%! Nice work!')
	print('Thank you and have a nice day!')

if __name__=="__main__":
    main()