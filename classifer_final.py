from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import * 
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.linear_model import SGDClassifier
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer, HashingTF, NGram
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from sklearn.metrics import r2_score,accuracy_score, precision_score, recall_score
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
import joblib
import numpy as np
import csv
flag = 0 

def csv_writer(acc,fscore,score,precision,recall,file_path):
    if flag == 0:
        fields = ['Accuracy', 'F1_score', 'Score', 'Precision', 'Recall']
        with open(f_name, 'a') as c:#c-> csvfile
		        cwriter_obj = csv.writer(c)#creating csv writer object
		        csvwriter_obj.writerow(fields)#writing the data rows
		        
		        
    row = [acc,fscore,score,precision,recall]
    
    f_name = file_path[6:9]+".csv"
    try:
    	with open(f_name, 'a') as c:#c-> csvfile
		    cwriter_obj = csv.writer(c)#creating csv writer object
		    csvwriter_obj.writerow(row)#writing the data rows
    global flag = 1
 
    
    
------------------------------------------------------------------------------------------


def data_preprocessing(tup,sc):
    spark = SparkSession(sc)#-----------
    #df = spark.createDataFrame(tup,schema='Sentiment,subject_of_message string,content_of_message string,ham_spam string')
    df = spark.createDataFrame(tup,schema=['tweet','Sentiment'])
	
    # preprocessing part (can add/remove stuff) , right now taking the column subject_of_message for spam detection
    #tokenizer = Tokenizer(inputCol="content_of_message", outputCol="token_text")

    stage_2 = RegexTokenizer(inputCol= 'tweet' , outputCol= 'tokens', pattern= '\\W')
    # define stage 2: remove the stop words
    stage_3 = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
    # define stage 3: create a word vector of the size 100
    stage_4 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize=8000)
    
    stage_1 = StringIndexer(inputCol='Sentiment',outputCol='label')
    # applying the pre procesed pipeling model on the batches of data recieved
    pipe = Pipeline(stages=[stage_1,stage_2,stage_3,stage4])
    
    cleaner = pipe.fit(df)
    
    
    
    clean_data = cleaner.transform(df)
    
    clean_data = clean_data.select(['label','stop_tokens','ht','bigrams'])
    
    #clean_data.show()
    

    # splitting the batch data into 70:30 training and testing data
    
    (training,testing) = clean_data.randomSplit([0.7,0.3])
    
    X_train = np.array(training.select('ht').collect())
    
    y_train = np.array(training.select('label').collect())

    # reshaping the data
    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny))

    # doing the above stuff for the test data
    X_test = np.array(testing.select('ht').collect())
    y_test = np.array(testing.select('label').collect())
    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))
    return (X_test,y_test,X_train,y_train)
    
    
    
    
    
    
 # Model for SGDClassifier
def pre_process_spam_SGD(tup,sc):
    X_test,y_test,X_train,y_train = data_preprocessing(tup,sc)
    '''
    Implement incremental learning
    '''
    try:
        '''
        loading the partial model
        '''
        print("Started increment learning")
        clf_load = joblib.load('build/SGD.pkl')
        clf_load.partial_fit(X_train,y_train.ravel())
        pred_batch = clf_load.predict(X_test)
        # calculating the metrics for performance
        score = r2_score(y_test, pred_batch)
        accuracy = accuracy_score(y_test, pred_batch)
        precision = precision_score(y_test, pred_batch)
        recall = recall_score(y_test, pred_batch)
        if precision == 0 or recall == 0:
         fscore = 0
        else:
            fscore = (2*recall*precision)/(recall+precision)
        print("The r2 score is : ",score)
        print("the accuacy is ",accuracy)
        print("the precision: ",precision)
        print("the recall is:",recall)
        print("the F1 score is:",fscore,end = '\n')
        csv_writer(score, accuracy, precision, recall, fscore, 'build/SGD')
        joblib.dump(clf_load, 'build/SGD.pkl')
    except Exception as e:
        '''
        training the model for the first time
        '''
        print("Started first train of SGD model")
        clf = SGDClassifier()
        clf.partial_fit(X_train,y_train.ravel(),classes=np.unique(y_train))
        pred_batch = clf.predict(X_test)
        print(pred_batch)
        # calculating the metrics for performance
        score = r2_score(y_test, pred_batch)
        acc = accuracy_score(y_test, pred_batch)
        pr = precision_score(y_test, pred_batch)
        re = recall_score(y_test, pred_batch)
        if pr == 0 or re == 0:
         fscore = 0
        else:
            fscore = (2*re*pr)/(re+pr)
        print("The r2 score is : ",score)
        print("the accuacy is ",acc)
        print("the precision: ",pr)
        print("the recall is:",re)
        print("the F1 score is:",fscore,end = '\n')
        csv_writer(score, acc, pr, re, fscore, 'build/SGD')
        joblib.dump(clf, 'build/SGD.pkl')

    # showing the data after preprocessing
    # clean_data.show()"""
