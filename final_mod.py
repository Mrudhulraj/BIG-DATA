from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import *
from sklearn.metrics import r2_score,accuracy_score, precision_score, recall_score

from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.linear_model import SGDClassifier
from pyspark.sql import Row
import pyspark.sql.types as tp
import joblib
import numpy as np
import csv

def write_to_logs(score, acc, pr, recall, fscore, path,flag):

    #column names
    fields = ['Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    #data row
    row = [score, acc, pr, recall, fscore]
    if flag==0:
        
    filename = path[6:9]+".csv" #To be changed

    with open(filename, 'a') as csvfile:
        
        write_to_csv = csv.writer(csvfile) #csv writer object created
        write_to_csv.writerow(row) #Writing to data rows

def preprocess(l,sc):
    spark = SparkSession(sc)
    
    #df = spark.createDataFrame(l,schema='Sentiment,subject_of_message string,content_of_message string,ham_spam string')
    df = spark.createDataFrame(l,schema=['tweet','Sentiment'])
    stage_1 = RegexTokenizer(inputCol= 'tweet' , outputCol= 'tokens', pattern= '\\W')
    # define stage 2: remove the stop words
    stage_2 = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
    # define stage 3: create a word vector of the size 100
    stage_3 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 8000)
    
    stage_4= StringIndexer(inputCol='Sentiment',outputCol='label')
    # applying the pre procesed pipeling model on the batches of data recieved
    pipe = Pipeline(stages=[stage_4,stage_1,stage_2,stage_3])
    
    cleaner = pipe.fit(df)
    
    clean_data = cleaner.transform(df)
    
    clean_data = clean_data.select(['label','tweet','filtered_words','vector'])
    
    clean_data.show()
    

    # splitting the batch data into 70:30 training and testing data
    
    (training,testing) = clean_data.randomSplit([0.75,0.25])
    
    X_train = np.array(training.select('vector').collect())
    
    y_train = np.array(training.select('label').collect())

    # reshaping the data
    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny))

    # doing the above stuff for the test data
    X_test = np.array(testing.select('vector').collect())
    y_test = np.array(testing.select('label').collect())
    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))
    return (X_test,y_test,X_train,y_train)


def pre_process_spam_SGD(l,sc):
    X_test,y_test,X_train,y_train = preprocess(l,sc)
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
        print(pred_batch)
        # calculating the metrics for performance
        score = r2_score(y_test, pred_batch)
        acc = accuracy_score(y_test, pred_batch)
        pr = precision_score(y_test, pred_batch)
        re = recall_score(y_test, pred_batch)
        if pr==0 or re == 0:
         fscore = 0
        else:
            fscore = (2*re*pr)/(re+pr)
        print("The r2 score is : ",score)
        print("the accuacy is ",acc)
        print("the precision: ",pr)
        print("the recall is:",re)
        print("the F1 score is:",fscore)
        log_write(score, acc, pr, re, fscore, 'build/SGD')
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
        if pr==0 or re == 0:
         fscore = 0
        else:
            fscore = (2*re*pr)/(re+pr)
        print("The r2 score is : ",score)
        print("the accuacy is ",acc)
        print("the precision: ",pr)
        print("the recall is:",re)
        print("the F1 score is:",fscore)
        write_to_logs(score, acc, pr, re, fscore, 'build/SGD')
        joblib.dump(clf, 'build/SGD.pkl')