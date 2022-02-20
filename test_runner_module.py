from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import * 
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import *
from sklearn.metrics import r2_score,accuracy_score, precision_score, recall_score
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml.pipeline import PipelineModel
from sklearn.linear_model import SGDClassifier
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer,StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer, NGram
from pyspark.ml.feature import HashingTF as HT
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
import joblib
import numpy as np
import csv
import pyspark.sql.functions as F
import re

flag = 0 
model_flag = 0
max_f1score = 0

def csv_writer(accuracy , fscore , precision, recall , score , max_f1score , file_path):
    
    global flag
    f_name = file_path+".csv"
    
    if flag == 0 :
        flag =1
        with open(f_name,'a') as c:
            cwriter_obj = csv.writer(c)
            cwriter_obj.writerow(["accuracy" , "fscore" , "precision" , "recall" , "score" , "max_f1score" ])
    
    #creating data row
    row = [accuracy , fscore , precision, recall , score , max_f1score]
    
    with open(f_name, 'a') as c:
            cwriter_obj = csv.writer(c)
            cwriter_obj.writerow(row)
        


def data_preprocessing(tup,sc):
    spark = SparkSession(sc)
    df = spark.createDataFrame(tup,schema=['tweet','Sentiment'])

    df = (df.withColumn("tweet", F.regexp_replace("tweet", r"[@#&][A-Za-z0-9-]+", " ")))
    # preprocessing part (can add/remove stuff) , right now taking the column subject_of_message for spam detection
    stage_2 = Tokenizer(inputCol="tweet", outputCol="token")
    
    stopwords = StopWordsRemover().getStopWords() + ['-']
    
    stage_3 = StopWordsRemover(inputCol='token',outputCol='filtered_words').setStopWords(stopwords)
       
    stage_4 = NGram(n=2,inputCol='filtered_words',outputCol='bigrams')
    
    stage_5 = HT(inputCol="bigrams", outputCol="vector",numFeatures=8000)
    
    stage_1 = StringIndexer(inputCol='Sentiment',outputCol='label')


    # applying the pre procesed pipeling model on the batches of data recieved
    pipe = Pipeline(stages=[stage_1,stage_2,stage_3,stage_4,stage_5])
    cleaner = pipe.fit(df)
    
    clean_data = cleaner.transform(df)
    
    clean_data = clean_data.select(['label','tweet','filtered_words','vector'])    
    
    X_test = np.array(clean_data.select('vector').collect())
    
    y_test = np.array(clean_data.select('label').collect())

    dim_samples, dim_x, dim_y = X_test.shape
    
    X_test = X_test.reshape((dim_samples,dim_x*dim_y))
    
    return (X_test,y_test)
    
    
    
    
    
# Model for logistic_regression
def test_model(tup,sc):
    X_test,y_test = data_preprocessing(tup,sc)
    global model_flag,max_f1score
    try: 
        model_load = joblib.load('weights/MNB.pkl')
        pred_batch = model_load.predict(X_test)
        score = r2_score(y_test, pred_batch)
        accuracy = accuracy_score(y_test, pred_batch)
        precision = precision_score(y_test, pred_batch,zero_division=0)
        recall = recall_score(y_test, pred_batch)
        if precision == 0 or recall == 0:
            fscore = 0
        else:
            fscore = (2*recall*precision)/(recall+precision)   
        
        print("Accuracy:",accuracy*100,"%")
        print("Precision: ",precision)
        print("Recall:",recall)
        print("F1score:",fscore)
        print("R2 score:",score)
        if max_f1score < fscore:
            max_f1score = fscore
        csv_writer(accuracy , fscore , precision, recall , score , max_f1score, 'Test')
        print("\n iteration ended \n")
    except Exception as e:
        print("error occured",e)
