from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import * 
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import final_mod as final
import json


sc = SparkContext("local[2]","test")

id_count = 1
ssc = StreamingContext(sc,1)
spark = SparkSession(sc)
sql_context = SQLContext(sc)
lines = ssc.socketTextStream('localhost',6100)
batches = lines.flatMap(lambda line: line.split("\n"))
def process(rdd,id_count):
        if not rdd.isEmpty():
            json_strings = rdd.collect()
            for rows in json_strings:
                temp_obj = json.loads(rows,strict = False)
            rows_spam = []
            for i in temp_obj.keys():
                temp_l = []
                temp_l.append(str(temp_obj[i]['feature1']))
                temp_l.append(str(temp_obj[i]['feature0']).strip(' '))
                rows_spam.append(tuple(temp_l))
            print("Recieved batch of data of length :",len(rows_spam))
            final.pre_process_spam_SGD(rows_spam,sc)
            #print(rows_spam)
batches.foreachRDD(lambda rdd : process(rdd,id_count))
id_count+=1

ssc.start()
ssc.awaitTermination()
