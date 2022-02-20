from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import * 
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import process_temp
import json


sc = SparkContext("local[2]","test")

id_count = 1
ssc = StreamingContext(sc,1)
spark = SparkSession(sc)
sql_context = SQLContext(sc)
lines = ssc.socketTextStream('localhost',6100)
batches = lines.flatMap(lambda line: line.split("\n"))
def process(rdd,id_count):
        # id_count is redundant
        if not rdd.isEmpty():
            json_strings = rdd.collect()
            for rows in json_strings:
                temp_obj = json.loads(rows,strict = False)
            rows_spam = []
            for i in temp_obj.keys():
                temp_l = []
                temp_l.append(len(str(temp_obj[i]['feature1'])))
                temp_l.append(str(temp_obj[i]['feature0']).strip(' '))
                temp_l.append(str(temp_obj[i]['feature1']).strip(' '))
                temp_l.append(str(temp_obj[i]['feature2']).strip(' '))
                rows_spam.append(tuple(temp_l))
            print("Recieved batch of data of length :",len(rows_spam))
            
            # calling the bernouli nb model
            # process_temp.pre_process_spam_bnb(rows_spam,sc)
            
            # calling the multinomial nb model
            process_temp.pre_process_spam_mnb(rows_spam,sc)

            # calling the svgd classifer
            #process_temp.pre_process_spam_SGD(rows_spam,sc)

batches.foreachRDD(lambda rdd : process(rdd,id_count))
id_count+=1

ssc.start()
ssc.awaitTermination()