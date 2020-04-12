from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import os
import shutil
import csv
import boto3
import zipfile
import tarfile
from awsglue.utils import getResolvedOptions

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import StructField, StructType, StringType, DoubleType
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import *
from mleap.pyspark.spark_support import SimpleSparkSerializer


def csv_line(data):
    r = ','.join(str(d) for d in data[1])
    return str(data[0]) + "," + r


def main():
    # Initialize Spark session and variables
    spark = SparkSession.builder.appName("PySparkAbalone").getOrCreate()
    args = getResolvedOptions(sys.argv, ['S3_INPUT_BUCKET', 'S3_INPUT_KEY_PREFIX', 'S3_OUTPUT_BUCKET',
                                         'S3_OUTPUT_KEY_PREFIX', 'S3_MODEL_BUCKET', 'S3_MODEL_KEY_PREFIX'])
    
    # Save RDDs which is the only way to write nested Dataframes into CSV format
    spark.sparkContext._jsc.hadoopConfiguration().set("mapred.output.committer.class",
                                                      "org.apache.hadoop.mapred.FileOutputCommitter")
    
    # Defining the schema corresponding to the input data.
    schema = StructType([StructField("sex", StringType(), True), StructField("length", DoubleType(), True),
                         StructField("diameter", DoubleType(), True), StructField("height", DoubleType(), True),
                         StructField("whole_weight", DoubleType(), True), StructField("shucked_weight", DoubleType(), True),
                         StructField("viscera_weight", DoubleType(), True), StructField("shell_weight", DoubleType(), True),
                         StructField("rings", DoubleType(), True)])

    # Downloading the data from S3 into a Dataframe
    s3_path = 's3://' + os.path.join(args['S3_INPUT_BUCKET'], args['S3_INPUT_KEY_PREFIX'], 'abalone.csv')
    total_df = spark.read.csv(s3_path, header=False, schema=schema)

    # Build a feature preprocessing pipeline for categorical values, one-hot-encoding and vectorization
    cols = ["sex_vec", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]
    pipeline = Pipeline(stages=[StringIndexer(inputCol='sex', outputCol='indexed_sex'),
                               OneHotEncoder(inputCol="indexed_sex", outputCol="sex_vec"),
                               VectorAssembler(inputCols=cols, outputCol="features")])

    # Fit the data to our pipeline and split into training and validation sets
    etl = pipeline.fit(total_df)
    transformed_total_df = etl.transform(total_df)
    train_df, val_df = transformed_total_df.randomSplit([0.8, 0.2])

    # Convert train and val sets into RDD, save as CSV and upload to S3
    for df, name in [(train_df, 'train'), (val_df, 'valid')]:
        rdd = df.rdd.map(lambda x: (x.rings, x.features)).map(csv_line)
        rdd.saveAsTextFile('s3://' + os.path.join(args['S3_OUTPUT_BUCKET'], args['S3_OUTPUT_KEY_PREFIX'], name))

    # Serialize ETL pipeline, convert into tar.gz file and store binary using MLeap
    SimpleSparkSerializer().serializeToBundle(etl, "jar:file:/tmp/model.zip", val_df)
    with zipfile.ZipFile("/tmp/model.zip") as zf:
        zf.extractall("/tmp/model")

    with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
        tar.add("/tmp/model/bundle.json", arcname='bundle.json')
        tar.add("/tmp/model/root", arcname='root')
    
    # Upload the ETL pipeline in tar.gz format to S3 so that it can be used with SageMaker for inference later
    s3 = boto3.resource('s3') 
    file_name = os.path.join(args['S3_MODEL_KEY_PREFIX'], 'model.tar.gz')
    s3.Bucket(args['S3_MODEL_BUCKET']).upload_file('/tmp/model.tar.gz', file_name)


if __name__ == "__main__":
    main()
