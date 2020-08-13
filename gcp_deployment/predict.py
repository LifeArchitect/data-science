import apache_beam as beam
import argparse
import json
import pandas as pd
import joblib
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.io.gcp.bigquery import WriteToBigQuery, BigQueryDisposition as bqd
from google.cloud import bigquery, storage
from datetime import datetime
from io import BytesIO
from tempfile import TemporaryFile

class Predict(beam.DoFn):

    def __init__(self):
        self._model = None
        self._storage = storage
        self._pd = pd
        self.model_file = 'model.joblib'
     
    def process(self, element):
        # get model from storage
        if self._model is None:
            blob = self._storage.Client().get_bucket('models123').get_blob(self.model_file)
            # blob.download_to_file(BytesIO())
            #self._model = joblib.load(self.model_file)
            with open("xgb.joblib", "wb"):
                blob.download_to_filename("xgb.joblib")
                self._model = joblib.load("xgb.joblib")
                
        
        new_x = self._pd.DataFrame.from_dict(element, orient = "index").transpose().fillna(0)   
        new_x.pageviews = float(new_x.pageviews)
        
        # make prediction
        prob = str(self._model.predict_proba(new_x.iloc[:, 1:])[0][1])
        return [{ 'customer_id': str(element['customer_id']), 
                  'purchase_probability': prob,
                  'model': 'XGBClassifier',
                  'datetime': str(datetime.now().strftime("%d-%m-%Y,%H:%M:%S")),
                }]
    
def collect(prediction):
    # groupby customer_id to get column means and visit_count
    output.append(prediction)
    return prediction

if __name__ == '__main__':
    pipeline_args = {'project': 'ml-pipeline-285203', 'runner:': 'DirectRunner', 'streaming': True}
    pipeline_options = PipelineOptions(**pipeline_args)

    # define the pipeline steps
    p = beam.Pipeline(options=pipeline_options)
    output = []
    query = " SELECT \
                customer_id, \
                SUM(session_duration) as timeOnSite, \
                SUM(page_views) as pageviews, \
                AVG(satisfaction_score) as sessionQualityDim, \
                AVG(product_price) as productPrice, \
                COUNT(*) AS visits \
              FROM `ml-pipeline-285203.raw_data.raw_data` \
              GROUP BY `customer_id`"
    source = beam.io.BigQuerySource(query=query, use_standard_sql=True)
    pipeline = (
        p | 'Retrieve Data' >> beam.io.Read(source)
          | 'Apply Model' >> beam.ParDo(Predict())
          | 'View Predictions' >> beam.Map(collect)
          | 'Save to BigQuery' >>  WriteToBigQuery(
                                   table='ml-pipeline-285203:prediction_data.predictions',
                                   schema='customer_id:INTEGER, purchase_probability:FLOAT, model:STRING, datetime:STRING',
                                   create_disposition=bqd.CREATE_IF_NEEDED, 
                                   write_disposition=bqd.WRITE_APPEND)
    )

    # run the pipeline
    result = p.run()
    result.wait_until_finish()
    print(output)