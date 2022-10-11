import unittest

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mlflow.tracking.client import MlflowClient

from cicd_demo.jobs.sample.entrypoint import SampleJob
from uuid import uuid4
from pyspark.dbutils import DBUtils  # noqa


class SampleJobIntegrationTest(unittest.TestCase):
    def setUp(self):

        self.test_dir = "dbfs:/tmp/tests/sample/%s" % str(uuid4())
        self.test_config = {"output_format": "delta", "output_path": self.test_dir}

        self.job = SampleJob(init_conf=self.test_config)
        self.dbutils = DBUtils(self.job.spark)
        self.spark = self.job.spark

    def test_sample(self):

        self.job.launch()

        output_count = (
            self.spark.read.format(self.test_config["output_format"])
            .load(self.test_config["output_path"])
            .count()
        )

        self.assertGreater(output_count, 0)

    def tearDown(self):
        self.dbutils.fs.rm(self.test_dir, True)


class demo_model():
    def __init__(self):
        self.x = load_iris().data
        self.y = load_iris().target
    def model_build(self):
        x_train , x_test , y_train , y_test = train_test_split(self.x,self.y,test_size = 0.3)
        with mlflow.start_run(run_name = "cicd_demo"):
            model = DecisionTreeClassifier(max_depth = 1 , random_state = 10)
            model.fit(x_train , y_train) 
            predicted = model.predict(x_test)
            accuracy = metrics.accuracy_score(y_test , predicted)
            mlflow.log_param("random_state" , 10)
            mlflow.log_param("max_depth" , 1)
            mlflow.log_metric("accuarcy" , accuracy)
            mlflow.sklearn.log_model(model , "decision_tree")
            #modelpath = "/dbfs/mlflow/iris/model-%s-%f" % ("decision_tree" , 2)
            #mlflow.sklearn.save_model(model,modelpath)
            #run_id  = mlflow.active_run().info.run_id
        
"""         model_name = "CICD_IRIS_model"
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id = run_id , artifact_path = "decision_tree")
        model_details = mlflow.register_model(model_uri = model_uri , name = model_name) """

"""         client = MlflowClient()
        client.transition_model_version_stage(
                    model_details.name , 
                    model_details.version,
                    stage = "Production") """


if __name__ == "__main__":
    # please don't change the logic of test result checks here
    # it's intentionally done in this way to comply with jobs run result checks
    # for other tests, please simply replace the SampleJobIntegrationTest with your custom class name
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(SampleJobIntegrationTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(tests)
    if not result.wasSuccessful():
        raise RuntimeError(
            "One or multiple tests failed. Please check job logs for additional information."
        )
    
    model_object = demo_model()
    model_object.model_build()
