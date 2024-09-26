import mlflow 
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema
from mlflow.models.signature import infer_signature
from mlflow.types.schema import ParamSchema
from mlflow.types.schema import ParamSpec
from mlflow.types.schema import ColSpec

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification 
import pandas as pd 
from typing import Tuple 


def get_train_data()->tuple[pd.DataFrame]:
    
    """
    Generate train and test data.
    :return:x_train,x_test,y_train,y_test
    """
    
    x,y =make_classification()
    features = [f"feature_{i+1}" for i in range(x.shape[1])]
    df = pd.DataFrame(x,columns=features)
    df["label"] = y
    
    return df[features],df["label"]




if __name__ =="__main__":
    x_train ,y_train = get_train_data()
    
#     col_Spec = []
    
#     data_map = {
#         'int64':'integer',
#         'float64':'double',
#         'bool':'boolean',
#         'str':'string',
#         "date":'datetime'
#     }
    
    
#     #schema = Schema()
    
#     # for name,dtype in x_train.dtypes.to_dict().items():
#     #     col_Spec.append(ColSpec(name=name,type=data_map[str(dtype)]))
        
        
#     input_schema = Schema(inputs=col_Spec)
#     output_schema = Schema([ColSpec(name="label",type="integer")])
    
#     parameter = ParamSpec(name="model_name",dtype="string",default="model1")
#     param_schema = ParamSchema(params=[parameter])
    
    
#     model_signature = ModelSignature(inputs=input_schema, outputs=output_schema,params=param_schema)
#     print("Model Signature")
#     print(model_signature.to_dict())

model_signature = infer_signature(x_train, y_train,params={"model_name":"model1"})
print("Model Signature")
print(model_signature.to_dict())


if __name__ == "__main__":
    
    
    experiment = mlflow.create_experiment(
        'Model_signature',
        #artifact_location = "testing_mlflow1_artifacts",
        tags = {"env":"dev","version":"1.0.0"},
    )
    
    mlflow.set_experiment(experiment_name="Model_signature")
    
    with mlflow.start_run(run_name="Model_signature") as run:
        mlflow.sklearn.log_model(sk_model=RandomForestClassifier(),artifact_path="model_signature",signature=model_signature)
      