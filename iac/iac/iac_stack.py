from aws_cdk import (
    # Duration,
    Stack,
    aws_lambda_python_alpha,
    aws_lambda as _lambda,
    aws_iam as iam
    
)
from constructs import Construct

import aws_cdk as cdk

class IacStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here

        # creates a Lambda Layer for the dependencies from models/requirements.txt
        # dependencies_lambda_layer = aws_lambda_python_alpha.PythonLayerVersion(
        #     self, "DependenciesLayer",
        #     entry="../models",
        #     compatible_runtimes=[aws_lambda.Runtime.PYTHON_3_9, aws_lambda.Runtime.PYTHON_3_10, aws_lambda.Runtime.PYTHON_3_10],
        #     layer_version_name="DependenciesLayer",
        #     bundling={
        #     "asset_excludes":[".ipynb", 'venv']
        # }
    # )

        role = iam.Role(
            self,
            'myRole',
            assumed_by=iam.ServicePrincipal('lambda.amazonaws.com'),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name(managed_policy_name=('service-role/AWSLambdaBasicExecutionRole'))],
        )

        func = _lambda.DockerImageFunction(
            self,
            'EcrLambda',
            code=_lambda.DockerImageCode.from_image_asset(directory='../docker'),
            role=role,
            memory_size=3008,
            timeout=cdk.Duration.seconds(240),
            architecture=_lambda.Architecture.ARM_64
            )
        

        
        
                                  

