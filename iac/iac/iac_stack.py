from aws_cdk import (
    # Duration,
    Stack,
    aws_lambda_python_alpha,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_ec2 as ec2, aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns
    
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

        func.add_function_url(
            auth_type=_lambda.FunctionUrlAuthType.NONE
        )
        
        # vpc = ec2.Vpc(self, "VPC-Fargate", max_azs=3)     # default is all AZs in region

        # cluster = ecs.Cluster(self, "RecomendacaoCluster", vpc=vpc)

        # ecs_patterns.ApplicationLoadBalancedFargateService(self, "RecomendacaoService",
        #     cluster=cluster,            # Required
        #     cpu=512,                    # Default is 256
        #     desired_count=6,            # Default is 1
        #     task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
        #         image=ecs.ContainerImage.from_registry("558695568462.dkr.ecr.us-east-1.amazonaws.com/cdk-hnb659fds-container-assets-558695568462-us-east-1")),
        #     memory_limit_mib=2048,      # Default is 512
        #     public_load_balancer=True)  # Default is True

        
        
        
                                  

