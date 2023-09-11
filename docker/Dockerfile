FROM public.ecr.aws/lambda/python:3.9

# Copies requirements.txt file into the container
COPY models/requirements.txt ${LAMBDA_TASK_ROOT}
COPY dependencies-test.py ${LAMBDA_TASK_ROOT}
COPY models/hybrid_model-python.py ${LAMBDA_TASK_ROOT}
COPY models/content_based-python.py ${LAMBDA_TASK_ROOT}
COPY models/content_base_lamda.py ${LAMBDA_TASK_ROOT}


# Installs dependencies found in your requirements.txt file
RUN  pip3 install -r requirements.txt

# Copy datasets
RUN mkdir ${LAMBDA_TASK_ROOT}/datasets
COPY datasets/ /${LAMBDA_TASK_ROOT}/datasets/




# Set the CMD to your handler start function
# CMD [ "dependencies-test.lambda_handler" ]
CMD [ "content_base_lamda.lambda_handler" ]


