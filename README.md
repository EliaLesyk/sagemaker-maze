# Amazon SageMaker Maze

## What is it about

This maze exercise offers an interactive and progressive hands-on experience with SageMaker. The objective is to implement an end-to-end ML use-case and explore various aspects of ML life-cycle and SageMaker components along the way. Step by step we will be progressing through the journey, you will be asked to complete tasks and answer questions. The exercise offers minimum instructions and guidelines and relies on your curiosity and engineering mindset. Having a prior experience with SageMaker is preferred, but not necessary. We assume, though, that you practice data science / machine learning in your day job and comfortable with Python.

## How to play

* AWS environments are provided to all participants, a SageMaker Studio Domain is pre-provisioned and a GitHub repository for collaboration is provided.
* All participants are given the same tasks and questions. The recommendation is to organize yourself in groups of two. We heard that people who use SageMaker Studio for 20 years and those who joined the workshop for free food make a perfect team!
* Every task has a suggested time to complete (roughly 20 minutes) and time to review results (about 10 minutes). This should give you some guidance, but we don’t want you to rush. We want to explore SageMaker and learn together with your neighbouring teams.
* A group that completes a task and answers all questions is invited to walk the group through their results, share their answers with the rest of the team and push results of work to the shared git repository.
* As we build on top of what we learn, we will start every subsequent task from where we left in the previous task. We suggest using the work of the group that just presented as the starting point for everybody.

## Let's get started

We will open a SageMaker Studio application, clone a provided git repository. Let's get started.

## Chapter 1 - Where it all begins (training)

Model training can be quite a journey. It should not, however, be a long and tedious one. We will train a model from scratch and will make sure the process is repeatable and reliable.

### Task 1 - Simple scikit-learn Random Forest classifier

We will create a simple digit classification script that we can run from a terminal.

Create 'train.py' in SageMaker Studio under 'chapter_1/task_1' folder. The script should take [sklearn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) and implement digit class prediction using the built-in [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). To report model performance the script should calculate precision, recall, f1 metrics and print them to stdout. Also, the model should be stored in the local folder. Use the command ```conda activate base``` to initiate your environment. Run ```python train.py``` in a terminal (or in a notebook) to prove that the script is working.

#### 🤔 Mind tickling

* On which compute environment does the model training happen? Can you see that compute instance?
* What is the difference between "Open Image Terminal" and "System Terminal" options in the Launcher window (big plus icon in the top left corner)?
* Can you spot problems with training the model this way (with scikit-learn on SageMaker studio in a terminal)?

### Task 2 - Preparing for SageMaker Training

We now have a model training script. But how do we move it to a SageMaker Training container? To do that we need to change the structure of our script, just a bit. You will change the structure of the training script to ease development and to make sure that you can run and debug the script locally and eventually with SageMaker Training.

Create a Jupyter notebook and a local folder 'src/' under 'chapter_1/task_2'. To create Python file directly from a newly created Jupyter notebook, you can use ```%%writefile src/train.py``` Jupyter magic. It will let you save the script directly from your notebook into a Python file starting with the path where your notebook is at. Restructure 'train.py' from Task 1 so that there are two methods: ```main``` and ```fit``` as we want to be able to test the script locally. The main method should use ```argparse``` module to read command line arguments and pass them over to the ```fit``` method. Add this to the notebook cell with Jupyter magic. SageMaker Training will eventually call your script and pass over few important parameters: data paths, output model path, hyperparameters. Here is an example of how a training script looks like: [Preparing a Scikit-learn Training Script](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#prepare-a-scikit-learn-training-script).

#### 🤔 Mind tickling

* On which compute environment does the model training happen now? Can you see that compute instance?
* Can you change compute environment on which the notebook runs?

### Task 3 - SageMaker Training (with pre-built images)

Now that our 'train.py' accepts arguments, it is ready for more scalable training with SageMaker Training (and pre-built images)! You will upload training data to the dedicated Amazon S3 bucket and use SKLearn estimator that is part of SageMaker SDK to run a SageMaker Training process.

1. Place a new Jupyter notebook under 'chapter_1/task_3'.
2. Use ```boto3``` or AWS CLI to upload data to the dedicated S3 bucket. You will find the following boilerplate code useful:

```python
import os
import pandas as pd
import numpy as np
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# SDK setup
role = sagemaker.get_execution_role()
region = boto3.Session().region_name
sm = boto3.client('sagemaker')
boto_sess = boto3.Session(region_name=region)
sm_sess = sagemaker.session.Session(boto_session=boto_sess, sagemaker_client=sm)

# To store our data we use Amazon SageMaker default bucket.
BUCKET = sm_sess.default_bucket()
PREFIX = 'sagemaker-maze'
s3_data_url = f's3://{BUCKET}/{PREFIX}/data'
```

3. Adjust how 'train.py' parses command line arguments so that it uses environment variables for `model_dir` and `train` arguments. You might want to review [Preparing a Scikit-learn Training Script](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#prepare-a-scikit-learn-training-script) again. Have you noticed how SageMaker passes train and test data? What about hyperparameters?
4. Finally, create an instance of the class [SKLearn](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html) and invoke ```fit``` method. You can utilize the guide on how to [create a Scikit-learn Estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#create-an-estimator). Hint: make sure to specify `source_dir='src'` when creating an instance of SKLearn estimator and adjust hyperparameters.

The task is done when the SKLearn estimator completes training.

#### 🤔 Mind tickling

* On which compute environment does the model training happen now? Can you see that compute instance?
* Can you spot where the model is stored now? Can you download and unpack it?
* Can you locate where the training data is? In the AWS Management Console? By using AWS CLI?

### Task 4 - Unlocking meanings with observation

You have noticed that we keep using the same scikit-learn training script that we created in the beginning. Let us report more metrics and inspect the training process more closely. You will make SageMaker aware of the metrics you are reporting by adding 'metric_definitions' property to Sklearn estimator.

Copy the notebook from the previous task into the 'chapter_1/task_4' folder and follow "Define Metrics Using the SageMaker Python SDK section" from the [official documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html#define-train-metrics). Make sure to use the same definition of SkLearn estimator as before, just now with the 'metric_definitions' property set. Please pay attention to the regular expressions for your precision, recall and f1 metrics. Run the training job again.

The task is finished once you see the metrics from the completed Training job on SageMaker Console.

#### 🤔 Mind tickling

* Once the training job has started, where can you find the logs for it?
* Where can we see the custom metrics reported? In the SageMaker Management Console? In Amazon CloudWatch?
* How much MEM and CPU the training job consumed?
* Where to find hyperparameters that were used to launch the training job? In SageMaker Console? Using AWS CLI?

## Chapter 2 - Bringing value with predictions (inference)

Congratulations on successfully completing chapter 1! Now we can deploy the model trained and start making predictions.

### Task 1 - Roll the first inference endpoint out

We have trained our first scikit-learn model. To make predictions the model has to be deployed to an appropriate compute environment. You will deploy a new inference endpoint and will start making predictions.

Create a new Jupyter notebook 'inference.ipynb' under 'chapter_2/task_1'. Create a new custom script ```src/predict.py``` and place function ```def model_fn(model_dir)``` there (see the example below). As shown in the [SKLearnModel class documentation](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#load-a-model) this function helps SageMaker to load the model trained.

```python
%%writefile src/predict.py

import os 
import joblib

def model_fn(model_dir):
    # load sk learn model from model_dir
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model
```

Then create an instance of the [SKLearnModel class](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#attach-to-existing-training-jobs) by attaching SKLearn estimator to the existing job and invoking [```estimator.create_model()```](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#sagemaker.sklearn.estimator.SKLearn.create_model) function. Make sure to pass the recently created custom script as an entry point. Use the model instance to deploy a new inference endpoint with ```model.deploy()```. Regularly check the endpoint status and endpoint container logs in the SageMaker Console to make sure that you detect any issue with the endpoint provisioning. Use the endpoint created to predict numbers from the test dataset.

The task is finished once you make predictions using the endpoint deployed.

#### 🤔 Mind tickling

* On which compute environment does the model inference run? Can you see that compute instance?
* What other aspects of the inference process you can adjust with a custom script?
* How much time does it take to load the model (CloudWatch is your friend)?
* By looking at the inference logs, can you spot what HTTP server software is serving inference requests?

### Task 2 - Going serverless

We have just created a real-time inference endpoint, but is it the only option available? Well, there are other types of endpoints that you might find interesting as described [here](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html#deploy-model-options). For use cases with intermittent traffic a serverless inference option might be a good fit! You will change an inference endpoint to serverless, will run a simple load test and will see how the endpoint scales.

Update ```model.deploy``` call and pass an instance of the [ServerlessInferenceConfig](https://sagemaker.readthedocs.io/en/stable/api/inference/serverless.html#sagemaker.serverless.serverless_inference_config.ServerlessInferenceConfig) class. Make sure to allocate the minimum amount of memory possible. Then run a simple load test by using the code below.

```python
from multiprocessing import Process
from datetime import datetime

processes = []
def predict(i):
    response = predictor.predict(digits_df.iloc[:, :-1])
    now = datetime.now()
    print(f'{now} {response}. Process N {i} has completed.', end='\r')

for i in range(0, 1000):
    p = Process(target=predict, args=[i])
    processes.append(p)
    p.start()

for p in processes:
    p.join()
```

The task is finished when the load test completes.

#### 🤔 Mind tickling

* How much memory and vCPU is allocated to the inference endpoint by default? Can you constrain max concurrency?
* Can you see how the endpoint scales in response to the traffic increase?
* What are serverless endpoint's advantages over real-time inference endpoints?
* Would a serverless endpoint be the right choice for Large Language Models (LLMs)?

### Task 3 - Increasing throughput with batching

There are cases when we need to make inference over a large data set (GBs), or when we do not need to run a persistent endpoint. This is where batch transforms come into play. You will replace the endpoint definition with a batch transform definition and will run the inference again.

Upload data for inference to a dedicated location in the S3 bucket. Replace model.deploy() with [model.transformer()](https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.transformer) to create two instances of batch transformer, perform inference over the data uploaded. Have you noticed that only one EC2 instance was used? Fix it so that both instances are used for inference.

#### 🤔 Mind tickling

* How did you make sure that two instances of the batch transformer are used?
* Can you constrain how much data is passed to a single transformer instance at a time?
* How can you easily filter input data and output data using the transformer class?
