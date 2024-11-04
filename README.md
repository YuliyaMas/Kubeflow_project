# Kubeflow Pipeline for Author Classification and Text Generation

This project demonstrates a Kubeflow pipeline for a machine learning workflow that classifies authors based on text data
and generates text in the style of a predicted author.
Each stage of the ML lifecycle (preprocessing, training, evaluation, prediction) is containerized and orchestrated
through a Kubeflow pipeline on Minikube.

## Project Structure

``` 
kubeflow_pipeline_project/
├── components/
│   ├── preprocess/
│   │   ├── preprocess.py                # Preprocessing script
│   │   ├── Dockerfile                   # Dockerfile for preprocessing component
│   │   └── preprocess_component.yaml    # YAML definition for Kubeflow component
│   ├── train/
│   │   ├── train.py                     # Training script
│   │   ├── Dockerfile                   # Dockerfile for training component
│   │   └── train_component.yaml         # YAML definition for Kubeflow component
│   ├── evaluate/
│   │   ├── evaluate.py                  # Evaluation script
│   │   ├── Dockerfile                   # Dockerfile for evaluation component
│   │   └── eval_component.yaml          # YAML definition for Kubeflow component
│   └── predict/
│       ├── predict.py                   # Prediction and text generation script
│       ├── Dockerfile                   # Dockerfile for prediction component
│       └── predict_component.yaml       # YAML definition for Kubeflow component
├── pipeline.py                          # Kubeflow pipeline orchestration file
├── requirements.txt                     # Python dependencies
└── README.md                            # Documentation file
```


Each component (preprocess, train, evaluate, and predict) runs as an independent operation in the pipeline, managed by
Kubernetes on Minikube.

## Requirements

* Docker: Install Docker to build and deploy images. 
* Minikube: Install Minikube and Kubeflow to run the pipeline locally. 
* Python: Make sure you have Python 3.8 or higher. 
* Kubeflow Pipelines SDK: Install the Kubeflow Pipelines SDK to compile the pipeline. 

## Step 1: Build Docker Images

Navigate to each component subdirectory (e.g., components/preprocess, components/train, etc.), build the Docker images,
and push them to an image registry (Docker Hub, Google Container Registry,
or a local registry configured for Minikube).

Commandes bash:

docker build -t kubeflow_pipeline_project/components/preprocess_component:latest -f components/preprocess/Dockerfile .
docker push kubeflow_pipeline_project/components/preprocess_component:latest

docker build -t kubeflow_pipeline_project/components/train_component:latest -f components/train/Dockerfile .
docker push kubeflow_pipeline_project/components/train_component:latest

docker build -t kubeflow_pipeline_project/components/eval_component:latest -f components/evaluate/Dockerfile .
docker push kubeflow_pipeline_project/components/eval_component:latest

docker build -t kubeflow_pipeline_project/components/predict_component:latest -f components/generate/Dockerfile .
docker push kubeflow_pipeline_project/components/predict_component:latest

## Step 2: Compile the Pipeline

Compile the pipeline into a YAML file that Kubeflow can use by running pipeline.py. This will generate an
author_classification_pipeline.yaml file.
Commande bash:

python pipeline.py

## Step 3: Deploy the Pipeline on Kubeflow

Access the Kubeflow UI:
Start Minikube with Kubeflow, ensuring that the Kubeflow UI is accessible.
Open the Kubeflow Pipelines UI in your web browser.

Upload the Pipeline:
In the Kubeflow UI, go to Pipelines.
Click on Upload pipeline.
Select the author_classification_pipeline.yaml file that you generated.

Create a Pipeline Run:
Once the pipeline is uploaded, create a new pipeline run.
Configure the necessary parameters (e.g., input_path for input data and output_path for results).
Click Start to launch the pipeline.
	
