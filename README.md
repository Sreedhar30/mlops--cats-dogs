**Cats vs Dogs – End-to-End MLOps Pipeline** 

**MLOPS Assignment 2 Group 59**



**Project Overview**



This project implements a complete end-to-end MLOps pipeline for a Cats vs Dogs image classification model using:



PyTorch



FastAPI



Docker



GitHub Actions



CI/CD Automation



Monitoring \& Logging



**Git repository** : https://github.com/Sreedhar30/mlops--cats-dogs



**Data Preprocessing**



Before inference, uploaded images undergo preprocessing:



Image decoding using OpenCV



Resizing to 224x224



Conversion to tensor using torchvision transforms



Adding batch dimension



Sending tensor to correct device (CPU/GPU)



This ensures consistency between training and inference pipeline.





**Model Serving (M1)**



The trained CNN model is deployed using FastAPI.



Endpoints:



POST /predict → Returns prediction and probabilities



GET /health → Used for monitoring and smoke testing



**Testing (M2)**



Implemented using Pytest.



Tests include:



Model loading validation



Preprocessing validation



API behavior validation



All tests run automatically during CI.



**Dockerization (M3)**



Dockerfile created



Image built and pushed to DockerHub



Containerized inference



Docker Image:



sreedharmlops/cats-dogs-api:latest



**CI/CD Pipeline (M3 + M4)**



On every push to main branch:



Install dependencies



Run unit tests



Build Docker image



Push image to DockerHub



Deploy container



Run smoke test (/health)



Pipeline file:



.github/workflows/ci.yml



**Monitoring \& Logging (M5)**



Implemented:



Structured logging



Request logging middleware



Execution time logging



Prediction logging



Error logging



Health endpoint monitoring



Logs accessible using:



docker logs sreedharmlops



**Deployment**



Run locally:



docker build -t cats-dogs-api .

docker run -p 8000:8000 cats-dogs-api



Open:



http://localhost:8000/docs

