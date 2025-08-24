# Transformer-Based Abbreviation Detection System

This repository contains the coursework for a **Natural Language Processing (NLP)** project focused on **Abbreviation Detection** using various techniques and models. The project includes exploratory data analysis, multiple experiments comparing preprocessing techniques, loss functions, and word embeddings, as well as a Flask-based web application for serving the trained model.

---

## Table of Contents
- [Project Overview](#project-overview)  
- [Project Structure](#project-structure)  
- [Experiments](#experiments)  
  - [Experiment 1: Preprocessing Techniques](#experiment-1-preprocessing-techniques)  
  - [Experiment 2: Loss Functions](#experiment-2-loss-functions)  
  - [Experiment 3: Word Embeddings](#experiment-3-word-embeddings)  
  - [Experiment 4: Additional Training/Validation Data](#experiment-4-additional-trainingvalidation-data)  
- [Deployment](#deployment)  
- [How to Run](#how-to-run)  
- [Reporting](#reporting)  

---

## Project Overview
The core objective of this project is to develop and evaluate a **Named Entity Recognition (NER)** system specifically tailored for **abbreviation detection**. It explores how different NLP methodologies impact model performance, culminating in a deployable web service.

---

## Project Structure
The repository is organized as follows:

- `Main_Notebook.ipynb`: Primary notebook for overall project context, exploratory data analysis (EDA), and links to individual experiments.  
- `EXPERIMENT-1.ipynb`: Compares different preprocessing techniques.  
- `EXPERIMENT-2.ipynb`: Compares various loss functions.  
- `EXPERIMENT-3.ipynb`: Explores the impact of different word embeddings.  
- `EXPERIMENT-4.ipynb`: Investigates the effect of augmenting the training dataset.  
- `app.py`: Flask application for model serving.  
- `requirements.txt`: Lists all Python dependencies.  
- `ci-cd-pipeline.yml`: GitHub Actions workflow for CI/CD.  
- `locustfile.py`: Locust script for API load testing.  
- `deployment-report.pdf`: Documentation detailing the deployment process.  
- `project-report.pdf`: Comprehensive project report.  
- `roberta-base-model/`: Directory containing the saved RoBERTa base model for token classification.  

---

## Experiments

This section details the various experiments conducted to optimize the abbreviation detection model. Each experiment focuses on a different aspect of the NLP pipeline.

### Experiment 1: Preprocessing Techniques
Compares the performance of the NER model using different preprocessing strategies:  

- **System 1**: Preprocessing by Lowercasing  
- **System 2**: Preprocessing by Stopword and Punctuation Removal  
- **System 3**: Preprocessing by Lemmatization  
- **System 4**: All of the above  

### Experiment 2: Loss Functions
Evaluates the impact of different loss functions:  

- **System 1**: Categorical Cross Entropy Loss  
- **System 2**: Label Smoothing with Categorical Cross Entropy  
- **System 3**: Focal Loss  

### Experiment 3: Word Embeddings
Investigates the influence of word embeddings:  

- **System 1**: Word2Vec Embeddings  
- **System 2**: FastText Embeddings  

### Experiment 4: Additional Training/Validation Data
Explores the effect of dataset augmentation using **PLOD-Filtered** combined with **PLOD-CW**:  

- **System 1**: Train solely on PLOD-CW  
- **System 2**: Additional 1% of data from PLOD-Filtered + PLOD-CW  
- **System 3**: Additional 5% of data from PLOD-Filtered + PLOD-CW  

---

## Deployment
The system includes a **Flask-based web application (`app.py`)** that serves the trained NER model via a user-friendly API endpoint.  

- **Deployment strategy**, **testing procedures**, **logging**, and **CI/CD pipeline** are documented in `deployment-report.pdf`.  
- A CI/CD pipeline (`ci-cd-pipeline.yml`) is configured using **GitHub Actions** to automate build, test, and deployment.  
- **Load testing** is performed using `locustfile.py` to measure performance under different loads.  

---

## How to Run

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    It's recommended to create a virtual environment first:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Download the PLOD-CW and PLOD-Filtered datasets:**
    The notebooks use `load_dataset("surrey-nlp/PLOD-CW")` and `load_dataset("surrey-nlp/PLOD-Filtered")`. Ensure you have an internet connection when running the notebooks for the first time so these datasets can be downloaded.

4.  **Run the Jupyter Notebooks:**
    To explore the experiments and data analysis:

    ```bash
    jupyter notebook Main_Notebook.ipynb
    ```

    From `Main_Notebook.ipynb`, you can navigate to `EXPERIMENT-1.ipynb`, `EXPERIMENT-2.ipynb`, `EXPERIMENT-3.ipynb`, and `EXPERIMENT-4.ipynb`.

5.  **Run the Flask API:**
    To start the abbreviation detection API:

    ```bash
    python app.py
    ```

    The API will typically run on `http://0.0.0.0:5001`. You can then send POST requests to `/predict` with JSON `{"text": "your input sentence"}` to get predictions.

6.  **Run Load Tests (Optional):**
    With the Flask app running, you can perform load testing using Locust:

    ```bash
    locust -f locustfile.py --host [http://0.0.0.0:5001](http://0.0.0.0:5001)
    ```

    Then, open your browser and go to `http://localhost:8089` to access the Locust web UI.

## Reporting

A detailed **project report** (`project-report.pdf`) is included, providing in-depth analysis of the dataset, experimental setups, results, and discussion of findings. The **deployment report** (`deployment-report.pdf`) further elaborates on the choices made for model serving, testing, and CI/CD.
