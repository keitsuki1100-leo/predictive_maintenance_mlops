// Jenkinsfile — Predictive Maintenance MLOps Pipeline

pipeline {
    agent any

    environment {
        PYTHON = 'py'
        PROJECT_DIR = 'C:\\Users\\MAYANK\\Desktop\\predictive_maintenance_mlops'
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'Checking out project...'
                echo 'Project: Predictive Maintenance for Remote IoT'
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing Python dependencies...'
                bat 'py -m pip install mlflow scikit-learn pandas numpy fastapi uvicorn'
            }
        }

        stage('Generate Sensor Data') {
            steps {
                echo 'Generating IoT sensor data...'
                bat 'py src/generate_sensor_data.py'
            }
        }

        stage('Train Model') {
            steps {
                echo 'Training RandomForest model...'
                bat 'py src/train_model.py'
            }
        }

        stage('Evaluate and Register') {
            steps {
                echo 'Evaluating model and registering to MLflow...'
                bat 'py src/evaluate_and_register.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image...'
                bat 'docker build -t predictive-maintenance-api .'
            }
        }

        stage('Run Docker Container') {
            steps {
                echo 'Running Docker container...'
                bat 'docker stop pm-api || exit 0'
                bat 'docker rm pm-api || exit 0'
                bat 'docker run -d -p 8000:8000 --name pm-api predictive-maintenance-api'
                echo 'API is running at http://localhost:8000'
            }
        }

    }

    post {
        success {
            echo '================================================'
            echo 'Pipeline completed successfully!'
            echo 'Model trained, evaluated and deployed!'
            echo 'API running at http://localhost:8000/docs'
            echo '================================================'
        }
        failure {
            echo '================================================'
            echo 'Pipeline failed! Check the logs above.'
            echo '================================================'
        }
    }
}