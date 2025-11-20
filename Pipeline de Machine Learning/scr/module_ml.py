# Módulo encargado de:
    # Definir y entrenar varios modelos de clasificación.
    # Evaluar cada modelo con diferentes métricas.
    # Integrar MLflow para registrar experimentos, métricas y parámetros.

# Instalamos las librerías necesarias para manejo de modelos, métricas y MLflow
import time
from typing import Dict, Any

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Clase para manejar la experimentación de múltiples modelos sobre los datos preprocesados
class WiDSModelRunner:
    
    def __init__(self, experiment_name: str = "WiDS_2024_Experiments"): # Inicializador
        
        # Configuramos MLflow para que use la carpeta mlruns local
        mlflow.set_tracking_uri("mlruns")  
        mlflow.set_experiment(experiment_name) 
        self.experiment_name = experiment_name 

        # 1. Definimos los modelos a experimentar
    def get_models(self) -> Dict[str, Any]: 
        
        # Devuelve un diccionario de modelos a evaluar y podemos agregar más o quitar según se necesite.
             
        models = {
            "LogisticRegression": LogisticRegression( 
                max_iter=500, 
                multi_class="auto", 
                n_jobs=-1, 
                solver="lbfgs"  
            ), 
            "DecisionTree": DecisionTreeClassifier( 
                max_depth=None, 
                random_state=42
            ),
            "RandomForest": RandomForestClassifier( 
                n_estimators=200, 
                max_depth=None, 
                n_jobs=-1, 
                random_state=42 
            ),
            "KNN": KNeighborsClassifier( 
                n_neighbors=7 
            ),
            "MLP": MLPClassifier( 
                hidden_layer_sizes=(64, 32), 
                activation="relu", 
                max_iter=300, 
                random_state=42 
            ),
        }
        return models

    # 2. Entrenamos y evaluamos un modelo individual con MLflow
    def train_and_evaluate_model( 
        self, 
        model_name: str, 
        model, 
        preprocessor, 
        X_train, 
        y_train, 
        X_val, 
        y_val,
    ) -> Dict[str, float]: 
        
        # Se entrena un modelo envuelto en un Pipeline (preprocesador + modelo), se evalúa y se registran métricas en MLflow.
               
        # Creamos un Pipeline con sklearn: primero preprocesa, luego entrena el modelo 
        clf = Pipeline(steps=[ 
            ("preprocessor", preprocessor), 
            ("model", model) 
        ])

        # Entrenamiento del modelo, medimos el tiempo
        start_time = time.time()
        clf.fit(X_train, y_train) 
        train_time = time.time() - start_time 

        # Predicciones en el conjunto de validación
        y_pred = clf.predict(X_val) 

        # Algunas métricas básicas en sklearn para clasificación
        acc = accuracy_score(y_val, y_pred) 
        f1 = f1_score(y_val, y_pred, average="weighted")

        # ROC-AUC requiere probabilidades y que sea problema binario o one-vs-rest para multiclase e intenta calcularlo
        try:
            if hasattr(clf, "predict_proba"): 
                y_proba = clf.predict_proba(X_val)[:, 1] 
                roc_auc = roc_auc_score(y_val, y_proba) 
            else: 
                roc_auc = None 
        except Exception: 
            roc_auc = None

        # Registro en MLflow, iniciamos una nueva ejecución
        with mlflow.start_run(run_name=model_name):
            # Parámetros: Se pueden loguear hiperparámetros importantes del modelo
            mlflow.log_param("model_name", model_name) 
            mlflow.log_param("train_samples", X_train.shape[0]) 
            mlflow.log_param("val_samples", X_val.shape[0]) 

            # Métricas para evaluar el modelo
            mlflow.log_metric("accuracy", acc) 
            mlflow.log_metric("f1_weighted", f1) 
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("train_time_sec", train_time) 

            # Guardamos el modelo completo (pipeline)
            mlflow.sklearn.log_model(clf, artifact_path="model") 
        
        # Imprimimos un resumen de las métricas
        print(f"\n=== Resultados para {model_name} === ") 
        print(f"Accuracy:  {acc:.4f}") 
        print(f"F1 (w):    {f1:.4f}") 
        if roc_auc is not None: 
            print(f"ROC-AUC:   {roc_auc:.4f}") 
        else: 
            print("ROC-AUC:   No disponible para este modelo.")
        print(f"Tiempo entrenamiento: {train_time:.2f} segundos")
        print("\nReporte de clasificación:")
        print(classification_report(y_val, y_pred))

        metrics = { 
            "accuracy": acc,
            "f1_weighted": f1, 
            "roc_auc": roc_auc if roc_auc is not None else float("nan"), 
            "train_time_sec": train_time,
        }
        return metrics

        # 3. Corremos el experimento con varios modelos y los compararmos
    def run_experiments(
        self, 
        preprocessor, 
        X_train, 
        y_train, 
        X_val, 
        y_val, 
    ) -> Dict[str, Dict[str, float]]: 
        
        # Ejecuta múltiples modelos, registra todo en MLflow y devuelve un diccionario con los resultados.
                
        models = self.get_models() 
        results = {} 

        for name, model in models.items():
            print(f"Entrenando modelo: {name}")
            
            metrics = self.train_and_evaluate_model( 
                model_name=name, 
                model=model, 
                preprocessor=preprocessor, 
                X_train=X_train, 
                y_train=y_train, 
                X_val=X_val, 
                y_val=y_val,
            )
            results[name] = metrics 

        return results 
