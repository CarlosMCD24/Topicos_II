"""
module_ml.py
------------
Módulo encargado de:
- Definir y entrenar varios modelos de clasificación.
- Evaluar cada modelo con diferentes métricas.
- Integrar MLflow para registrar experimentos, métricas y parámetros.
"""

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


class WiDSModelRunner:
    """
    Clase para manejar la experimentación de múltiples modelos sobre los datos preprocesados.
    """
    
    def __init__(self, experiment_name: str = "WiDS_2024_Experiments"): # Inicializador
        # Configuramos MLflow para que use la carpeta mlruns local
        mlflow.set_tracking_uri("mlruns") # Carpeta local
        mlflow.set_experiment(experiment_name) # Nombre del experimento
        self.experiment_name = experiment_name # Nombre del experimento

    # ---------------------------------------------------------
    # 1) Definir modelos a experimentar
    # ---------------------------------------------------------
    def get_models(self) -> Dict[str, Any]: # Devuelve diccionario de modelos
        """
        Devuelve un diccionario de modelos a evaluar.
        Puedes agregar o quitar modelos según tus necesidades.
        """

        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=500, # Aumentado para convergencia
                multi_class="auto", # Manejo automático de clases múltiples
                n_jobs=-1, # Usar todos los núcleos disponibles
                solver="lbfgs"  # suficiente para un primer experimento
            ), # Modelo de regresión logística
            "DecisionTree": DecisionTreeClassifier( # Modelo de árbol de decisión
                max_depth=None, # Sin límite de profundidad
                random_state=42 # Para reproducibilidad
            ),
            "RandomForest": RandomForestClassifier( # Modelo de bosque aleatorio
                n_estimators=200, # Número de árboles
                max_depth=None, # Sin límite de profundidad
                n_jobs=-1, # Usar todos los núcleos disponibles
                random_state=42 # Para reproducibilidad
            ),
            "KNN": KNeighborsClassifier( # Modelo K-vecinos más cercanos
                n_neighbors=7 # Número de vecinos
            ),
            "MLP": MLPClassifier( # Perceptrón multicapa
                hidden_layer_sizes=(64, 32), # Dos capas ocultas
                activation="relu", # Función de activación ReLU
                max_iter=300, # Máximo de iteraciones
                random_state=42 # Para reproducibilidad
            ),
        }
        return models # Diccionario de modelos

    # ---------------------------------------------------------
    # 2) Entrenar y evaluar un modelo individual con MLflow
    # ---------------------------------------------------------
    def train_and_evaluate_model( # Función para entrenar y evaluar un modelo
        self, # Referencia a la instancia
        model_name: str, # Nombre del modelo
        model, # Instancia del modelo
        preprocessor, # Preprocesador (Pipeline)
        X_train, # Conjunto de entrenamiento
        y_train, # Etiquetas de entrenamiento
        X_val, # Conjunto de validación
        y_val, # Etiquetas de validación
    ) -> Dict[str, float]: # Devuelve diccionario de métricas
        """
        Entrena un modelo envuelto en un Pipeline (preprocesador + modelo),
        calcula métricas y registra todo en MLflow.
        """
        # Creamos un Pipeline sklearn: primero preprocesa, luego entrena el modelo
        clf = Pipeline(steps=[ # Pipeline de sklearn
            ("preprocessor", preprocessor), # Preprocesador
            ("model", model) # Modelo
        ])

        start_time = time.time() # Tiempo de inicio
        clf.fit(X_train, y_train) # Entrenamiento del modelo
        train_time = time.time() - start_time # Tiempo de entrenamiento

        # Predicciones
        y_pred = clf.predict(X_val) # Predicciones en validación

        # Algunas métricas básicas
        acc = accuracy_score(y_val, y_pred) # Exactitud
        f1 = f1_score(y_val, y_pred, average="weighted") # F1 ponderado

        # ROC-AUC requiere probabilidades y que sea problema binario o one-vs-rest
        try: # Intentar calcular ROC-AUC
            if hasattr(clf, "predict_proba"): # Verificar si el modelo tiene predict_proba
                y_proba = clf.predict_proba(X_val)[:, 1] # Probabilidades para la clase positiva
                roc_auc = roc_auc_score(y_val, y_proba) # Calcular ROC-AUC
            else: # Si no tiene predict_proba
                roc_auc = None # No disponible
        except Exception: # En caso de error
            roc_auc = None # No disponible

        # -------------------------------------------------
        # Registro en MLflow
        # -------------------------------------------------
        with mlflow.start_run(run_name=model_name): # Iniciar ejecución en MLflow
            # Parámetros: puedes loguear hiperparámetros importantes del modelo
            mlflow.log_param("model_name", model_name) # Nombre del modelo
            mlflow.log_param("train_samples", X_train.shape[0]) # Muestras de entrenamiento
            mlflow.log_param("val_samples", X_val.shape[0]) # Muestras de validación

            # Métricas
            mlflow.log_metric("accuracy", acc) # Exactitud
            mlflow.log_metric("f1_weighted", f1) # F1 ponderado
            if roc_auc is not None: # Si ROC-AUC está disponible
                mlflow.log_metric("roc_auc", roc_auc) # ROC-AUC
            mlflow.log_metric("train_time_sec", train_time) # Tiempo de entrenamiento

            # Guardamos el modelo completo (pipeline)
            mlflow.sklearn.log_model(clf, artifact_path="model") # Guardar modelo

        print(f"\n===== Resultados para {model_name} =====") # Resultados
        print(f"Accuracy:  {acc:.4f}") # Exactitud
        print(f"F1 (w):    {f1:.4f}") # F1 ponderado
        if roc_auc is not None: # Si ROC-AUC está disponible
            print(f"ROC-AUC:   {roc_auc:.4f}") # ROC-AUC
        else: # Si no está disponible
            print("ROC-AUC:   No disponible para este modelo.")
        print(f"Tiempo entrenamiento: {train_time:.2f} segundos")
        print("\nReporte de clasificación:")
        print(classification_report(y_val, y_pred))

        metrics = { # Diccionario de métricas
            "accuracy": acc, # Exactitud
            "f1_weighted": f1, # F1 ponderado
            "roc_auc": roc_auc if roc_auc is not None else float("nan"), # ROC-AUC
            "train_time_sec": train_time, # Tiempo de entrenamiento
        }
        return metrics

    # ---------------------------------------------------------
    # 3) Correr experimento con varios modelos y compararlos
    # ---------------------------------------------------------
    def run_experiments(
        self, # Referencia a la instancia
        preprocessor, # Preprocesador (Pipeline)
        X_train, # Conjunto de entrenamiento
        y_train, # Etiquetas de entrenamiento
        X_val, # Conjunto de validación
        y_val, # Etiquetas de validación
    ) -> Dict[str, Dict[str, float]]: # Devuelve diccionario de métricas por modelo
        """
        Ejecuta múltiples modelos, registra todo en MLflow y devuelve
        un diccionario con métricas por modelo.
        """
        models = self.get_models() # Obtener modelos
        results = {} # Diccionario para resultados

        for name, model in models.items():
            print("\n----------------------------------------")
            print(f"Entrenando modelo: {name}")
            print("----------------------------------------")
            metrics = self.train_and_evaluate_model( # Entrenar y evaluar modelo
                model_name=name, # Nombre del modelo
                model=model, # Instancia del modelo
                preprocessor=preprocessor, # Preprocesador
                X_train=X_train, # Conjunto de entrenamiento
                y_train=y_train, # Etiquetas de entrenamiento
                X_val=X_val, # Conjunto de validación
                y_val=y_val, # Etiquetas de validación
            )
            results[name] = metrics # Guardar métricas en resultados

        return results # Devolver resultados
