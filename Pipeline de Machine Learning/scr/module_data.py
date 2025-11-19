"""Módulo encargado de:

- Cargar los datos de WiDS Datathon 2024 desde la carpeta data/.
- Separar la variable objetivo.
- Detectar variables numéricas y categóricas.
- Construir un preprocesador (imputación + escalado + OneHotEncoder).
- Generar splits de entrenamiento.
"""
# Librerías necesarias para manejo de datos y preprocesamiento
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Clase principal para manejo de datos
class WiDSDataModule:
    # Inicialización con parámetros de configuración
    def __init__(
        self,
        data_dir: str = "data", # Directorio donde están los datos
        train_filename: str = "training.csv", # Nombre del archivo de entrenamiento
        test_filename: str = "test.csv", # Nombre del archivo de test
        target_col: str = "DiagPeriodL90D",  # AJUSTA si tu target se llama diferente
        scaler_type: str = "standard", # "standard" o "minmax"
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        # Rutas completas a los archivos
        self.data_dir = data_dir
        self.train_path = os.path.join(self.data_dir, train_filename)
        self.test_path = os.path.join(self.data_dir, test_filename)
        
        # Nombre de la columna objetivo
        self.target_col = target_col
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.random_state = random_state

        # Estos se rellenan después de cargar y preparar los datos
        self.df_train = None
        self.df_test = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.preprocessor = None

        # 1. Carga training.csv y test.csv desde la carpeta data
   
    def load_data(self):
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"No se encontró el archivo de entrenamiento: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"No se encontró el archivo de test: {self.test_path}")

        print("Cargando datos...")
        print(f" - training: {self.train_path}")
        print(f" - test    : {self.test_path}")

        # Cargar los datos en dataframes
        self.df_train = pd.read_csv(self.train_path)
        self.df_test = pd.read_csv(self.test_path)

        # Verificar que la columna objetivo exista
        if self.target_col not in self.df_train.columns:
            raise ValueError(
                f"La columna objetivo '{self.target_col}' no existe en df_train.\n"
                f"Columnas disponibles: {self.df_train.columns.tolist()}"
            )

        print("Datos cargados correctamente.")
        print(f" - df_train: {self.df_train.shape[0]} filas, {self.df_train.shape[1]} columnas")
        print(f" - df_test : {self.df_test.shape[0]} filas, {self.df_test.shape[1]} columnas")

        # 2. Preprocesador (imputación + escalado + one-hot) Construye un preprocesador usando ColumnTransformer y Pipelines.
            #  Detecta columnas numéricas y categóricas 
            # Imputación de NaN en numéricas y categóricas
            # Escalado en numéricas
            # OneHotEncoder en categóricas
    def _build_preprocessor(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        print(f"Columnas numéricas: {len(num_cols)}")
        print(f"Columnas categóricas: {len(cat_cols)}")

        # Elegimos tipo de scaler
        if self.scaler_type == "standard":
            scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type debe ser 'standard' o 'minmax'.")

        # Numéricas: imputar mediana + escalar
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ])

        # Categóricas: imputar valor más frecuente + one-hot
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        # Combinamos ambos transformadores
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ]
        )

        return preprocessor

    # 3) Preparar datos (split train/val + preprocesador)
    def prepare_data(self):
        
        # Separa X/y, hace el train/validation split y construye el preprocesador.
        if self.df_train is None:
            raise ValueError("Primero debes llamar a load_data().")

        # Separar features y target
        X = self.df_train.drop(columns=[self.target_col])
        y = self.df_train[self.target_col]

        # Split estratificado
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        print(f"Split realizado: {self.X_train.shape[0]} train / {self.X_val.shape[0]} val")

        # Construir preprocesador con las columnas de X_train
        self.preprocessor = self._build_preprocessor(self.X_train)
        print("Preprocesador construido.")

    
    # 4) Devuelve todo listo para los modelos
    def get_splits_and_preprocessor(self):
        
        # Devuelve X_train, X_val, y_train, y_val y el preprocesador.
        if any(v is None for v in [self.X_train, self.X_val, self.y_train, self.y_val, self.preprocessor]):
            raise ValueError("Debes llamar a prepare_data() antes de obtener los splits.")
        return self.X_train, self.X_val, self.y_train, self.y_val, self.preprocessor
