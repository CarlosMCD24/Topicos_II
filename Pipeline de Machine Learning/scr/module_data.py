"""
module_data.py
---------------
Módulo encargado de:
- Cargar los datos de WiDS Datathon 2024 desde la carpeta data/.
- Separar la variable objetivo.
- Detectar variables numéricas y categóricas.
- Construir un preprocesador (scaler + OneHotEncoder).
- Generar splits de entrenamiento y validación listos para usar con modelos de sklearn.
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class WiDSDataModule:
    """
    Clase para manejar la carga y preprocesamiento de los datos de WiDS 2024.
    """

    def __init__(
        self,
        data_dir: str = r"C:\Users\INIFAP-MOVIL\Documents\3 TERCER SEMESTRE\Topicos II\Trabajos\Topicos_II\Topicos_II\Pipeline de Machine Learning\data",
        train_filename: str = "training.csv",
        test_filename: str = "test.csv",
        target_col: str = "DiagPeriodL90D",  # 👈 Ajusta si tu columna objetivo tiene otro nombre
        scaler_type: str = "standard",       # "standard" (StandardScaler) o "minmax" (MinMaxScaler)
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        # Ruta base donde está tu carpeta data
        self.data_dir = data_dir
        self.train_path = os.path.join(self.data_dir, train_filename)
        self.test_path = os.path.join(self.data_dir, test_filename)

        self.target_col = target_col
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.random_state = random_state

        # Estos atributos se llenan al llamar a load_data() y prepare_data()
        self.df_train = None
        self.df_test = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.preprocessor = None  # ColumnTransformer

    # ---------------------------------------------------------
    # 1) Carga de datos
    # ---------------------------------------------------------
    def load_data(self):
        """Carga los CSV de training y test desde la carpeta data/ especificada."""
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"No se encontró el archivo de entrenamiento: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"No se encontró el archivo de test: {self.test_path}")

        print("📂 Cargando datos...")
        print(f" - training: {self.train_path}")
        print(f" - test    : {self.test_path}")

        self.df_train = pd.read_csv(self.train_path)
        self.df_test = pd.read_csv(self.test_path)

        # Aseguramos que el target existe en df_train
        if self.target_col not in self.df_train.columns:
            raise ValueError(
                f"La columna objetivo '{self.target_col}' no existe en df_train.\n"
                f"Columnas disponibles: {self.df_train.columns.tolist()}"
            )

        print("✅ Datos cargados correctamente.")
        print(f" - df_train: {self.df_train.shape[0]} filas, {self.df_train.shape[1]} columnas")
        print(f" - df_test : {self.df_test.shape[0]} filas, {self.df_test.shape[1]} columnas")

    # ---------------------------------------------------------
    # 2) Construcción del preprocesador (numéricas + categóricas)
    # ---------------------------------------------------------
    def _build_preprocessor(self, X: pd.DataFrame):
        """
        Construye un ColumnTransformer que:
        - Escala las variables numéricas (StandardScaler o MinMaxScaler).
        - Aplica OneHotEncoder a las categóricas con handle_unknown='ignore'.
        """
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        print(f"📊 Columnas numéricas: {len(num_cols)}")
        print(f"🔡 Columnas categóricas: {len(cat_cols)}")

        # Elegir el tipo de scaler
        if self.scaler_type == "standard":
            scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type debe ser 'standard' o 'minmax'.")

        numeric_transformer = Pipeline(steps=[
            ("scaler", scaler)
        ])

        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ]
        )

        return preprocessor

    # ---------------------------------------------------------
    # 3) Preparar datos: separar target, hacer split train/val
    # ---------------------------------------------------------
    def prepare_data(self):
        """
        Separa la variable objetivo, construye preprocesador y realiza split train/validation.
        """
        if self.df_train is None:
            raise ValueError("Primero debes llamar a load_data().")

        # Separamos X (features) e y (target)
        X = self.df_train.drop(columns=[self.target_col])
        y = self.df_train[self.target_col]

        # Split train/validación estratificado por la variable objetivo
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        print(f"✂️ Split realizado: {self.X_train.shape[0]} train / {self.X_val.shape[0]} val")

        # Construimos preprocesador en base a X_train
        self.preprocessor = self._build_preprocessor(self.X_train)
        print("✅ Preprocesador construido.")

    # ---------------------------------------------------------
    # 4) Accesor para obtener todo listo para modelos
    # ---------------------------------------------------------
    def get_splits_and_preprocessor(self):
        """
        Devuelve los splits y el preprocesador listos para usar en ML:
        X_train, X_val, y_train, y_val, preprocessor
        """
        if any(v is None for v in [self.X_train, self.X_val, self.y_train, self.y_val, self.preprocessor]):
            raise ValueError("Debes llamar a prepare_data() antes de obtener los splits.")
        return self.X_train, self.X_val, self.y_train, self.y_val, self.preprocessor


