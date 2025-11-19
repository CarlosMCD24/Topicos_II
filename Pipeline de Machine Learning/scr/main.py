
# Se exportan los módulos necesarios y se definen las funciones principales
from module_data import WiDSDataModule
from module_ml import WiDSModelRunner
import pandas as pd

# Punto de entrada principal
def main(): 
    data_module = WiDSDataModule(
        data_dir=r"C:\\Users\\INIFAP-MOVIL\\Documents\\3 TERCER SEMESTRE\\Topicos II\\Trabajos\\Topicos_II\\Topicos_II\\Pipeline de Machine Learning\\data",
        train_filename="training.csv", # Nombres de archivos
        test_filename="test.csv", # Nombres de archivos
        target_col="DiagPeriodL90D", # Columna objetivo
        scaler_type="standard" # "standard" o "minmax"
    )
    # Se Cargan y preparan los datos
    data_module.load_data()
    data_module.prepare_data() 
    X_train, X_val, y_train, y_val, preprocessor = data_module.get_splits_and_preprocessor() # Obtener divisiones y preprocesador

    # Se ejecutan los experimentos de modelos
    model_runner = WiDSModelRunner() 
    results = model_runner.run_experiments( 
        preprocessor, X_train, y_train, X_val, y_val
    ) 

    print("\nRESUMEN FINAL")
    print(pd.DataFrame(results).T)
if __name__ == "__main__": # 
    main()


