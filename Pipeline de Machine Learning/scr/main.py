"""
main.py
-------
Script principal que:
- Carga y prepara los datos con WiDSDataModule.
- Ejecuta experimentos con varios modelos usando WiDSModelRunner.
- Muestra un resumen de métricas para comparar modelos.
"""

from module_data import WiDSDataModule
from module_ml import WiDSModelRunner
import pandas as pd



def main():
    # 1) Preparar datos
    data_module = WiDSDataModule(
        data_dir="data",
        train_filename="training.csv",
        test_filename="test.csv",
        target_col="DiagPeriodL90D",  # 👈 Ajusta si tu target tiene otro nombre
        scaler_type="standard",       # "standard" o "minmax"
        test_size=0.2,
        random_state=42,
    )

    data_module.load_data()
    data_module.prepare_data()
    X_train, X_val, y_train, y_val, preprocessor = data_module.get_splits_and_preprocessor()

    # 2) Correr experimentos de modelos
    model_runner = WiDSModelRunner(experiment_name="WiDS_2024_Experiments")
    results = model_runner.run_experiments(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    # 3) Resumen final en forma de tabla
    print("\n================= RESUMEN DE MODELOS =================")
    df_results = pd.DataFrame(results).T  # modelos en filas
    print(df_results.sort_values(by="accuracy", ascending=False))


if __name__ == "__main__":
    main()
