
# importamos las librerias necesarias
import random, datetime, os, shutil, math

# definimos los directorios de entrenamiento y prueba
train_dir =r"C:/Users/INIFAP-MOVIL/Documents/3 TERCER SEMESTRE/Topicos II/Trabajos/Topicos_II/Topicos_II/Pokemon_Clasificacion/data/train"
test_dir =r"C:/Users/INIFAP-MOVIL/Documents/3 TERCER SEMESTRE/Topicos II/Trabajos/Topicos_II/Topicos_II/Pokemon_Clasificacion/data/test"

# funcion para preparar los datos de prueba
def prep_test_data(pokemon, train_dir, test_dir):
    pop = os.listdir(train_dir+'/'+pokemon)
    test_data=random.sample(pop, 15)
    print(test_data)
    for f in test_data:
        shutil.copy(train_dir+'/'+pokemon+'/'+f, test_dir+'/'+pokemon+'/')

# iteramos sobre cada pokemon en el directorio de entrenamiento
for poke in os.listdir(train_dir):
    os.makedirs(test_dir+'/'+poke, exist_ok=True)
    prep_test_data(poke, train_dir, test_dir)
    print(f"Se han copiado las imagenes de {poke} correctamente")