import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, col, sum as spark_sum, avg
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.window import Window
import pyspark.sql.functions as F

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import when

import os
os.environ["PYSPARK_PYTHON"] = r"C:\Users\crist\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\crist\AppData\Local\Programs\Python\Python311\python.exe"

from pyspark.sql import SparkSession

#Codigo usando Python 3.11

#1. Carga y exploración de datos
#spark = SparkSession.builder.appName("Migraciones_SQL").getOrCreate()
#spark = SparkSession.builder.appName("Migraciones_SQL").config("spark.hadoop.hadoop.native.io", "false").getOrCreate()
spark = SparkSession.builder \
    .appName("Migraciones_SQL") \
    .config("spark.hadoop.hadoop.native.io", "false") \
    .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
    .config("spark.hadoop.fs.checksum.file", "false") \
    .getOrCreate()
sc = spark.sparkContext

# Carga el dataset proporcionado en Spark.
migraciones_df = spark.read.csv("migraciones.csv", header=True, inferSchema=True)
print("Spark OK:", spark.version)

# Convierte los datos en un RDD y un DataFrame.
#migraciones_rdd = sc.textFile("migraciones.csv")
migraciones_rdd = migraciones_df.rdd

# Explora los datos: muestra las primeras filas, el esquema y genera estadísticas descriptivas.
# Mostrar las primeras filas
print("Primeras filas del dataset:")
migraciones_df.show(10)

# Ver esquema (columnas y tipos de datos)
print("Esquema del DataFrame:")
migraciones_df.printSchema()

# Número de filas y columnas
print(f"Número de filas: {migraciones_df.count()}")
print(f"Número de columnas: {len(migraciones_df.columns)}")

# Mostrar estadísticas descriptivas de todas las columnas numéricas
print("Estadísticas descriptivas:")
migraciones_df.describe().show()

# Conteo de valores nulos por columna (opcional)
from pyspark.sql.functions import col, sum

nulos = migraciones_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in migraciones_df.columns])
print("Valores nulos por columna:")
nulos.show()

#Aplica transformaciones sobre los RDDs (filter, map, flatMap).
# Filtrar (filter)
# Ejemplo: quedarnos solo con las filas donde la columna "Año" sea mayor a 2015
migraciones_filtradas = migraciones_rdd.filter(lambda row: row["Año"] > 2015)
print("Filtrado (Año > 2015):")
print(migraciones_filtradas.take(5))

#  Map (transformar cada fila)
# Ejemplo: extraer solo País y Migrantes en una tupla
migraciones_mapeadas = migraciones_rdd.map(lambda row: (row["Origen"], row["Destino"]))
print("Map (Origen, Destino):")
print(migraciones_mapeadas.take(5))

# FlatMap (aplanar listas)
# Ejemplo: dividir el nombre de los países en palabras
paises_palabras = migraciones_rdd.flatMap(lambda row: row["Origen"].split(" "))
print("FlatMap (palabras de Países):")
print(paises_palabras.take(10))

# collect() → trae TODOS los elementos del RDD a una lista en Python
todos = migraciones_rdd.collect()
print("Collect (primeros 3 registros):")
print(todos[:3])   # ⚠ cuidado: si el dataset es muy grande puede colapsar la memoria

# take(n) → obtiene los primeros n elementos
primeros = migraciones_rdd.take(5)
print("Take (5 registros):")
for fila in primeros:
    print(fila)

# count() → cuenta el número de elementos en el RDD
total = migraciones_rdd.count()
print(f"Total de registros en el RDD: {total}")

#Realiza operaciones con DataFrames: filtrado, agregaciones y ordenamiento.
migraciones_map = migraciones_rdd.map(lambda row: (row["Origen"], row["PIB_Origen"]))
migraciones_map.take(5)  # Muestra los primeros 5 registros

migraciones_filtradas = migraciones_map.filter(lambda x: x[1] > 3000)
migraciones_filtradas.collect()


migraciones_agg = migraciones_map.reduceByKey(lambda a, b: a + b)
migraciones_agg.collect()

migraciones_ordenadas = migraciones_agg.sortBy(lambda x: x[1], ascending=False)
migraciones_ordenadas.collect()

# Definir esquema
schema = StructType([
    StructField("Origen", StringType(), True),
    StructField("PIB_Total", IntegerType(), True)
])

# Convertir RDD a DataFrame
migraciones_df_final = spark.createDataFrame(migraciones_ordenadas, schema=schema)
migraciones_df_final.show(5)

# Guardar el DataFrame final en formato Parquet
migraciones_df_final.write.mode("overwrite").parquet("migraciones_ordenadas.parquet")
#migraciones_df_final.write.mode("overwrite").csv("migraciones_ordenadas.csv", header=True)
# Registrar DataFrame final como tabla temporal
migraciones_df.createOrReplaceTempView("migraciones")

top_origen = spark.sql("""
    SELECT Origen, SUM(Población_Origen) AS Total_Migrantes
    FROM migraciones
    GROUP BY Origen
    ORDER BY Total_Migrantes DESC
    LIMIT 10
""")
top_origen.show()

top_destino = spark.sql("""
    SELECT Destino, SUM(Población_Destino) AS Total_Migrantes
    FROM migraciones
    GROUP BY Destino
    ORDER BY Total_Migrantes DESC
    LIMIT 10
""")
top_destino.show()

razones_origen = spark.sql("""
    SELECT Origen, Razón, COUNT(*) AS Cantidad
    FROM migraciones
    GROUP BY Origen, Razón
    ORDER BY Origen, Cantidad DESC
""")
razones_origen.show(20)

window = Window.partitionBy("Origen").orderBy(F.desc("Cantidad"))
razones_top_origen = razones_origen.withColumn("rank", F.row_number().over(window)).filter(F.col("rank") == 1)
razones_top_origen.show()


#Aplicación de MLlib para predicción de flujos migratorios

# Crear variable objetivo binaria
# Por ejemplo: migración "alta" si Población_Origen > 1000
migraciones_ml = migraciones_df.withColumn(
    "label",
    when(F.col("Población_Origen") > 1000, 1).otherwise(0)
)

# Codificar variables categóricas
indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
    for col in ["Origen", "Destino", "Razón", "Nivel_Educativo_Origen", "Nivel_Educativo_Destino"]
]

for indexer in indexers:
    migraciones_ml = indexer.fit(migraciones_ml).transform(migraciones_ml)

# Seleccionar columnas de características
feature_cols = [
    "PIB_Origen", "PIB_Destino",
    "Tasa_Desempleo_Origen", "Tasa_Desempleo_Destino",
    "Origen_idx", "Destino_idx", "Razón_idx",
    "Nivel_Educativo_Origen_idx", "Nivel_Educativo_Destino_idx"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
migraciones_ml = assembler.transform(migraciones_ml)

# Dividir en train y test
train_df, test_df = migraciones_ml.randomSplit([0.7, 0.3], seed=42)

# Ajustar el modelo de regresión logística
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_df)

# Predicciones
predictions = lr_model.transform(test_df)
predictions.select("label", "prediction", "probability").show(10)

# Evaluación del modelo
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"AUC del modelo: {auc:.3f}")

# Precisión
accuracy = predictions.filter(F.col("label") == F.col("prediction")).count() / predictions.count()
print(f"Precisión del modelo: {accuracy:.3f}")