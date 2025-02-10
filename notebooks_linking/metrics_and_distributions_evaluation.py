# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

from splink.spark.spark_linker import SparkLinker
import splink.spark.spark_comparison_library as cl
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import json
from pyspark.sql.functions import concat_ws
import Levenshtein as lv

# COMMAND ----------

# MAGIC %run ../parameters_linking

# COMMAND ----------

# MAGIC %run ../utils/eval_utils

# COMMAND ----------

# MAGIC %run ../utils/model_utils

# COMMAND ----------

# MAGIC %run ../utils/preprocessing_utils

# COMMAND ----------

# MAGIC %run ../utils/dataset_ingestion_utils

# COMMAND ----------

dbutils.widgets.text("params", "")
params_serialized = dbutils.widgets.get("params")
params_dict = json.loads(params_serialized)

params = Params(
    params_dict["integration_tests"],
    params_dict["comps_dict"],
    params_dict["model_hash"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest data

# COMMAND ----------

# Set database

display(spark.sql(f"USE DATABASE {params.DATABASE}"))

# COMMAND ----------

df_evaluation_predictions = spark.table(
    f"mps_enhancement_collab.{params.BEST_MATCH_TABLE}"
)

# COMMAND ----------

# fill nulls, so that comparisons for the truth table don't have to handle nulls

df_evaluation_predictions = df_evaluation_predictions.fillna(0)

df_evaluation_predictions.cache().count()

# COMMAND ----------

# display some example predictions

# note on NHS numbers and their meaning....

# NHS_NO_l_submitted is from the evaluation table. This is the NHS number submitted to MPS.
# NHS_NO_r_chosen_by_splink is from PDS, it is the NHS number chosen by Splink.

display(df_evaluation_predictions)

# COMMAND ----------

clean_up(params.DATABASE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distribution of probabilities

# COMMAND ----------

# Calculate and display distribution of the highest scoring match probability for each evaluation record.

display(
    df_evaluation_predictions.groupBy("match_weight")
    .count()
    .orderBy(F.col("match_weight").asc())
)