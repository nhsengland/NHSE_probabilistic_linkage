# Databricks notebook source
from splink.spark.spark_linker import SparkLinker
from datetime import datetime
from pyspark.sql import DataFrame, SparkSession
from typing import List
import json
import Levenshtein as lv

# COMMAND ----------

spark.udf.register('jaro_winkler_udf', lv.jaro_winkler)

# COMMAND ----------

# MAGIC %run ./parameters_dedupe

# COMMAND ----------

# MAGIC %run ./utils/model_utils

# COMMAND ----------

# MAGIC %run ./utils/dataset_ingestion_utils

# COMMAND ----------

# MAGIC %run ./utils/preprocessing_utils

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
# MAGIC # Defining the Linker

# COMMAND ----------

# Import data

df_pds_exploded = load_pds_full_or_exploded(params.DATABASE_PDS, params.TABLE_PDS, exploded=True)

df_pds_exploded = preprocess_all_demographics(df_pds_exploded,
                                  preprocess_postcode_args=params.PREPROCESS_POSTCODE_ARGS,
                                  preprocess_givenname_args=params.PREPROCESS_GIVEN_NAME_ARGS,
                                  preprocess_dob_args=params.PREPROCESS_DOB_ARGS,
                                  preprocess_familyname_args=params.PREPROCESS_FAMILY_NAME_ARGS,
                                  preprocess_fullname_args = params.PREPROCESS_FULL_NAME_ARGS
                                  )

clean_up(params.DATABASE)

# Define Linker
linker = SparkLinker(
  df_pds_exploded,
  get_model(params.DATABASE, params.TABLE_MODEL_RUNS, params.MODEL_DESCRIPTION),
  database=params.DATABASE,
  break_lineage_method='persist',
  register_udfs_automatically = False
)

# COMMAND ----------

# Export the match weights chart

display(
  spark.createDataFrame(
    [(json.dumps(linker.match_weights_chart().to_dict()), ), ],
    ['chart_spec_dict']
  )
)

# COMMAND ----------

print("Exploded PDS DataFrame Count: ", df_pds_exploded.cache().count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract out pairwise predictions

# COMMAND ----------

pairwise_predictions = linker.predict()

# COMMAND ----------

df_predictions = spark.table(pairwise_predictions.physical_name)
df_predictions = change_column_l_and_r_names(df_predictions)
df_predictions.cache().count()

# COMMAND ----------

df_predictions.write.mode('overwrite').saveAsTable(f'{params.DATABASE}.{params.PREDICTIONS_TABLE}')