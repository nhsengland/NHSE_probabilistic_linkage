# Databricks notebook source
from splink.spark.spark_linker import SparkLinker
from datetime import datetime
from pyspark.sql import DataFrame, SparkSession
from typing import List
import json
import Levenshtein as lv

# COMMAND ----------

spark.udf.register("jaro_winkler_udf", lv.jaro_winkler)

# COMMAND ----------

# MAGIC %run ./utils/model_utils

# COMMAND ----------

# MAGIC %run ./utils/dataset_ingestion_utils

# COMMAND ----------

# MAGIC %run ./utils/preprocessing_utils

# COMMAND ----------

# MAGIC %run ./parameters_linking

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

# Define linker settings
# Import data
df_pds_exploded = load_pds_full_or_exploded(
    params.DATABASE_PDS, params.TABLE_PDS, exploded=True
)
df_evaluation = load_data_to_link(
    params.DATABASE,
    params.LINKING_TABLE,
    params.TRAINING_GIVEN_NAME_COLUMN,
    params.TRAINING_FAMILY_NAME_COLUMN,
    params.TRAINING_GENDER_COLUMN,
    params.TRAINING_POSTCODE_COLUMN,
    params.TRAINING_DOB_COLUMN,
    params.TRAINING_UNIQUE_REFERENCE_COLUMN,
).select(
    "UNIQUE_REFERENCE",
    "GENDER",
    "GIVEN_NAME",
    "NHS_NO",
    "FAMILY_NAME",
    "DATE_OF_BIRTH",
    "POSTCODE",
    "LABEL",
)

df_pds_exploded = preprocess_all_demographics(df_pds_exploded,
                                  preprocess_postcode_args=params.PREPROCESS_POSTCODE_ARGS,
                                  preprocess_givenname_args=params.PREPROCESS_GIVEN_NAME_ARGS,
                                  preprocess_dob_args=params.PREPROCESS_DOB_ARGS,
                                  preprocess_familyname_args=params.PREPROCESS_FAMILY_NAME_ARGS,
                                  preprocess_fullname_args = params.PREPROCESS_FULL_NAME_ARGS
                                  )
df_evaluation = preprocess_all_demographics(df_evaluation, 
                                  preprocess_postcode_args=params.PREPROCESS_POSTCODE_ARGS,
                                  preprocess_givenname_args=params.PREPROCESS_GIVEN_NAME_ARGS,
                                  preprocess_dob_args=params.PREPROCESS_DOB_ARGS,
                                  preprocess_familyname_args=params.PREPROCESS_FAMILY_NAME_ARGS,
                                  preprocess_fullname_args = params.PREPROCESS_FULL_NAME_ARGS
                                  )

clean_up(params.DATABASE)

linker = SparkLinker(
    [df_pds_exploded, df_evaluation],
    get_model(params.DATABASE, params.TABLE_MODEL_RUNS, params.MODEL_DESCRIPTION),
    database=params.DATABASE,
    break_lineage_method="persist",
    input_table_aliases=["a_pds", "b_other_table"],
    register_udfs_automatically=False,
)

# COMMAND ----------

# Export the match weights chart

display(
    spark.createDataFrame(
        [
            (json.dumps(linker.match_weights_chart().to_dict()),),
        ],
        ["chart_spec_dict"],
    )
)

# COMMAND ----------

print("Evaluation DataFrame Count: ", df_evaluation.cache().count())
print("Exploded PDS DataFrame Count: ", df_pds_exploded.cache().count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract out pairwise predictions

# COMMAND ----------

pairwise_predictions = linker.predict()

# COMMAND ----------

df_predictions = spark.table(pairwise_predictions.physical_name)
df_predictions = change_column_l_and_r_names(df_predictions)

# COMMAND ----------

df_predictions.cache().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract out pairwise predictions

# COMMAND ----------

df_candidate_match_probabilities = match_probabilities_output(df_predictions, df_evaluation)

# COMMAND ----------

df_candidate_match_probabilities.write.mode("overwrite").saveAsTable(f"{params.DATABASE}.{params.MATCH_PROBABILITIES_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Return the best match for each individual

# COMMAND ----------

df_best_matches = get_best_match(df_candidate_match_probabilities, params.CLOSE_MATCHES_THRESHOLD, params.MATCH_WEIGHT_THRESHOLD)

# COMMAND ----------

df_best_matches.write.mode("overwrite").saveAsTable(f"{params.DATABASE}.{params.BEST_MATCH_TABLE}")