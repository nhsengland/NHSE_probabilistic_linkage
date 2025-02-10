# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Training linking

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Import libraries

# COMMAND ----------

from splink.spark.spark_linker import SparkLinker
import splink.spark.spark_comparison_library as cl
import pyspark.sql.functions as F
from datetime import datetime
import json
import pandas as pd
import statistics
from pyspark.sql import SparkSession
import Levenshtein as lv

# COMMAND ----------

spark.udf.register("jaro_winkler_udf", lv.jaro_winkler)
# spark.udf.register('damerau_levenshtein_udf', damerau_levenshtein_as_int)

# COMMAND ----------

# MAGIC %run ../utils/model_utils

# COMMAND ----------

# MAGIC %run ../utils/preprocessing_utils

# COMMAND ----------

# MAGIC %run ../utils/dataset_ingestion_utils

# COMMAND ----------

# MAGIC %run ../parameters_linking

# COMMAND ----------

dbutils.widgets.text("params", "")
params_serialized = dbutils.widgets.get("params")
params_dict = json.loads(params_serialized)

# Recreate Params instance
params = Params(
    params_dict["integration_tests"],
    params_dict["comps_dict"],
    params_dict["model_hash"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Initialise linker

# COMMAND ----------

# Define linker settings
# Import data
df_pds_full = spark.table(f"{params.DATABASE}.{params.TRAINING_PDS_TABLE}")
df_training = spark.table(f"{params.DATABASE}.{params.TRAINING_LINKING_TABLE}")

df_pds_full = preprocess_all_demographics(df_pds_full,
                                  preprocess_postcode_args=params.PREPROCESS_POSTCODE_ARGS,
                                  preprocess_givenname_args=params.PREPROCESS_GIVEN_NAME_ARGS,
                                  preprocess_dob_args=params.PREPROCESS_DOB_ARGS,
                                  preprocess_familyname_args=params.PREPROCESS_FAMILY_NAME_ARGS,
                                  preprocess_fullname_args = params.PREPROCESS_FULL_NAME_ARGS
                                  )
df_training = preprocess_all_demographics(df_training, 
                                  preprocess_postcode_args=params.PREPROCESS_POSTCODE_ARGS,
                                  preprocess_givenname_args=params.PREPROCESS_GIVEN_NAME_ARGS,
                                  preprocess_dob_args=params.PREPROCESS_DOB_ARGS,
                                  preprocess_familyname_args=params.PREPROCESS_FAMILY_NAME_ARGS,
                                  preprocess_fullname_args = params.PREPROCESS_FULL_NAME_ARGS
                                  )

try:
    clean_up(params.DATABASE)
except:
    print("Didn't need to clean up")

# first the model is set up for PDS only, so that the U values will be trained from PDS only
settings = {
    "probability_two_random_records_match": params.PROPORTION_OF_LINKS_EXPECTED
    / df_pds_full.count(),
    "link_type": "dedupe_only",
    "unique_id_column_name": params.TRAINING_UNIQUE_REFERENCE_COLUMN,
    "comparisons": params.COMPARISONS_LIST,
    "blocking_rules_to_generate_predictions": params.BLOCKING_RULES,
}

linker = SparkLinker(
    df_pds_full,
    settings,
    database=params.DATABASE,
    break_lineage_method="delta_lake_table",
    input_table_aliases=["a_pds"],
    register_udfs_automatically=False,
)

# COMMAND ----------

# Estimate u probablities
linker.estimate_u_using_random_sampling(max_pairs=1e8)

# save the model and change its settings for a link task
model_with_u_values = linker.save_model_to_json()
model_with_u_values.update({"link_type": "link_only"})

# COMMAND ----------

# train several sets of m values, each with different blocking rules for training
# then we will take the average m value
models_with_u_and_m_values = []

for blocking_rule in params.BLOCKING_RULES_FOR_TRAINING:
    try:
        clean_up(params.DATABASE)
    except:
        print("Didn't need to clean up")

    linker = SparkLinker(
        [df_pds_full, df_training],
        model_with_u_values,
        database=params.DATABASE,
        break_lineage_method="delta_lake_table",
        input_table_aliases=["a_pds", "b_other_table"],
        register_udfs_automatically=False,
    )

    # Estimate m probability for the blocking rules
    linker.estimate_parameters_using_expectation_maximisation(blocking_rule)
    models_with_u_and_m_values.append(linker.save_model_to_json())

# COMMAND ----------

# MAGIC %md
# MAGIC **Get the average m values**

# COMMAND ----------

# convert the comparisons dictionary to a dataframe for each feature
df_comparisons = pd.DataFrame.from_dict(model_with_u_values["comparisons"])

# COMMAND ----------

# for each comparison, the get_average_m_values_from_models function loops through the comparison levels to get the M values, and then averages them
dict_comparison_levels_name = get_average_m_values_from_models(
    "NAME", df_comparisons, models_with_u_and_m_values
)
dict_comparison_levels_date_of_birth = get_average_m_values_from_models(
    "DATE_OF_BIRTH", df_comparisons, models_with_u_and_m_values
)
dict_comparison_levels_postcode = get_average_m_values_from_models(
    "POSTCODE", df_comparisons, models_with_u_and_m_values
)
dict_comparison_levels_gender = get_average_m_values_from_models(
    "GENDER", df_comparisons, models_with_u_and_m_values
)

# COMMAND ----------

# reconstruct the comparisons dictionary from the separate dictionaries for each feature.
model_with_u_and_average_m_values = model_with_u_values

new_comparisons = [
    {
        "output_column_name": "NAME",
        "comparison_levels": dict_comparison_levels_name,
        "comparison_description": "Name comparisons",
    },
    {
        "output_column_name": "DATE_OF_BIRTH",
        "comparison_levels": dict_comparison_levels_date_of_birth,
        "comparison_description": "Date of birth comparisons",
    },
    {
        "output_column_name": "POSTCODE",
        "comparison_levels": dict_comparison_levels_postcode,
        "comparison_description": "Postcode comparisons",
    },
    {
        "output_column_name": "GENDER",
        "comparison_levels": dict_comparison_levels_gender,
        "comparison_description": "Gender comparisons",
    },
]

model_with_u_and_average_m_values.update({"comparisons": new_comparisons})

# COMMAND ----------

# save intermediate calculations
model_with_u_and_average_m_values["retain_intermediate_calculation_columns"] = True
model_with_u_and_average_m_values["retain_matching_columns"] = True

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Save model

# COMMAND ----------

#display(get_m_and_u_probabilities(model_with_u_and_average_m_values))

# COMMAND ----------

save_model(
    model_with_u_and_average_m_values,
    params.MODEL_DESCRIPTION,
    params.DATABASE,
    params.TABLE_MODEL_RUNS,
)