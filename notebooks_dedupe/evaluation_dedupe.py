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

# MAGIC %run ../parameters_dedupe

# COMMAND ----------

# MAGIC %run ../utils/model_utils

# COMMAND ----------

# MAGIC %run ../utils/dataset_ingestion_utils

# COMMAND ----------

# MAGIC %run ../utils/preprocessing_utils

# COMMAND ----------

# MAGIC %run ../utils/eval_utils

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
# MAGIC ## Load Evaluation DataFrame and extract out all duplicate comparisons with their corresponding agreement patterns

# COMMAND ----------

df_evaluation = spark.table(f'{params.DATABASE}.{params.EVALUATION_TABLE}')
#  Defines the comparisons used for this evaluation
evaluation_comparisons = params.COMPARISONS_LIST
#  Creates the gamma patterns using the comparisons defined to get the agreement pattern for each duplicate identified.
gamma_pattern_df = create_gamma_pattern_df_from_linked_duplicates(df_evaluation, evaluation_comparisons)

# COMMAND ----------

gamma_pattern_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the model linker to be evaluated

# COMMAND ----------

try:
  clean_up(params.DATABASE)
except:
  print('Did not need to clean up')

# COMMAND ----------

# Define Linker
linker = SparkLinker(
  df_evaluation,
  get_model(params.DATABASE, params.TABLE_MODEL_RUNS, params.MODEL_DESCRIPTION),
  database=params.DATABASE,
  break_lineage_method='persist',
  register_udfs_automatically = False
)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract out pairwise predictions and join this dataset onto the evaluation dataset

# COMMAND ----------

pairwise_predictions = linker.predict()
df_predictions = spark.table(pairwise_predictions.physical_name)
df_predictions = change_column_l_and_r_names(df_predictions)
df_predictions = add_agreement_pattern(df_predictions)

# COMMAND ----------

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

combine_df = combine_predictions_and_gamma_pattern_df(df_predictions, gamma_pattern_df, df_evaluation, thresholds)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate agreement patterns that have avoided the blocking rules defined

# COMMAND ----------

missing_duplicates = combine_df.filter(F.col("missed_duplicate") == 1)
missing_duplicates.count()

# COMMAND ----------

gamma_patterns_of_missing_duplicates = missing_duplicates.groupby("all_agreement_patterns").count()
display(gamma_patterns_of_missing_duplicates)

# COMMAND ----------

top_missing_gamma_patterns = categorise_by_count_and_aggregate(gamma_patterns_of_missing_duplicates, column_name="count", agg_column="all_agreement_patterns", thresholds=[2, 3, 5, 10, 50, 100, 500, 1000, 5000, 10000], exact_values=[1])

# COMMAND ----------

display(top_missing_gamma_patterns.drop("threshold_sort"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate TP, FP, TN, FN values as threshold for a range of match probabilities 

# COMMAND ----------

#  This can take a long time to run ~ 40 minutes - so run with caution of this.
if not params.integration_tests:
  confusion_matrix_per_threshold = create_confusion_matrix_per_threshold(combine_df, thresholds)

# COMMAND ----------

if not params.integration_tests:
  confusion_matrix_per_threshold 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate TP, FP, TN, FN values for a given match probability threshold

# COMMAND ----------

threshold = 0.5
thresh_str = "_".join(str(threshold).split("."))

tp_df = combine_df.filter(F.col(f"above_threshold_{thresh_str}") & F.col("true_duplicate"))
fp_df = combine_df.filter(F.col(f"above_threshold_{thresh_str}") & ~F.col("true_duplicate"))
tn_df = combine_df.filter(F.col(f"below_threshold_{thresh_str}") & ~F.col("true_duplicate"))
fn_df = combine_df.filter(F.col(f"below_threshold_{thresh_str}") & F.col("true_duplicate"))

# COMMAND ----------

match_probability_per_pattern = combine_df.select(["match_probability", "agreement_patterns"]).drop_duplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC ### TP Evaluation

# COMMAND ----------

tp_evaluation = tp_df.groupby("agreement_patterns").count()
tp_evaluation = tp_evaluation.join(match_probability_per_pattern, on="agreement_patterns", how="left")
display(tp_evaluation)

# COMMAND ----------

# MAGIC %md
# MAGIC ### FP Evaluation

# COMMAND ----------

fp_evaluation = fp_df.groupby("agreement_patterns").count()
fp_evaluation = fp_evaluation.join(match_probability_per_pattern, on="agreement_patterns", how="left")
display(fp_evaluation)

# COMMAND ----------

# MAGIC %md
# MAGIC ### TN Evaluation

# COMMAND ----------

tn_evaluation = tn_df.groupby("agreement_patterns").count()
tn_evaluation = tn_evaluation.join(match_probability_per_pattern, on="agreement_patterns", how="left")
display(tn_evaluation)

# COMMAND ----------

# MAGIC %md
# MAGIC ### FN Evaluation

# COMMAND ----------

fn_evaluation = fn_df.groupby("agreement_patterns").count()
fn_evaluation = fn_evaluation.join(match_probability_per_pattern, on="agreement_patterns", how="left")
display(fn_evaluation)

# COMMAND ----------

gamma_pattern_str= "0651"
display(fn_df.filter(F.col("agreement_patterns") == gamma_pattern_str))

# COMMAND ----------

