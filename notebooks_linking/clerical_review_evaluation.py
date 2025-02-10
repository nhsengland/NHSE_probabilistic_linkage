# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluation with Clerically Reviewed Records

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

# This takes 100,000 records from the training set to evaluate on.

df_training_subset_for_eval = load_data_to_link(
    params.DATABASE,
    params.LINKING_TABLE,
    params.TRAINING_GIVEN_NAME_COLUMN,
    params.TRAINING_FAMILY_NAME_COLUMN,
    params.TRAINING_GENDER_COLUMN,
    params.TRAINING_POSTCODE_COLUMN,
    params.TRAINING_DOB_COLUMN,
    params.TRAINING_UNIQUE_REFERENCE_COLUMN,
).dropDuplicates()

df_evaluation = df_training_subset_for_eval.select(
    "UNIQUE_REFERENCE",
    "GENDER",
    "GIVEN_NAME",
    "NHS_NO",
    "FAMILY_NAME",
    "DATE_OF_BIRTH",
    "POSTCODE",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run model

# COMMAND ----------

# Load predictions, including records that didnt have any matches in the blocking rules.

df_predictions = spark.table(f"mps_enhancement_collab.{params.MATCH_PROBABILITIES_TABLE}").drop("LABEL")
print(df_predictions.count())

# COMMAND ----------

df_predictions = add_agreement_pattern(df_predictions)

# Show the updated DataFrame
print(df_predictions.count())

# COMMAND ----------

# For each evaluation record, get the label and label type, and join it to the highest scoring prediction from the model.
df_evaluation_predictions = spark.table(f"mps_enhancement_collab.{params.BEST_MATCH_TABLE}")

# COMMAND ----------

assert df_evaluation.count() == df_evaluation_predictions.count()

# COMMAND ----------

# fill nulls, so that comparisons for the truth table don't have to handle nulls

df_evaluation_predictions = df_evaluation_predictions.fillna(0)

df_evaluation_predictions.cache().count()

# COMMAND ----------

# display some example predictions

# note on NHS numbers and their meaning....

# NHS_NO_l_submitted is from the evaluation table. This is the NHS number submitted to MPS.
# NHS_NO_r_chosen_by_splink is from PDS, it is the NHS number chosen by Splink.
df_evaluation_predictions = add_agreement_pattern(df_evaluation_predictions)
display(df_evaluation_predictions)

# COMMAND ----------

clean_up(params.DATABASE)

# COMMAND ----------

# MAGIC %md
# MAGIC # Confusion Matrix for Labelled Data

# COMMAND ----------

clerical_labels = spark.table("mps_enhancement_collab.clerical_review_labels")

clerical_cf_df = create_clerical_review_confusion_matrix(
    df_evaluation_predictions, clerical_labels
)
display(clerical_cf_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix counts

# COMMAND ----------

df_false_positives = filter_confusion_matrix(clerical_cf_df, "FP")
display(df_false_positives)

# COMMAND ----------

display(df_false_positives.groupBy(["agreement_patterns"]).count())

# COMMAND ----------

df_false_negatives = filter_confusion_matrix(clerical_cf_df, "FN")
display(df_false_negatives)

# COMMAND ----------

display(df_false_negatives.groupBy("agreement_patterns").count())

# COMMAND ----------

# It could be worthwhile investigating true negatives too.
# Why was there not a link to PDS? Was there insufficient demographic data to make a link?

df_true_negatives = filter_confusion_matrix(clerical_cf_df, "TN")
display(df_true_negatives)

# COMMAND ----------

# for stats purposes it is useful to have a count of the number of true positives
df_true_positives = filter_confusion_matrix(clerical_cf_df, "TP")
print(df_true_positives.count())

# COMMAND ----------

# analyse records that dont get compared in splink at all
df_no_splink_comparisons = filter_confusion_matrix(
    clerical_cf_df, "no_splink_comparisons"
)
display(df_no_splink_comparisons)