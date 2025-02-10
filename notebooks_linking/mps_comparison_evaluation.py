# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluation
# MAGIC
# MAGIC The training data contains gold and silver labels.
# MAGIC
# MAGIC According to the gold and silver labels, you will generate counts for a confusion matrix.
# MAGIC
# MAGIC Gold label is where MPS found a match at the cross check trace step. This means that the NHS number and Date Of Birth matched exactly to a PDS record. In this case we are very confident that the link chosen by MPS was correct.
# MAGIC
# MAGIC Silver label is where MPS found a match at the alphanumeric or algorithmic trace step. This is a looser match so we are less confident that the link found by MPS was correct. When Splink chose a PDS record which is different to the PDS record chosen by MPS, this is very interesting! We will clerically review these cases and consider which we think is the better link.

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

spark.udf.register("jaro_winkler_udf", lv.jaro_winkler)

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
    "LABEL",
)

df_evaluation_label_types = df_training_subset_for_eval.select(
    "UNIQUE_REFERENCE", "LABEL_TYPE", "LABEL"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run model

# COMMAND ----------

# For each evaluation record, get the label and label type, and join it to the highest scoring prediction from the model.
df_evaluation_predictions = add_agreement_pattern(spark.table(f"{params.DATABASE}.{params.BEST_MATCH_TABLE}")).drop("LABEL")

df_evaluation_predictions = df_evaluation_label_types.join(
    df_evaluation_predictions,
    df_evaluation_label_types["UNIQUE_REFERENCE"] == df_evaluation_predictions["UNIQUE_REFERENCE_other_table"],
    "left",
)


# COMMAND ----------

assert df_evaluation.count() == df_evaluation_predictions.count()

# COMMAND ----------

# fill nulls, so that comparisons for the truth table don't have to handle nulls

df_evaluation_predictions = df_evaluation_predictions.fillna(0).fillna("no label", "LABEL_TYPE")

df_evaluation_predictions.cache().count()

# COMMAND ----------

# display some example predictions

# note on NHS numbers and their meaning....

# NHS_NO_l_submitted is from the evaluation table. This is the NHS number submitted to MPS.
# NHS_NO_r_chosen_by_splink is from PDS, it is the NHS number chosen by Splink.
# LABEL is the NHS number returned by MPS.

display(df_evaluation_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add MPS responses for comparison

# COMMAND ----------

# When clerically reviewing the possible false positive, we should join to the relevant record in the MPS response table.
# Then we can review the demographics of the link found by Splink, alongside the demographics of the link found by PDS, to decide which is better.
df_pds_replaced_by = spark.table("pds.replaced_by")

# filter and remove duplicates from mps_responses
# TODO: this doesn't necessarily keep the same record as was chosen in select_training_data, as we didn't keep the local_id.
# There are 11 records in our current evaluation data with dupicate unique_reference, fortunately they all have the same features.
# But if we change the evaluation data we should be more careful about this.
# Also, it is not good to have hard-coded the dataset_id here. Again, if we keep the local_id in select_training_data, we could filter on this instead.
df_mps_responses = (
    spark.table("mps_archive.responses")
    .filter(F.col("dataset_id").startswith("n"))
    .dropDuplicates(["unique_reference"])
)

df_mps_responses = load_mps_responses(df_mps_responses, df_pds_replaced_by)

# Filter df_mps_responses to keep only the records that are in the evaluation dataset
df_mps_responses = df_mps_responses.join(
    df_evaluation_predictions,
    df_evaluation_predictions["UNIQUE_REFERENCE_other_table"] == df_mps_responses["UNIQUE_REFERENCE"],
    "left_semi",
)

# COMMAND ----------

# We will use the distance metrics from our linker model to compare mps request features to mps response features.
# We only need to compare the single pair of records, not several candidate records. So replace the blocking rules with a single blocking rule to force this.

linker_settings_for_mps_request_response = get_model(params.DATABASE, params.TABLE_MODEL_RUNS, params.MODEL_DESCRIPTION)
linker_settings_for_mps_request_response["blocking_rules_to_generate_predictions"] = [{"blocking_rule": "l.UNIQUE_REFERENCE = r.UNIQUE_REFERENCE", "sql_dialect": None}]

# COMMAND ----------

# Pre-process mps responses so that they are ready to be compared with our distance metrics

df_mps_responses_for_linker = df_mps_responses.select(
    "UNIQUE_REFERENCE",
    F.col("GENDER_mps").alias("GENDER"),
    F.col("GIVEN_NAME_mps").alias("GIVEN_NAME"),
    F.col("NHS_NO_mps").alias("NHS_NO"),
    F.col("FAMILY_NAME_mps").alias("FAMILY_NAME"),
    F.col("DATE_OF_BIRTH_mps").alias("DATE_OF_BIRTH"),
    F.col("POSTCODE_mps").alias("POSTCODE"),
)

df_mps_responses_for_linker = preprocess_all_demographics(df_mps_responses_for_linker,
                                  preprocess_postcode_args=params.PREPROCESS_POSTCODE_ARGS,
                                  preprocess_givenname_args=params.PREPROCESS_GIVEN_NAME_ARGS,
                                  preprocess_dob_args=params.PREPROCESS_DOB_ARGS,
                                  preprocess_familyname_args=params.PREPROCESS_FAMILY_NAME_ARGS,
                                  preprocess_fullname_args = params.PREPROCESS_FULL_NAME_ARGS
                                                         )

df_mps_responses_for_linker.cache().count()

# COMMAND ----------

# Pre-process the mps requests so that they are ready to be compared with our distance metrics
# We already have the mps requests, in df_evaluation.

df_mps_requests_for_linker = df_evaluation.drop("LABEL")
df_mps_requests_for_linker = preprocess_all_demographics(df_mps_requests_for_linker,
                                  preprocess_postcode_args=params.PREPROCESS_POSTCODE_ARGS,
                                  preprocess_givenname_args=params.PREPROCESS_GIVEN_NAME_ARGS,
                                  preprocess_dob_args=params.PREPROCESS_DOB_ARGS,
                                  preprocess_familyname_args=params.PREPROCESS_FAMILY_NAME_ARGS,
                                  preprocess_fullname_args = params.PREPROCESS_FULL_NAME_ARGS
                                                         )

df_mps_requests_for_linker.cache().count()

# COMMAND ----------

clean_up(params.DATABASE)

# COMMAND ----------

# run the linker

linker_for_mps_request_response = SparkLinker(
    [df_mps_responses_for_linker, df_mps_requests_for_linker],
    linker_settings_for_mps_request_response,
    database=params.DATABASE,
    break_lineage_method="persist",
    input_table_aliases=["a_pds", "b_other_table"],
    register_udfs_automatically=False,
)

pairwise_predictions_mps = linker_for_mps_request_response.predict()
df_predictions_mps = spark.table(pairwise_predictions_mps.physical_name)

# COMMAND ----------

# all we want from the linker output is the agreement patterns
df_predictions_mps = add_agreement_pattern(df_predictions_mps).select(
    "UNIQUE_REFERENCE_l", "agreement_patterns"
)

# which we then join onto df_mps_responses
df_mps_responses = (
    df_mps_responses.join(
        df_predictions_mps,
        [
            df_mps_responses["UNIQUE_REFERENCE"]
            == df_predictions_mps["UNIQUE_REFERENCE_l"]
        ],
        "left",
    )
    .drop("UNIQUE_REFERENCE_l")
    .withColumnRenamed("agreement_patterns", "mps_agreement_pattern")
)

# COMMAND ----------

df_with_mps = join_with_mps_responses(
    df_evaluation_predictions.drop("UNIQUE_REFERENCE"), df_mps_responses
)
display(df_with_mps)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix counts

# COMMAND ----------

# Where the splink result agrees with a gold or silver label, this is a true positive
df_confusion_matrix_with_mps = add_confusion_matrix_to_df(
    df_with_mps, params.MATCH_WEIGHT_THRESHOLD
)

# COMMAND ----------

df_false_positives = filter_confusion_matrix(
    df_confusion_matrix_with_mps, "false_positives"
)
display(df_false_positives)

# COMMAND ----------

display(
    df_false_positives.groupBy(["agreement_patterns", "mps_agreement_pattern"]).count()
)

# COMMAND ----------

df_possible_false_positives = filter_confusion_matrix(
    df_confusion_matrix_with_mps, "possible_false_positives"
)
display(df_possible_false_positives)

# COMMAND ----------

display(
    df_possible_false_positives.filter(F.col("LABEL_TYPE") == "SILVER")
    .groupBy(["agreement_patterns", "mps_agreement_pattern"])
    .count()
)

# COMMAND ----------

display(
    df_possible_false_positives.filter(F.col("LABEL_TYPE") == "no label")
    .filter(F.col("mps_multiple_pds_matches") == True)
    .groupBy(["agreement_patterns"])
    .count()
)

# COMMAND ----------

display(
    df_possible_false_positives.filter(F.col("LABEL_TYPE") == "no label")
    .filter(F.col("mps_multiple_pds_matches") == False)
    .groupBy(["agreement_patterns"])
    .count()
)

# COMMAND ----------

# For the possible false negatives, we should review with the demographics of the link found by splink, alongside the demographics of the link found by MPS, to decide which is better.

df_possible_false_negatives = filter_confusion_matrix(
    df_confusion_matrix_with_mps, "possible_false_negatives"
)
display(df_possible_false_negatives)

# COMMAND ----------

display(df_possible_false_negatives.groupBy(F.col("splink_close_match")).count())

# COMMAND ----------

display(
    df_possible_false_negatives.filter(F.col("splink_close_match") == False)
    .groupBy(["agreement_patterns", "mps_agreement_pattern"])
    .count()
)

# COMMAND ----------


display(
    df_possible_false_negatives.filter(F.col("splink_close_match") == True)
    .groupBy(["agreement_patterns", "mps_agreement_pattern"])
    .count()
)

# COMMAND ----------

df_false_negatives = filter_confusion_matrix(
    df_confusion_matrix_with_mps, "false_negatives"
)
display(df_false_negatives)

# COMMAND ----------

display(
    df_false_negatives.groupBy("agreement_patterns", "mps_agreement_pattern").count()
)

# COMMAND ----------

# It could be worthwhile investigating true negatives too.
# Why was there not a link to PDS? Was there insufficient demographic data to make a link?

df_true_negatives = filter_confusion_matrix(
    df_confusion_matrix_with_mps, "true_negatives"
)
display(df_true_negatives)

# COMMAND ----------

# for stats purposes it is useful to have a count of the number of true positives
df_true_positives = filter_confusion_matrix(
    df_confusion_matrix_with_mps, "true_positives"
)
print(df_true_positives.count())

# COMMAND ----------

# analyse records that dont get compared in splink at all
df_no_splink_comparisons = filter_confusion_matrix(
    df_confusion_matrix_with_mps, "no_splink_comparisons"
)
display(df_no_splink_comparisons)
