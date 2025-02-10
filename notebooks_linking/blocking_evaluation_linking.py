# Databricks notebook source
import pyspark.sql.functions as F
from splink.spark.spark_linker import SparkLinker
import Levenshtein as lv

# COMMAND ----------

# MAGIC %run ../parameters_linking

# COMMAND ----------

# MAGIC %run ../utils/dataset_ingestion_utils

# COMMAND ----------

# MAGIC %run ../utils/preprocessing_utils

# COMMAND ----------

# MAGIC %run ../utils/model_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Linker Model

# COMMAND ----------

settings = {
    "link_type": "link_only",
    "unique_id_column_name": "UNIQUE_REFERENCE",
    "comparisons": params.BR_COMPARISONS,
    "blocking_rules_to_generate_predictions": params.BLOCKING_RULES,
}

# COMMAND ----------

df_training_subset = spark.table(f"{params.DATABASE}.{params.BR_TRAINING_TABLE}").select(
    "UNIQUE_REFERENCE",
    "GENDER",
    "GIVEN_NAME",
    "NHS_NO",
    "FAMILY_NAME",
    "DATE_OF_BIRTH",
    "POSTCODE",
    "LABEL",
)
df_pds_subset = spark.table(f"{params.DATABASE}.{params.BR_LINKING_TABLE}").select(
    "UNIQUE_REFERENCE",
    "GENDER",
    "GIVEN_NAME",
    "NHS_NO",
    "FAMILY_NAME",
    "DATE_OF_BIRTH",
    "POSTCODE",
    "LABEL",
)

# COMMAND ----------

df_training_subset = preprocess_all_demographics(df_training_subset)
df_pds_subset = preprocess_all_demographics(df_pds_subset)

# COMMAND ----------

linker = SparkLinker(
    [df_training_subset, df_pds_subset],
    settings,
    database=params.DATABASE,
    break_lineage_method="delta_lake_table",
    input_table_aliases=["_training", "_pds"],
    register_udfs_automatically=False,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the distribution of blocking rules

# COMMAND ----------

pairwise_predictions = linker.predict()
df_predictions = spark.table(pairwise_predictions.physical_name)
df_predictions.cache().count()

# COMMAND ----------

df_records_with_zero_candidates = df_training_subset.join(
    df_predictions,
    df_training_subset["unique_reference"] == df_predictions["unique_reference_r"],
    "left_anti",
)

df_candidates_per_training_record = (
    df_predictions.filter(F.col("match_weight").isNotNull())
    .groupBy("unique_reference_r")
    .count()
    .withColumn(
        "n_candidates_bin",
        F.when(F.col("count") <= 1, 1)
        .when(F.col("count") <= 2, 2)
        .when(F.col("count") <= 5, 5)
        .when(F.col("count") <= 10, 10)
        .when(F.col("count") <= 20, 20)
        .when(F.col("count") <= 50, 50)
        .when(F.col("count") <= 100, 100)
        .when(F.col("count") <= 200, 200)
        .when(F.col("count") <= 500, 500)
        .when(F.col("count") <= 1000, 1000)
        .otherwise(None),
    )
    .groupBy("n_candidates_bin")
    .count()
    .union(
        spark.createDataFrame(
            [("0", df_records_with_zero_candidates.count())],
            ["n_candidates_bin", "count"],
        )
    )
    .orderBy(F.col("n_candidates_bin").cast("int").asc_nulls_last())
)
display(df_candidates_per_training_record)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Records with zero candidates from PDS
# MAGIC If we think these records exist on PDS, our blocking rules are too tight

# COMMAND ----------

display(df_records_with_zero_candidates)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Records with only one candidate from PDS
# MAGIC
# MAGIC Ideally there shouldn't be many here, so that probabilistic linkage gets a chance to differentiate between candidates.

# COMMAND ----------

window_spec = Window.partitionBy("unique_reference_r")

df_with_count = df_predictions.withColumn("count", F.count("*").over(window_spec))

df_one_candidate = df_with_count.filter(F.col("count") == 1).drop("count")

display(df_one_candidate)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find number of new NHS number comparisons that each blocking rule generates if you have them in the given order

# COMMAND ----------

b_rules_simplified = [
    [t[2:] for t in rule.split() if t.startswith("l.")]
    for rule in params.BLOCKING_RULES
]

# COMMAND ----------

new_unique_nhs_no_comparisons = {}
df_predictions_decreased = df_predictions

window = Window.partitionBy([F.col("NHS_NO_l"), F.col("UNIQUE_REFERENCE_r")]).orderBy(
    F.col("match_weight")
)

for b_rule in b_rules_simplified:
    df_br_match = df_predictions_decreased

    for b_item in b_rule:
        df_br_match = df_br_match.filter(F.col(f"gamma_{b_item}") == 1)

    new_unique_nhs_no_comparisons[(", ".join(b_rule))] = (
        df_br_match.withColumn("row_number", F.row_number().over(window))
        .filter(F.col("row_number") == 1)
        .count()
    )

    df_predictions_decreased = df_predictions_decreased.exceptAll(df_br_match)

# COMMAND ----------

display(
    spark.createDataFrame(
        list(map(list, new_unique_nhs_no_comparisons.items())),
        ["rule", "number of unique NHS nums for comparisons"],
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### How many comparisons does each blocking rule add?
# MAGIC
# MAGIC This will tell us which blocking rules are redundant because they overlap with others, or which rules are generating too many comparisons.
# MAGIC
# MAGIC Note: The last column shows how many of the comparisons are additional to those already found. The order of the blocking rules makes a difference to this.

# COMMAND ----------

df_blocking_cumulative_counts = spark.createDataFrame(
    linker.cumulative_comparisons_from_blocking_rules_records()
).withColumnRenamed("row_count", "new_comparisons_count")

df_blocking_counts = (
    spark.createDataFrame(
        [
            {
                "rule": rule,
                "comparisons_count": linker.count_num_comparisons_from_blocking_rule(
                    rule
                ),
            }
            for rule in params.BLOCKING_RULES
        ]
    )
    .withColumn("sequence", F.monotonically_increasing_id())
    .join(df_blocking_cumulative_counts, "rule", "inner")
)

display(df_blocking_counts.orderBy("sequence").drop("sequence"))