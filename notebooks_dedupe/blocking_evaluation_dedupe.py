# Databricks notebook source
import pyspark.sql.functions as F
from splink.spark.spark_linker import SparkLinker
import Levenshtein as lv


# COMMAND ----------

spark.udf.register('jaro_winkler_udf', lv.jaro_winkler)

# COMMAND ----------

# MAGIC %run ../parameters_dedupe

# COMMAND ----------

# MAGIC %run ../utils/dataset_ingestion_utils

# COMMAND ----------

# MAGIC %run ../utils/preprocessing_utils

# COMMAND ----------

# MAGIC %run ../utils/model_utils

# COMMAND ----------

link_only=False

# COMMAND ----------

# MAGIC %md
# MAGIC # Subsample Blocking Rule Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Linker Model

# COMMAND ----------

settings = {
  'link_type': 'dedupe_only',
  'unique_id_column_name': params.TRAINING_UNIQUE_REFERENCE_COLUMN,
  'comparisons': params.BR_COMPARISONS,
  'blocking_rules_to_generate_predictions': params.BLOCKING_RULES
}

# COMMAND ----------

df_pds_subsample = (spark.table(f'{DATABASE}.{BR_TRAINING_TABLE}'))

# COMMAND ----------

# Instantiate linker
linker = SparkLinker(
  df_pds_subsample,
  settings,
  database = params.DATABASE,
  break_lineage_method = 'persist',
  register_udfs_automatically = False
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluating Blocking Rules for subsample dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extracting out prediction count per individual

# COMMAND ----------

pairwise_predictions = linker.predict()
df_predictions = spark.table(pairwise_predictions.physical_name)

# COMMAND ----------

df_records_with_zero_candidates = (
  df_pds_subsample
  .join(
    df_predictions, 
    df_pds_subsample['unique_reference'] == df_predictions['unique_reference_r'], 
    'left_anti'
  )
)

df_candidates_per_training_record = (
  df_predictions
  .filter(F.col('match_probability').isNotNull())
  .groupBy('unique_reference_r').count()
  .withColumn('n_candidates_bin', 
              F.when(F.col('count') <= 1, 1)
               .when(F.col('count') <= 2, 2)
               .when(F.col('count') <= 5, 5)
               .when(F.col('count') <= 10, 10)
               .when(F.col('count') <= 20, 20)
               .when(F.col('count') <= 50, 50)
               .when(F.col('count') <= 100, 100)
               .when(F.col('count') <= 200, 200)
               .when(F.col('count') <= 500, 500)
               .when(F.col('count') <= 1000, 1000)
               .otherwise(None)
             )
  .groupBy('n_candidates_bin').count()
  .union(
    spark.createDataFrame(
      [('0', df_records_with_zero_candidates.count())], 
      ['n_candidates_bin', 'count']
    )
  )
  .orderBy(F.col('n_candidates_bin').cast('int').asc_nulls_last())
)
display(df_candidates_per_training_record)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Records with zero candidates

# COMMAND ----------

display(df_records_with_zero_candidates)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training records with only one blocking candidate from PDS
# MAGIC
# MAGIC Ideally there shouldn't be many here, so that probabilistic linkage gets a chance to differentiate between candidates.

# COMMAND ----------

display(
  df_predictions
  .join(
    (
      df_predictions
      .groupBy('unique_reference_r')
      .count()
      .filter(F.col('count') == 1)
    ),
    'unique_reference_r',
    'left_semi'
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

df_blocking_cumulative_counts = (
  spark.createDataFrame(linker.cumulative_comparisons_from_blocking_rules_records())
  .withColumnRenamed('row_count', 'new_comparisons_count')
)

df_blocking_counts = (
  spark.createDataFrame(
    [{'rule': rule, 'comparisons_count': linker.count_num_comparisons_from_blocking_rule(rule)} for rule in blocking_rules]
  )
  .withColumn('sequence', F.monotonically_increasing_id())
  .join(df_blocking_cumulative_counts, 'rule', 'inner')
)

display(df_blocking_counts.orderBy('sequence').drop('sequence'))

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploded PDS Blocking Rule Evaluation

# COMMAND ----------

# Read and preprocess dataframes
df_pds_exploded = load_pds_full_or_exploded(exploded=True)
df_pds_exploded = process_splink_pipeline_data(df_pds_exploded)


# COMMAND ----------

# Instantiate linker
linker_pds = SparkLinker(
  df_pds_exploded,
  settings,
  database = params.DATABASE,
  break_lineage_method = 'persist',
  register_udfs_automatically = False
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluating Blocking Rules
# MAGIC
# MAGIC This evaluation stage is just to evaluate the effect on blocking comparison count for the whole of PDS, so we just need to get the counts out for each.

# COMMAND ----------

df_blocking_cumulative_counts = (
  spark.createDataFrame(linker_pds.cumulative_comparisons_from_blocking_rules_records())
  .withColumnRenamed('row_count', 'new_comparisons_count')
)

df_blocking_counts = (
  spark.createDataFrame(
    [{'rule': rule, 'comparisons_count': linker_pds.count_num_comparisons_from_blocking_rule(rule)} for rule in blocking_rules]
  )
  .withColumn('sequence', F.monotonically_increasing_id())
  .join(df_blocking_cumulative_counts, 'rule', 'inner')
)

display(df_blocking_counts.orderBy('sequence').drop('sequence'))

# COMMAND ----------

