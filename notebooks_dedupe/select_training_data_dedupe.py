# Databricks notebook source
import pyspark.sql.functions as F
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

# MAGIC %md
# MAGIC # Training Data Selection

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract Exploded PDS and show distribution

# COMMAND ----------

# Read and preprocess dataframes
df_pds_exploded = load_pds_full_or_exploded(exploded=True)
# df_pds_exploded = preprocess_all_demographics(df_pds_exploded)


# COMMAND ----------

# Count the occurrences of NHS Numbers in the exploded dataset to get the number of exploded counts per individual. 
pds_value_counts = df_pds_exploded.groupBy('NHS_NO').count().withColumnRenamed('count', 'value_count')

# Group by the counts and get a count of those counts
pds_count_distribution = pds_value_counts.groupBy('value_count').count().withColumnRenamed('count', 'count_of_counts').orderBy('value_count')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract out a subsample to use for training

# COMMAND ----------

# Define the number of sub-samples and the test fraction
num_sub_samples = 50

# Add a random column and an assignment column to the DataFrame
df_with_random = df_pds_exploded.withColumn("random", F.rand(seed=42))
df_with_assignment = df_with_random.withColumn("assignment", (F.floor(F.col("random") * num_sub_samples)).cast("int"))

# Filter for the test set
subsample_df = df_with_assignment.filter(F.col("assignment") == 1).drop("random", "assignment")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract out subsample dataframe

# COMMAND ----------

# Count occurrences of each value in the 'Category' column
sub_pds_value_counts = subsample_df.groupBy('NHS_NO').count().withColumnRenamed('count', 'value_count')

# Group by the counts and get a count of those counts
sub_pds_count_distribution = sub_pds_value_counts.groupBy('value_count').count().withColumnRenamed('count', 'count_of_counts').orderBy('value_count')

# COMMAND ----------

  (subsample_df
   .write
   .mode('overwrite')
   .format('delta')
   .saveAsTable(f'{params.DATABASE}.{params.TRAINING_TABLE}')
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Selecting Evaluation Dataframe

# COMMAND ----------

df_pds_full = (spark.table(f'{params.DATABASE_PDS}.{params.TABLE_PDS}'))
df_pds_replaced_by = spark.table(f'pds.replaced_by')

# Create dictionary of nhs_numbers and replaced_by nhs numbers.  
ids = [val["nhs_number"] for val in df_pds_replaced_by.select("nhs_number").collect()]
ids_linked = [val["replaced_by"]for val in df_pds_replaced_by.select("replaced_by").collect()]
id_dict = dict(zip(ids, ids_linked))

# Create a dictionary of all nhs numbers in "nhs_numbers" and "replaced_by" column to every nhs number they link to.   
connected_ids = find_connected_ids(id_dict)

# Transform the dictionary into a dataframe   
connected_df = spark.createDataFrame(connected_ids, ['nhs_number', 'linked_nhs_numbers'])
# Join new dataframe on an inner join to pds full.   
pds_superseded = connected_df.join(df_pds_full, on="nhs_number", how="inner")
pds_superseded = pds_superseded.withColumn("linked_nhs_number_count", F.size(F.col("linked_nhs_numbers")))


# COMMAND ----------

pds_superseded = (
  pds_superseded
  .withColumn('UNIQUE_REFERENCE', F.monotonically_increasing_id())
  .select(
    'UNIQUE_REFERENCE',
    F.col('nhs_number').alias('LABEL'),
    F.col('nhs_number').alias('NHS_NO'),
    F.col('linked_nhs_numbers').alias('LINKED_NHS_NO'),
    F.lower(F.concat_ws(' ', F.col('preferred_name.givenNames'))).alias('GIVEN_NAME'),
    F.lower(F.col('preferred_name.familyName')).alias('FAMILY_NAME'),
    F.col('gender_code').alias('GENDER'),
    F.col('dob').alias('DATE_OF_BIRTH'),
    F.upper(F.col('home_address.postalCode')).alias('POSTCODE'),
  )
)

pds_superseded = preprocess_all_demographics(pds_superseded,
                                  preprocess_postcode_args=params.PREPROCESS_POSTCODE_ARGS,
                                  preprocess_givenname_args=params.PREPROCESS_GIVEN_NAME_ARGS,
                                  preprocess_dob_args=params.PREPROCESS_DOB_ARGS,
                                  preprocess_familyname_args=params.PREPROCESS_FAMILY_NAME_ARGS,
                                  preprocess_fullname_args = params.PREPROCESS_FULL_NAME_ARGS
                                  )

# COMMAND ----------

(pds_superseded
 .write
 .mode('overwrite')
 .format('delta')
 .saveAsTable(f'{params.DATABASE}.{params.evaluation_table}')
)