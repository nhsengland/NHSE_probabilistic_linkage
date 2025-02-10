# Databricks notebook source
import pyspark.sql.functions as F
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %run ../utils/dataset_ingestion_utils

# COMMAND ----------

# MAGIC %run ../utils/preprocessing_utils

# COMMAND ----------

# MAGIC %run ../parameters_linking

# COMMAND ----------

link_only = True

# COMMAND ----------

# Load MPS request and response table, from which we will choose some records as training and evaluation data.

df_request_response = spark.table("mps_archive.request_response").select(
    "dataset_id",
    "local_id",
    "unique_reference",
    "req_NHS_NO",
    "req_FAMILY_NAME",
    "req_GIVEN_NAME",
    "req_GENDER",
    "req_DATE_OF_BIRTH",
    "req_POSTCODE",
    "res_MATCHED_NHS_NO",
    "res_MATCHED_ALGORITHM_INDICATOR",
    "res_MATCHED_CONFIDENCE_PERCENTAGE",
)

# Remove duplicates from request_response. Where the ambiguous join has occured, just remove all records (don't keep the most recent like we did in the MPSD pipeline).
# We are going to use training records, in which there may not be any duplicates, so this is just a safety net.
id_window = Window.partitionBy(["dataset_id", "unique_reference", "local_id"]).orderBy(
    "unique_reference"
)

df_request_response = (
    df_request_response.dropDuplicates()
    .withColumn("row", F.row_number().over(id_window))
    .withColumn("max_row", F.max("row").over(id_window))
    .filter(F.col("max_row") == 1)
    .drop("row", "max_row")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Select training / evaluation data
# MAGIC For training our model we want the NHS number and the 5 demographic columns to be populated.
# MAGIC
# MAGIC Data was chosen which has a high level of completeness in these fields.

# COMMAND ----------

# Select records where NHS number and the 5 main demographics are populated.
df_splink_training_and_evaluation_data = (
    df_request_response.filter(
        (F.col("dataset_id").startswith("n"))
        #     F.col('req_NHS_NO').isNotNull() &
        & F.col("req_FAMILY_NAME").isNotNull()
        & F.col("req_GIVEN_NAME").isNotNull()
        & F.col("req_GENDER").isNotNull()
        & F.col("req_DATE_OF_BIRTH").isNotNull()
        & F.col("req_POSTCODE").isNotNull()
    )
    .withColumnRenamed("req_NHS_NO", "NHS_NO")
    .withColumnRenamed("req_FAMILY_NAME", "FAMILY_NAME")
    .withColumnRenamed("req_GIVEN_NAME", "GIVEN_NAME")
    .withColumnRenamed("req_GENDER", "GENDER")
    .withColumnRenamed("req_DATE_OF_BIRTH", "DATE_OF_BIRTH")
    .withColumnRenamed("req_POSTCODE", "POSTCODE")
)

# COMMAND ----------

# Replace superseded NHS numbers with current NHS numbers, so that there are no superseded NHS numbers in the training data.
# This applies to both the request and repsonse NHS numbers.

df_pds_replaced_by = spark.table("pds.replaced_by")
df_splink_training_and_evaluation_data = update_superseded_nhs_numbers(
    df_splink_training_and_evaluation_data, df_pds_replaced_by, "NHS_NO"
)
df_splink_training_and_evaluation_data = update_superseded_nhs_numbers(
    df_splink_training_and_evaluation_data, df_pds_replaced_by, "res_MATCHED_NHS_NO"
)

# COMMAND ----------

# Add labels.
# Records that were matched by MPS at cross check trace (on NHS number and Date Of Birth) are assigned a Gold label.
# Records that were matched by MPS at a later trace step are assigned a Silver label.
df_splink_training_and_evaluation_data = (
    df_splink_training_and_evaluation_data.withColumn(
        "LABEL_TYPE",
        F.when(
            (
                (F.col("res_MATCHED_ALGORITHM_INDICATOR") == 1)
                & (F.col("res_MATCHED_CONFIDENCE_PERCENTAGE") == 100)
            ),
            F.lit("GOLD"),
        )
        .when(
            (
                (
                    (F.col("res_MATCHED_ALGORITHM_INDICATOR") == 3)
                    & (F.col("res_MATCHED_CONFIDENCE_PERCENTAGE") == 100)
                )
                | (
                    (F.col("res_MATCHED_ALGORITHM_INDICATOR") == 4)
                    & (F.col("res_MATCHED_CONFIDENCE_PERCENTAGE") > 50)
                )
            ),
            F.lit("SILVER"),
        )
        .otherwise(None),
    )
    .withColumn(
        "LABEL",
        F.when(F.col("LABEL_TYPE").isNotNull(), F.col("res_MATCHED_NHS_NO")).otherwise(
            None
        ),
    )
    .drop(
        "res_MATCHED_NHS_NO",
        "res_MATCHED_ALGORITHM_INDICATOR",
        "res_MATCHED_CONFIDENCE_PERCENTAGE",
    )
)

# COMMAND ----------

# Split data into two cohorts, 80% for training and 20% for evaluation.

training_and_evaluation_count = 100000
training_count = round(training_and_evaluation_count * 0.8)

df_splink_training_data = df_splink_training_and_evaluation_data.orderBy(
    F.rand(seed=1234)
).limit(training_count)
df_splink_training_data.count()

# COMMAND ----------

# Evaluation is the remaining 20%, found with an anti join.

df_splink_evaluation_data = df_splink_training_and_evaluation_data.join(
    df_splink_training_data, ["local_id", "UNIQUE_REFERENCE"], "left_anti"
).limit(training_and_evaluation_count - training_count)

# COMMAND ----------

# Save to tables.

df_splink_training_data.write.saveAsTable(f"{params.DATABASE}.{params.TABLE_NAME}")
df_splink_evaluation_data.write.saveAsTable(f"{params.DATABASE}.{params.LINKING_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Processing Training and Linking data for Training

# COMMAND ----------

df_pds_full = spark.table(f"{params.DATABASE_PDS}.{params.TABLE_PDS}")

df_pds_replaced_by = spark.table("pds.replaced_by")

df_pds_full = preprocess_full_pds(df_pds_full, df_pds_replaced_by).select(
    "UNIQUE_REFERENCE", "GENDER", "GIVEN_NAME", "NHS_NO", "FAMILY_NAME", "DATE_OF_BIRTH", "POSTCODE", "LABEL",
)

df_pds_full = preprocess_all_demographics(df_pds_full)

df_training = load_data_to_link(
                                params.DATABASE, params.TRAINING_TABLE, params.TRAINING_GIVEN_NAME_COLUMN, params.TRAINING_FAMILY_NAME_COLUMN, 
                                params.TRAINING_GENDER_COLUMN, params.TRAINING_POSTCODE_COLUMN, params.TRAINING_DOB_COLUMN, params.TRAINING_UNIQUE_REFERENCE
                                
                               ).select("UNIQUE_REFERENCE", "GENDER", "GIVEN_NAME", "NHS_NO", "FAMILY_NAME","DATE_OF_BIRTH", "POSTCODE","LABEL"
                                       )

df_training = preprocess_all_demographics(df_training)

# COMMAND ----------

df_pds_full.write.saveAsTable(
    f"{params.DATABASE}.temp_pds_full_preprocessed", mode="overwrite"
)

df_training.write.saveAsTable(
    f"{params.DATABASE}.temp_training_preprocessed", mode="overwrite"
)