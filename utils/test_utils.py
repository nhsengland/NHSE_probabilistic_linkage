# Databricks notebook source
# MAGIC %run ../parameters_linking

# COMMAND ----------

# MAGIC %run ../parameters_dedupe

# COMMAND ----------

# MAGIC %run ../utils/dataset_ingestion_utils

# COMMAND ----------

# MAGIC %run ../utils/preprocessing_utils

# COMMAND ----------

def add_random_data_to_test_dataset():
  ''' Create sample data for the linking integration test
  '''

    integration_sample = spark.table(
        "mps_enhancement_collab.splink_integration_test_data"
    )
    integration_sample_pds = spark.table(
        "mps_enhancement_collab.pds_subsample_integration_test"
    ).sample(False, 0.5)

    df_npex = spark.table("mps_enhancement_collab.splink_training_data_20240228")
    df_npex = df_npex.orderBy(F.rand()).limit(500)

    integration_sample = integration_sample.union(df_npex).orderBy(F.rand()).limit(10)

    integration_sample.write.mode("overwrite").saveAsTable(
        f"mps_enhancement_collab.{params.TRAINING_TABLE}"
    )
    integration_sample_pds.write.mode("overwrite").saveAsTable(
        f"mps_enhancement_collab.{params.TABLE_PDS}"
    )


# COMMAND ----------

def create_test_data_dedupe(test_dataset_pds_fraction):
  ''' Create sample test data for the dedupe integration test
  '''
  
    df_pds_exploded = load_pds_full_or_exploded("pds", "full", exploded=True)

    df_with_random = df_pds_exploded.withColumn("random", F.rand(seed=42))
    df_with_assignment = df_with_random.withColumn("assignment", (F.floor(F.col("random") * test_dataset_pds_fraction)).cast("int"))

    subsample_df = df_with_assignment.filter(F.col("assignment") == 1).drop("random", "assignment")

    subsample_df.write.mode("overwrite").saveAsTable(f"mps_enhancement_collab.{params.TRAINING_PDS_TRAINING_TABLE}")

    print("Table 1 has been saved")

    df_pds = spark.table("pds.full").sample(False, 1 / test_dataset_pds_fraction)
    df_pds.write.mode("overwrite").saveAsTable(f"mps_enhancement_collab.{params.TABLE_PDS}")
    
    print("Table 2 has been saved")

    #   selecting eval data
    df_pds_replaced_by = spark.table(f"pds.replaced_by")

    # Create dictionary of nhs_numbers and replaced_by nhs numbers.
    ids = [val["nhs_number"] for val in df_pds_replaced_by.select("nhs_number").collect()]
    ids_linked = [val["replaced_by"] for val in df_pds_replaced_by.select("replaced_by").collect()]
    id_dict = dict(zip(ids, ids_linked))

    # Create a dictionary of all nhs numbers in "nhs_numbers" and "replaced_by" column to every nhs number they link to.
    connected_ids = find_connected_ids(id_dict)

    # Transform the dictionary into a dataframe
    connected_df = spark.createDataFrame(connected_ids, ["nhs_number", "linked_nhs_numbers"])
    # Join new dataframe on an inner join to pds full.
    pds_superseded = connected_df.join(df_pds, on="nhs_number", how="inner")
    pds_superseded = pds_superseded.withColumn("linked_nhs_number_count", F.size(F.col("linked_nhs_numbers")))

    pds_superseded = pds_superseded.withColumn("UNIQUE_REFERENCE", F.monotonically_increasing_id()).select(
        "UNIQUE_REFERENCE", F.col("nhs_number").alias("LABEL"), F.col("nhs_number").alias("NHS_NO"), F.col("linked_nhs_numbers").alias("LINKED_NHS_NO"), 
        F.lower(F.concat_ws(" ", F.col("preferred_name.givenNames"))).alias("GIVEN_NAME"),F.lower(F.col("preferred_name.familyName")).alias("FAMILY_NAME"),
        F.col("gender_code").alias("GENDER"), F.col("dob").alias("DATE_OF_BIRTH"), F.upper(F.col("home_address.postalCode")).alias("POSTCODE")
    )
    
    pds_superseded = preprocess_all_demographics(pds_superseded)
    pds_superseded.write.mode("overwrite").saveAsTable(f"mps_enhancement_collab.{evaluation_table}")
    print("Table 3 has been saved")


# COMMAND ----------


def delete_test_datasets():
    spark.sql(f"DROP TABLE IF EXISTS mps_enhancement_collab.{params.TABLE_PDS}")
    spark.sql(f"DROP TABLE IF EXISTS mps_enhancement_collab.{params.TRAINING_TABLE}")
    spark.sql(f"DROP TABLE IF EXISTS mps_enhancement_collab.{params.MATCH_PROBABILITIES_TABLE}")
    spark.sql(f"DROP TABLE IF EXISTS mps_enhancement_collab.{PREDICTIONS_TABLE}")
    spark.sql(f"DROP TABLE IF EXISTS mps_enhancement_collab.{EVALUATION_TABLE}")