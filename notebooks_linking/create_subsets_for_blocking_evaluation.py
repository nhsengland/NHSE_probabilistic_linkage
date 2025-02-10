# Databricks notebook source
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %run ../parameters_linking

# COMMAND ----------

# MAGIC %run ../utils/model_utils

# COMMAND ----------

# MAGIC %run ../utils/dataset_ingestion_utils

# COMMAND ----------

# MAGIC %run ../utils/preprocessing_utils

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Ingest data

# COMMAND ----------

df_pds_exploded = load_pds_full_or_exploded(
    params.DATABASE_PDS, params.TABLE_PDS, exploded=True
)
df_pds_exploded = preprocess_all_demographics(df_pds_exploded).select(
    [
        "UNIQUE_REFERENCE",
        "LABEL",
        "NHS_NO",
        "GIVEN_NAME",
        "FAMILY_NAME",
        "DATE_OF_BIRTH",
        "POSTCODE",
        "GENDER",
    ]
)

# COMMAND ----------

# select 1000 random training records

df_training = spark.table(f"{params.DATABASE}.{params.TRAINING_TABLE}").select(
    "UNIQUE_REFERENCE",
    "GENDER",
    "GIVEN_NAME",
    "NHS_NO",
    "FAMILY_NAME",
    "DATE_OF_BIRTH",
    "POSTCODE",
    "LABEL",
)
df_training = preprocess_all_demographics(df_training)
df_training = df_training.orderBy(F.rand(1234)).limit(1000)
df_training.cache().count()

df_training.write.saveAsTable(
    f"{params.DATABASE}.{params.BR_TRAINING_TABLE}", mode="overwrite"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add derivations of the demographics to use as very loose blocking rules

# COMMAND ----------

df_pds_exploded = (
    df_pds_exploded.withColumn(
        "GIVEN_NAME_SOUNDEX", F.soundex(F.split(F.col("GIVEN_NAME"), " ")[0])
    )
    .withColumn("FAMILY_NAME_SOUNDEX", F.soundex(F.split(F.col("FAMILY_NAME"), " ")[0]))
    .withColumn("GIVEN_NAME_2_4", F.substring(F.col("GIVEN_NAME"), 2, 3))
    .withColumn("FAMILY_NAME_2_4", F.substring(F.col("FAMILY_NAME"), 2, 3))
    .withColumn("YEAR_AND_MONTH_OF_BIRTH", F.col("DATE_OF_BIRTH").substr(1, 6))
    .withColumn(
        "YEAR_AND_DAY_OF_BIRTH",
        F.concat(
            F.col("DATE_OF_BIRTH").substr(1, 4), F.col("DATE_OF_BIRTH").substr(7, 2)
        ),
    )
    .withColumn("MONTH_AND_DAY_OF_BIRTH", F.col("DATE_OF_BIRTH").substr(5, 4))
    .withColumn("INCODE", F.split(F.col("POSTCODE"), " ")[1])
    .withColumn("POSTCODE_INITIALS", F.col("POSTCODE").substr(1, 2))
)

# COMMAND ----------

df_training = (
    df_training.withColumn(
        "GIVEN_NAME_SOUNDEX", F.soundex(F.split(F.col("GIVEN_NAME"), " ")[0])
    )
    .withColumn("FAMILY_NAME_SOUNDEX", F.soundex(F.split(F.col("FAMILY_NAME"), " ")[0]))
    .withColumn("GIVEN_NAME_2_4", F.substring(F.col("GIVEN_NAME"), 2, 3))
    .withColumn("FAMILY_NAME_2_4", F.substring(F.col("FAMILY_NAME"), 2, 3))
    .withColumn("YEAR_AND_MONTH_OF_BIRTH", F.col("DATE_OF_BIRTH").substr(1, 6))
    .withColumn(
        "YEAR_AND_DAY_OF_BIRTH",
        F.concat(
            F.col("DATE_OF_BIRTH").substr(1, 4), F.col("DATE_OF_BIRTH").substr(7, 2)
        ),
    )
    .withColumn("MONTH_AND_DAY_OF_BIRTH", F.col("DATE_OF_BIRTH").substr(5, 4))
    .withColumn("INCODE", F.split(F.col("POSTCODE"), " ")[1])
    .withColumn("POSTCODE_INITIALS", F.col("POSTCODE").substr(1, 2))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join on pairs of the derived fields

# COMMAND ----------

df_matches_on_given_name_and_family_name = (
    df_pds_exploded.join(
        df_training,
        [
            (
                (df_pds_exploded.GIVEN_NAME_SOUNDEX == df_training.GIVEN_NAME_SOUNDEX)
                | (df_pds_exploded.GIVEN_NAME_2_4 == df_training.GIVEN_NAME_2_4)
            )
            & (
                (df_pds_exploded.FAMILY_NAME_SOUNDEX == df_training.FAMILY_NAME_SOUNDEX)
                | (df_pds_exploded.FAMILY_NAME_2_4 == df_training.FAMILY_NAME_2_4)
            )
        ],
        "left_semi",
    )
    .select("NHS_NO")
    .dropDuplicates()
)
df_matches_on_given_name_and_family_name.count()

# COMMAND ----------

df_matches_on_given_name_and_date_of_birth = (
    df_pds_exploded.join(
        df_training,
        [
            (
                (df_pds_exploded.GIVEN_NAME_SOUNDEX == df_training.GIVEN_NAME_SOUNDEX)
                | (df_pds_exploded.GIVEN_NAME_2_4 == df_training.GIVEN_NAME_2_4)
            )
            & (
                (
                    df_pds_exploded.YEAR_AND_MONTH_OF_BIRTH
                    == df_training.YEAR_AND_MONTH_OF_BIRTH
                )
                | (
                    df_pds_exploded.YEAR_AND_DAY_OF_BIRTH
                    == df_training.YEAR_AND_DAY_OF_BIRTH
                )
                | (
                    df_pds_exploded.MONTH_AND_DAY_OF_BIRTH
                    == df_training.MONTH_AND_DAY_OF_BIRTH
                )
            )
        ],
        "left_semi",
    )
    .select("NHS_NO")
    .dropDuplicates()
)
df_matches_on_given_name_and_date_of_birth.count()

# COMMAND ----------

df_matches_on_given_name_and_postcode = (
    df_pds_exploded.join(
        df_training,
        [
            (
                (df_pds_exploded.GIVEN_NAME_SOUNDEX == df_training.GIVEN_NAME_SOUNDEX)
                | (df_pds_exploded.GIVEN_NAME_2_4 == df_training.GIVEN_NAME_2_4)
            )
            & (
                (df_pds_exploded.INCODE == df_training.INCODE)
                | (df_pds_exploded.POSTCODE_INITIALS == df_training.POSTCODE_INITIALS)
            )
        ],
        "left_semi",
    )
    .select("NHS_NO")
    .dropDuplicates()
)
df_matches_on_given_name_and_postcode.count()

# COMMAND ----------

df_matches_on_family_name_and_date_of_birth = (
    df_pds_exploded.join(
        df_training,
        [
            (
                (df_pds_exploded.FAMILY_NAME_SOUNDEX == df_training.FAMILY_NAME_SOUNDEX)
                | (df_pds_exploded.FAMILY_NAME_2_4 == df_training.FAMILY_NAME_2_4)
            )
            & (
                (
                    df_pds_exploded.YEAR_AND_MONTH_OF_BIRTH
                    == df_training.YEAR_AND_MONTH_OF_BIRTH
                )
                | (
                    df_pds_exploded.YEAR_AND_DAY_OF_BIRTH
                    == df_training.YEAR_AND_DAY_OF_BIRTH
                )
                | (
                    df_pds_exploded.MONTH_AND_DAY_OF_BIRTH
                    == df_training.MONTH_AND_DAY_OF_BIRTH
                )
            )
        ],
        "left_semi",
    )
    .select("NHS_NO")
    .dropDuplicates()
)
df_matches_on_family_name_and_date_of_birth.count()

# COMMAND ----------

df_matches_on_family_name_and_postcode = (
    df_pds_exploded.join(
        df_training,
        [
            (
                (df_pds_exploded.FAMILY_NAME_SOUNDEX == df_training.FAMILY_NAME_SOUNDEX)
                | (df_pds_exploded.FAMILY_NAME_2_4 == df_training.FAMILY_NAME_2_4)
            )
            & (
                (df_pds_exploded.INCODE == df_training.INCODE)
                | (df_pds_exploded.POSTCODE_INITIALS == df_training.POSTCODE_INITIALS)
            )
        ],
        "left_semi",
    )
    .select("NHS_NO")
    .dropDuplicates()
)
df_matches_on_family_name_and_postcode.count()

# COMMAND ----------

df_matches_on_date_of_birth_and_postcode = (
    df_pds_exploded.join(
        df_training,
        [
            (
                (
                    df_pds_exploded.YEAR_AND_MONTH_OF_BIRTH
                    == df_training.YEAR_AND_MONTH_OF_BIRTH
                )
                | (
                    df_pds_exploded.YEAR_AND_DAY_OF_BIRTH
                    == df_training.YEAR_AND_DAY_OF_BIRTH
                )
                | (
                    df_pds_exploded.MONTH_AND_DAY_OF_BIRTH
                    == df_training.MONTH_AND_DAY_OF_BIRTH
                )
            )
            & (
                (df_pds_exploded.INCODE == df_training.INCODE)
                | (df_pds_exploded.POSTCODE_INITIALS == df_training.POSTCODE_INITIALS)
            )
        ],
        "left_semi",
    )
    .select("NHS_NO")
    .dropDuplicates()
)
df_matches_on_date_of_birth_and_postcode.count()

# COMMAND ----------

df_all_matches = (
    df_matches_on_given_name_and_family_name.union(
        df_matches_on_given_name_and_date_of_birth
    )
    .union(df_matches_on_given_name_and_postcode)
    .union(df_matches_on_family_name_and_date_of_birth)
    .union(df_matches_on_family_name_and_postcode)
    .union(df_matches_on_date_of_birth_and_postcode)
    .select("NHS_NO")
    .dropDuplicates()
)

df_all_matches.cache().count()

# COMMAND ----------

df_pds_exploded_subset = df_pds_exploded.join(
    df_all_matches, "NHS_NO", "left_semi"
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

# COMMAND ----------

df_pds_exploded_subset.write.saveAsTable(
    f"{params.DATABASE}.{params.BR_LINKING_TABLE}", mode="overwrite"
)