# Databricks notebook source
import os
import json

# COMMAND ----------

# MAGIC %md
# MAGIC # Unit Tests

# COMMAND ----------

# MAGIC %run ./utils/parameter_lists

# COMMAND ----------

params_serialized_linking = json.dumps({
    "integration_tests": True,
    "comps_dict": comparisons,
    "model_hash": ''
})

params_serialized_dedupe = json.dumps({
    "integration_tests": True,
    "comps_dict": comparisons,
    "model_hash": ''
})

# COMMAND ----------

dbutils.notebook.run('./tests/preprocessing_tests', 0)
dbutils.notebook.run('./tests/dataset_ingestion_tests', 0)
dbutils.notebook.run('./tests/model_tests', 0)
dbutils.notebook.run('./tests/DAE_only_tests', 0)
dbutils.notebook.run('./tests/eval_tests', 0)

# COMMAND ----------

# MAGIC %md
# MAGIC # Integration Tests

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dont forget to set integration_tests to True in the parameters file before running this!

# COMMAND ----------

dbutils.notebook.run('./tests/integration_tests/main_linking_tests', 0, {'params': params_serialized_linking})

# COMMAND ----------

dbutils.notebook.run('./tests/integration_tests/main_dedupe_tests', 0, {'params': params_serialized_dedupe})

# COMMAND ----------

