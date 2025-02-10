# Databricks notebook source
# MAGIC %md
# MAGIC # Set Up for Test 

# COMMAND ----------

# MAGIC %run ../../utils/test_utils

# COMMAND ----------

# MAGIC %run ../../utils/preprocessing_utils

# COMMAND ----------

# MAGIC %run ../../utils/model_utils

# COMMAND ----------

# MAGIC %run ../../parameters_dedupe

# COMMAND ----------

dbutils.widgets.text("params", "")
params_serialized = dbutils.widgets.get("params")

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Integration Test for Main

# COMMAND ----------

dbutils.notebook.run('../../notebooks_dedupe/training_dedupe', 0, {"params": params_serialized})

# COMMAND ----------

dbutils.notebook.run('../../predict_dedupe', 0, {"params": params_serialized})

# COMMAND ----------

dbutils.notebook.run('../../notebooks_dedupe/evaluation_dedupe', 0, {"params": params_serialized})

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC DELETE FROM mps_enhancement_collab.firebreak_splink_models WHERE description='temp_splink_integration_test_model';

# COMMAND ----------

