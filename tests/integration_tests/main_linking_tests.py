# Databricks notebook source
# MAGIC %md
# MAGIC # Set Up for Test 

# COMMAND ----------

# MAGIC %run ../../utils/test_utils

# COMMAND ----------

# MAGIC %run ../../utils/model_utils

# COMMAND ----------

dbutils.widgets.text("params", "")
params_serialized = dbutils.widgets.get("params")

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Integration Test for Main

# COMMAND ----------

dbutils.notebook.run('../../notebooks_linking/training_linking', 0, {"params": params_serialized})

# COMMAND ----------

dbutils.notebook.run('../../predict_linking', 0, {"params": params_serialized})

# COMMAND ----------

dbutils.notebook.run('../../notebooks_linking/mps_comparison_evaluation', 0, {"params": params_serialized})

# COMMAND ----------

dbutils.notebook.run('../../notebooks_linking/clerical_review_evaluation', 0, {"params": params_serialized})

# COMMAND ----------

dbutils.notebook.run('../../notebooks_linking/metrics_and_distributions_evaluation', 0, {"params": params_serialized})

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC DELETE FROM mps_enhancement_collab.firebreak_splink_models WHERE description='temp_splink_integration_test_model_linking';
# MAGIC DELETE FROM mps_enhancement_collab.firebreak_splink_models WHERE description='temp_splink_integration_test_model';

# COMMAND ----------

