# Databricks notebook source
import os
import json

# COMMAND ----------

# MAGIC %run ./utils/parameter_lists

# COMMAND ----------

# MAGIC %md
# MAGIC # Settings

# COMMAND ----------

dedupe_or_link = 'link' # change to 'dedupe' if you want to run the dedupe pipeline

run_training = True
run_predict = True
run_eval = True
 
model_hash_to_use = '' # change this to the model you want to load up if you arent running training and/or predict. 

# COMMAND ----------

if dedupe_or_link == 'link':
  params_serialized = json.dumps({
      "integration_tests": False,  
      "comps_dict": comparisons,
      "model_hash": model_hash_to_use
  })

if dedupe_or_link == 'dedupe':
  params_serialized = json.dumps({
      "integration_tests": False,
      "comps_dict": comparisons,
      "model_hash": model_hash_to_use
  })

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Pipeline

# COMMAND ----------

# MAGIC %run ./utils/preprocessing_utils

# COMMAND ----------

# MAGIC %run ./utils/model_utils

# COMMAND ----------

# MAGIC %run ./parameters_dedupe

# COMMAND ----------

if link_or_dedupe == 'link':
  
  if run_training:
    dbutils.notebook.run('../../notebooks_linking/training_linking', 0, {"params": params_serialized})
    
  if run_predict:
    dbutils.notebook.run('../../predict_linking', 0, {"params": params_serialized})
    
  if run_eval:
    dbutils.notebook.run('../../notebooks_linking/mps_comparison_evaluation', 0, {"params": params_serialized})
    dbutils.notebook.run('../../notebooks_linking/clerical_review_evaluation', 0, {"params": params_serialized})
    dbutils.notebook.run('../../notebooks_linking/metrics_and_distributions_evaluation', 0, {"params": params_serialized})

# COMMAND ----------

if link_or_dedupe == 'dedupe':
  
  if run_training:
    dbutils.notebook.run('../../notebooks_dedupe/training_dedupe', 0, {"params": params_serialized})
    
  if run_predict:
    dbutils.notebook.run('../../predict_dedupe', 0, {"params": params_serialized})
  
  if run_eval:
    dbutils.notebook.run('../../notebooks_dedupe/evaluation_dedupe', 0, {"params": params_serialized})
  

# COMMAND ----------

