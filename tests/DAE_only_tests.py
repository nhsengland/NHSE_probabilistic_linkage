# Databricks notebook source
import json 
from datetime import datetime

# COMMAND ----------

# MAGIC %run ./function_test_suite

# COMMAND ----------

# MAGIC %run ../utils/model_utils

# COMMAND ----------

# MAGIC %run ../utils/dataset_ingestion_utils

# COMMAND ----------

# MAGIC %md
# MAGIC # save_model

# COMMAND ----------

suite_save_model = FunctionTestSuite()

@suite_save_model.add_test
def check_it_saves_something():
  ''' Define a model and check that it can be saved with save_model
  '''
  
  test_model = {'link_type': 'link_only', 
           'unique_id_column_name': 'UNIQUE_REFERENCE', 
           'comparisons': [{
             'output_column_name': 'NHS_NO', 
             'comparison_levels': [{
               'sql_condition': '`NHS_NO_l` IS NULL OR `NHS_NO_r` IS NULL', 
               'label_for_charts': 'Null', 
               'is_null_level': True}, 
               {'sql_condition': '`NHS_NO_l` = `NHS_NO_r`', 
                'label_for_charts': 'Exact match', 
                'm_probability': 0.6, 
                'u_probability': 0.1}, 
               {'sql_condition': 'levenshtein(`NHS_NO_l`, `NHS_NO_r`) <= 2', 
                'label_for_charts': 'Levenshtein <= 2', 
                'm_probability': 0.5, 
                'u_probability': 0.2}, 
               {'sql_condition': 'ELSE', 
                'label_for_charts': 'All other comparisons', 
                'm_probability': 0.8, 
                'u_probability': 0.9}], 
             'comparison_description': 'Exact match vs. Nhs_No within levenshtein threshold 2 vs. anything else'}, 
             {
             'output_column_name': 'GIVEN_NAME', 
             'comparison_levels': [{
               'sql_condition': '`GIVEN_NAME_l` IS NULL OR `GIVEN_NAME_r` IS NULL', 
               'label_for_charts': 'Null', 'is_null_level': True}], 
              'comparison_description': 'Exact match vs. Nhs_No within levenshtein threshold 2 vs. anything else'}],
             'sql_dialect': 'spark', 
             'linker_uid': 'uzngqt88', 
             'probability_two_random_records_match': 6.260685969280594e-10}
  
  time_now = datetime.now()
  
  save_model(test_model, 'unit_test', 'mps_enhancement_collab', 'unit_test_models')
  
  df_splink_models = spark.table('mps_enhancement_collab.unit_test_models').filter(F.col('description')=='unit_test').filter(F.col('datetime')>time_now)
  assert df_splink_models.count()==1
  
  
@suite_save_model.add_test
def check_contents():
  ''' Save a model then load it. Check that it's contents match the original.
  '''
  
  test_model = {"test":1}
  
  time_now = datetime.now()
  
  save_model(test_model, 'unit_test', 'mps_enhancement_collab', 'unit_test_models')
  
  df_expected = spark.createDataFrame(
    [
      (json.dumps(test_model), 'unit_test'),
    ],
    ['json', 'description']
  )
  
  df_output = spark.table('mps_enhancement_collab.unit_test_models').filter(F.col('description')=='unit_test').filter(F.col('datetime')>time_now).select(['json','description'])
  
  assert compare_results(df_output, df_expected,['description'])
  
suite_save_model.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # get_model

# COMMAND ----------

suite_get_model = FunctionTestSuite()

@suite_get_model.add_test
def no_description():
  ''' Test that if we don't give get_model a model description, it just returns the most recent model.
  '''
  
  test_model = {"test":1}

  save_model(test_model,'unit_test', 'mps_enhancement_collab', 'unit_test_models')
  
  output = get_model('mps_enhancement_collab','unit_test_models')

  assert output == test_model
  
  
@suite_get_model.add_test
def description_given():
  ''' Test that if we give get_model a description, it returns the model matching that description.
  '''
  
  test_model = {"test_get_model":1}
  
  save_model(test_model, 'get_model_test', 'mps_enhancement_collab', 'unit_test_models')
  
  output = get_model('mps_enhancement_collab','unit_test_models','get_model_test')
  
  assert output == test_model
  
  
@suite_get_model.add_test
def description_given_not_most_recent_save():
  ''' Test that if we give get_model a description, it returns the model matching that description. Even if it wasn't the most recent model.
  '''
  
  test_model = {"test_get_model":1}
  test_model2 = {"test_get_model2":2}
  
  save_model(test_model, 'get_model_test', 'mps_enhancement_collab', 'unit_test_models')
  save_model(test_model2, 'get_model_test2', 'mps_enhancement_collab', 'unit_test_models')
  
  output = get_model('mps_enhancement_collab','unit_test_models','get_model_test')
  
  assert output == test_model
  
suite_get_model.run()