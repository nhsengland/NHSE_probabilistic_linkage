# Databricks notebook source
import os
import pandas as pd
import statistics

if not os.getcwd().startswith("/home/spark"):
    import sys

    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)
    parent_dir_path = os.path.dirname(current_dir_path)
    sys.path.append(parent_dir_path)

    os.environ["PYSPARK_PYTHON"] = "python"

    from pyspark.sql import SparkSession
    from function_test_suite import *
    from utils.model_utils import *
    import pandas as pd
    import statistics

    spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %run ./function_test_suite

# COMMAND ----------

# MAGIC %run ../utils/model_utils

# COMMAND ----------

# MAGIC %md
# MAGIC #get_m_and_u_probabilities

# COMMAND ----------

suite_get_m_and_u_probabilities = FunctionTestSuite()

@suite_get_m_and_u_probabilities.add_test
def all_scenarios():
  ''' Test that a full linker settings dictionary is correctly converted to a table of the m and u probabilities
  '''
  
  input = {'link_type': 'link_only', 
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

  
  df_expected = spark.createDataFrame(
    [
      ('NHS_NO','Exact match',0.6,0.1),
      ('NHS_NO','Levenshtein <= 2',0.5,0.2),
      ('NHS_NO','All other comparisons',0.8,0.9),
    ],
    ['variable', 'sql_condition', 'm_probability', 'u_probability']  
  )
  
  df_output = get_m_and_u_probabilities(input)
                                                                                                
  assert compare_results(df_output, df_expected,['variable','sql_condition'])
   
suite_get_m_and_u_probabilities.run()

# COMMAND ----------

suite_get_average_m_values_from_models = FunctionTestSuite()

@suite_get_average_m_values_from_models.add_test
def all_scenarios():
  ''' Test that we can correctly get the average m values from the m values of several trained models
  '''

  models = []
  models.append(
    {"comparisons":
      [
        {
          "output_column_name": "NAME", 
          "comparison_levels": 
          [
            {"label_for_charts": "Null",                            "is_null_level": True}, 
            {"label_for_charts": "Exact match",                     "is_null_level": False, "m_probability": 0.15}, 
            {"label_for_charts": "Exact match on given name only",  "is_null_level": False, "m_probability": 0.14}, 
            {"label_for_charts": "Exact match on family name only", "is_null_level": False, "m_probability": 0.02}, 
            {"label_for_charts": "All other comparisons",           "is_null_level": False, "m_probability": 0.49}
          ],
        },
      ]
    }
  )

  models.append(
    {"comparisons":
      [
        {
          "output_column_name": "NAME", 
          "comparison_levels": 
          [
            {"label_for_charts": "Null",                            "is_null_level": True}, 
            {"label_for_charts": "Exact match",                     "is_null_level": False, "m_probability": 0.05}, 
            {"label_for_charts": "Exact match on given name only",  "is_null_level": False, "m_probability": 0.14}, 
            {"label_for_charts": "Exact match on family name only", "is_null_level": False, "m_probability": 0.07}, 
            {"label_for_charts": "All other comparisons",           "is_null_level": False, "m_probability": 0.01}
          ],
        },
      ]
    }
  )

  models.append(
    {"comparisons":
      [
        {
          "output_column_name": "NAME", 
          "comparison_levels": 
          [
            {"label_for_charts": "Null",                            "is_null_level": True}, 
            {"label_for_charts": "Exact match",                     "is_null_level": False, "m_probability": 0.10}, 
            {"label_for_charts": "Exact match on given name only",  "is_null_level": False, "m_probability": 0.02}, 
            {"label_for_charts": "Exact match on family name only", "is_null_level": False, "m_probability": 0.06}, 
            {"label_for_charts": "All other comparisons",           "is_null_level": False, "m_probability": 0.25}
          ],
        },
      ]
    }
  )

  comparisons = [
    {
      "output_column_name": "NAME", 
      "comparison_levels": 
      [
        {"label_for_charts": "Null", "is_null_level": True}, 
        {"label_for_charts": "Exact match", "is_null_level": False, "u_probability": 0.01}, 
        {"label_for_charts": "Exact match on given name only", "is_null_level": False, "u_probability": 0}, 
        {"label_for_charts": "Exact match on family name only", "is_null_level": False, "u_probability": None}, 
        {"label_for_charts": "All other comparisons", "is_null_level": False, "u_probability": 0.97}
      ],
    },
  ] 

  df_comparisons = pd.DataFrame.from_dict(comparisons)
  dict_output = get_average_m_values_from_models('NAME', df_comparisons, models)

  dict_expected = [
    {'label_for_charts': 'Null', 'is_null_level': True},
    {'label_for_charts': 'Exact match', 'is_null_level': False, 'u_probability': 0.01, 'm_probability': 0.20},
    {'label_for_charts': 'Exact match on given name only', 'is_null_level': False, 'u_probability': 1e-09, 'm_probability': 0.20},
    {'label_for_charts': 'Exact match on family name only', 'is_null_level': False, 'u_probability': 1e-09, 'm_probability': 0.10},
    {'label_for_charts': 'All other comparisons', 'is_null_level': False, 'u_probability': 0.97, 'm_probability': 0.50}
  ]

  assert dict_output == dict_expected

suite_get_average_m_values_from_models.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # get_best_match

# COMMAND ----------

suite_get_best_match_with_close_matches = FunctionTestSuite()

@suite_get_best_match_with_close_matches.add_test
def all_scenarios():
  ''' Test that the reference with the highest match_weight is kept.
  Includes returning more than one NHS number if there are multiple close matches.
  '''  

  data = [
    ("ref1", "NHS1", 0.95, 26),
    ("ref1", "NHS2", 0.94, 20),
    ("ref1", "NHS3", 0.89, 13),
    ("ref2", "NHS4", 0.875, 14),
    ("ref2", "NHS4", 0.88, 20),
    ("ref2", "NHS5", 0.80, 13),
    ("ref3", "NHS6", 0.74, 20),
    ("ref3", "NHS7", 0.745, 18),
    ("ref3", "NHS8", 0.60, 10),
    ("ref4", "NHS9", 0.75, 24),
    ("ref4", "NHS10", 0.75, 24),
    ("ref4", "NHS11", 0.60, 13),
    ("ref5", "NHS12", 0.4, 3),
    ("ref5", "NHS13", 0.405, 4),
    ("ref5", "NHS14", 0.3, -4),
    ("ref6", "NHS15", 0.5001, 10),
    ("ref6", "NHS16", 0.4999, 9),
    ("ref7", None, None, None)
  ]

  columns = ["UNIQUE_REFERENCE_other_table", "NHS_NO_pds", 'match_probability', "match_weight"]

  df_predictions = spark.createDataFrame(data, columns)
  
  df_expected = spark.createDataFrame(
    [
      ("ref1", "NHS1", 0.95, 26, False, ['NHS1']),
      ("ref2", "NHS4", 0.88, 20, False, ['NHS4']),
      ("ref3", "NHS6", 0.74, 20, True, ['NHS7','NHS6']),
      ('ref4', 'NHS10', 0.75, 24, True, ['NHS10', 'NHS9']),
      ('ref5', 'NHS13', 0.405, 4, False, ['NHS12', 'NHS13']),
      ('ref6', 'NHS15', 0.5001, 10, True, ['NHS16', 'NHS15']),
      ("ref7", None, None, None, False, [])
    ],
    ["UNIQUE_REFERENCE_other_table", "NHS_NO_pds", 'match_probability', "match_weight", "splink_close_match", 'close_nhs_nums']
  )
  
  df_output = get_best_match(df_predictions, 5, 5)

  assert compare_results(df_output, df_expected, ['UNIQUE_REFERENCE_other_table'])
  
suite_get_best_match_with_close_matches.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # match_probabilities_output

# COMMAND ----------

suite_match_probabilities_output = FunctionTestSuite()

@suite_match_probabilities_output.add_test
def none_to_add():
  ''' Test that if all records in evaluation_df have at least one comparison predictions_df, this function simply adds the no_splink_comparisons flag
  '''
  
  predictions_df = spark.createDataFrame(
    [
      ('1',12, '08121999',1,'John','1234','Smith', 'LA21XN'),
      ('2',13, '09121999',2,'Jon','1235','Smithe', 'LA11XN'),
      ('3',14, '10121999',0,'Jonny','1236','Smithy', 'LA31XN'),
    ],
    ['UNIQUE_REFERENCE_other_table', 'match_weight', 'DATE_OF_BIRTH_other_table', 'GENDER_other_table', 'GIVEN_NAME_other_table', 'NHS_NO_other_table', 'FAMILY_NAME_other_table', 'POSTCODE_other_table']
  )
  
  evaluation_df = spark.createDataFrame(
    [
      ('1', '08121999',1,'John','1234','Smith', 'LA21XN'),
      ('2', '09121999',2,'Jon','1235','Smithe', 'LA11XN'),
      ('3', '10121999',0,'Jonny','1236','Smithy', 'LA31XN'),
    ],
    ['UNIQUE_REFERENCE', 'DATE_OF_BIRTH', 'GENDER', 'GIVEN_NAME', 'NHS_NO', 'FAMILY_NAME', 'POSTCODE']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('1',12, '08121999',1,'John','1234','Smith', 'LA21XN', False),
      ('2',13, '09121999',2,'Jon','1235','Smithe', 'LA11XN', False),
      ('3',14, '10121999',0,'Jonny','1236','Smithy', 'LA31XN', False),
    ],
    ['UNIQUE_REFERENCE_other_table', 'match_weight', 'DATE_OF_BIRTH_other_table', 'GENDER_other_table', 'GIVEN_NAME_other_table', 'NHS_NO_other_table', 'FAMILY_NAME_other_table', 'POSTCODE_other_table', 'no_splink_comparisons']
  )
  
  df_output = match_probabilities_output(predictions_df, evaluation_df)
  
  assert compare_results(df_output, df_expected, ['UNIQUE_REFERENCE_other_table'])

  
@suite_match_probabilities_output.add_test
def some_missing_records():
  ''' Test that if any records in evaulation_df have no comparisons in predictions_df, this function adds them to the output, with the match_weight of None and no_splink_comparisons True
  '''
  
  predictions_df = spark.createDataFrame(
    [
      ('1',12, '08121999',1,'John','1234','Smith', 'LA21XN'),
    ],
    ['UNIQUE_REFERENCE_other_table', 'match_weight', 'DATE_OF_BIRTH_other_table', 'GENDER_other_table', 'GIVEN_NAME_other_table', 'NHS_NO_other_table', 'FAMILY_NAME_other_table', 'POSTCODE_other_table']
  )
  
  evaluation_df = spark.createDataFrame(
    [
      ('1', '08121999',1,'John','1234','Smith', 'LA21XN'),
      ('2', '09121999',2,'Jon','1235','Smithe', 'LA11XN'),
      ('3', '10121999',0,'Jonny','1236','Smithy', 'LA31XN'),
    ],
    ['UNIQUE_REFERENCE', 'DATE_OF_BIRTH', 'GENDER', 'GIVEN_NAME', 'NHS_NO', 'FAMILY_NAME', 'POSTCODE']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('1',12, '08121999',1,'John','1234','Smith', 'LA21XN', False),
      ('2',None, '09121999',2,'Jon','1235','Smithe', 'LA11XN', True),
      ('3',None, '10121999',0,'Jonny','1236','Smithy', 'LA31XN', True),
    ],
    ['UNIQUE_REFERENCE_other_table', 'match_weight', 'DATE_OF_BIRTH_other_table', 'GENDER_other_table', 'GIVEN_NAME_other_table', 'NHS_NO_other_table', 'FAMILY_NAME_other_table', 'POSTCODE_other_table', 'no_splink_comparisons']
  )
  
  df_output = match_probabilities_output(predictions_df, evaluation_df)
  
  assert compare_results(df_output, df_expected, ['UNIQUE_REFERENCE_other_table'])

  
@suite_match_probabilities_output.add_test
def multiple_comparisons():
  ''' Test that if a record in evaluation_df has more than one comparison in predictions_df, all the comparisons are kept
  '''
  
  predictions_df = spark.createDataFrame(
    [
      ('1',12, '08121999',1,'John','1234','Smith', 'LA21XN'),
      ('1',18, '08121999',1,'John','1234','Smith', 'LA21XN'),
    ],
    ['UNIQUE_REFERENCE_other_table', 'match_weight', 'DATE_OF_BIRTH_other_table', 'GENDER_other_table', 'GIVEN_NAME_other_table', 'NHS_NO_other_table', 'FAMILY_NAME_other_table', 'POSTCODE_other_table']
  )
  
  evaluation_df = spark.createDataFrame(
    [
      ('1', '08121999',1,'John','1234','Smith', 'LA21XN'),
      ('2', '09121999',2,'Jon','1235','Smithe', 'LA11XN'),
      ('3', '10121999',0,'Jonny','1236','Smithy', 'LA31XN'),
    ],
    ['UNIQUE_REFERENCE', 'DATE_OF_BIRTH', 'GENDER', 'GIVEN_NAME', 'NHS_NO', 'FAMILY_NAME', 'POSTCODE']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('1',12, '08121999',1,'John','1234','Smith', 'LA21XN', False),
      ('1',18, '08121999',1,'John','1234','Smith', 'LA21XN', False),
      ('2',None, '09121999',2,'Jon','1235','Smithe', 'LA11XN', True),
      ('3',None, '10121999',0,'Jonny','1236','Smithy', 'LA31XN', True),
    ],
    ['UNIQUE_REFERENCE_other_table', 'match_weight', 'DATE_OF_BIRTH_other_table', 'GENDER_other_table', 'GIVEN_NAME_other_table', 'NHS_NO_other_table', 'FAMILY_NAME_other_table', 'POSTCODE_other_table', 'no_splink_comparisons']
  )
  
  df_output = match_probabilities_output(predictions_df, evaluation_df)

  assert compare_results(df_output, df_expected, ['UNIQUE_REFERENCE_other_table', 'match_weight'])

suite_match_probabilities_output.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # change_column_df_l_and_r_names

# COMMAND ----------

suite_change_column_df_l_and_r_names = FunctionTestSuite()

@suite_change_column_df_l_and_r_names.add_test
def some_l_and_some_r():
  ''' Test that column names are correctly modified
  '''
  
  df = spark.createDataFrame(
    [
      (1,'11','1',0.4,'12345','12345'),
      (2,'12','2',0.6,'12345','12345'),
      (3,'13','1',0.6,'12345','12345'),
      (4,'14','2',0.7,'12345','12345'),
      (5,'15','3',0.7,'12345','12345'),
      (6,'16','3',0.7,'12345','12345')
    ],
    ['test_id','UNIQUE_REFERENCE_l','UNIQUE_REFERENCE_r', 'match_probability','NHS_NO_r','NHS_NO_l']
  )
  
  df_expected = spark.createDataFrame(
    [
      (1,'11','1',0.4,'12345','12345'),
      (2,'12','2',0.6,'12345','12345'),
      (3,'13','1',0.6,'12345','12345'),
      (4,'14','2',0.7,'12345','12345'),
      (5,'15','3',0.7,'12345','12345'),
      (6,'16','3',0.7,'12345','12345')
    ],
    ['test_id','UNIQUE_REFERENCE_pds','UNIQUE_REFERENCE_other_table', 'match_probability','NHS_NO_other_table','NHS_NO_pds']
  )
  
  df_output = change_column_l_and_r_names(df)
  assert compare_results(df_output, df_expected, ['test_id'])

  
@suite_change_column_df_l_and_r_names.add_test
def no_l_but_yes_r():
  ''' Test that column names are correctly modified, for r table only
  '''
  
  df = spark.createDataFrame(
    [
      (1,'11','1',0.4,'12345','12345'),
      (2,'12','2',0.6,'12345','12345'),
      (3,'13','1',0.6,'12345','12345'),
      (4,'14','2',0.7,'12345','12345'),
      (5,'15','3',0.7,'12345','12345'),
      (6,'16','3',0.7,'12345','12345')
    ],
    ['test_id','UNIQUE_REFERENCE','UNIQUE_REFERENCE_r', 'match_probability','NHS_NO_r','NHS_NO']
  )
  
  df_expected = spark.createDataFrame(
    [
      (1,'11','1',0.4,'12345','12345'),
      (2,'12','2',0.6,'12345','12345'),
      (3,'13','1',0.6,'12345','12345'),
      (4,'14','2',0.7,'12345','12345'),
      (5,'15','3',0.7,'12345','12345'),
      (6,'16','3',0.7,'12345','12345')
    ],
    ['test_id','UNIQUE_REFERENCE','UNIQUE_REFERENCE_other_table', 'match_probability','NHS_NO_other_table','NHS_NO']
  )
  
  df_output = change_column_l_and_r_names(df)
  assert compare_results(df_output, df_expected, ['test_id'])
  
  
@suite_change_column_df_l_and_r_names.add_test
def no_expected_change():
  ''' Test that nothing changes if only l table is represented
  '''
  
  df = spark.createDataFrame(
    [
      (1,'11','1',0.4,'12345','12345'),
      (2,'12','2',0.6,'12345','12345'),
      (3,'13','1',0.6,'12345','12345'),
      (4,'14','2',0.7,'12345','12345'),
      (5,'15','3',0.7,'12345','12345'),
      (6,'16','3',0.7,'12345','12345')
    ],
    ['test_id','UNIQUE_REFERENCE_1','UNIQUE_REFERENCE', 'match_probability','NHS_NO_1','NHS_NO']
  )
  
  df_expected = spark.createDataFrame(
    [
      (1,'11','1',0.4,'12345','12345'),
      (2,'12','2',0.6,'12345','12345'),
      (3,'13','1',0.6,'12345','12345'),
      (4,'14','2',0.7,'12345','12345'),
      (5,'15','3',0.7,'12345','12345'),
      (6,'16','3',0.7,'12345','12345')
    ],
    ['test_id','UNIQUE_REFERENCE_1','UNIQUE_REFERENCE', 'match_probability','NHS_NO_1','NHS_NO']
  )
  
  df_output = change_column_l_and_r_names(df)
  assert compare_results(df_output, df_expected, ['test_id'])
  
suite_change_column_df_l_and_r_names.run()

# COMMAND ----------

