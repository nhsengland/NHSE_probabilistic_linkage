# Databricks notebook source
import os
from pyspark.sql import types as T

if not os.getcwd().startswith("/home/spark"):
    import sys

    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)
    parent_dir_path = os.path.dirname(current_dir_path)
    sys.path.append(parent_dir_path)

    os.environ["PYSPARK_PYTHON"] = "python"

    from pyspark.sql import SparkSession
    from function_test_suite import *
    from utils.dataset_ingestion_utils import *
    from utils.eval_utils import *

    spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %run ./function_test_suite

# COMMAND ----------

# MAGIC %run ../utils/dataset_ingestion_utils

# COMMAND ----------

# MAGIC %run ../utils/eval_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## add_confusion_matrix

# COMMAND ----------

suite_add_confusion_matrix = FunctionTestSuite()

@suite_add_confusion_matrix.add_test
def default_match_weight_threshold():
  ''' Test that confusion matrix columns are added correctly when the match weight threshold is not given (so default)
  '''
  
  df_input = spark.createDataFrame(
    [
      ('1', '', '', '', 0, False, False),
      ('2', 'GOLD', '111111111', '111111111', 6, False, False),
      ('3', 'SILVER', '111111111', '111111111', 6, False, False),
      ('4', 'silver', '111111111', '111111111', 6, False, False),
      ('5', 'GOLD', '111111111', '111111112', 6, False, False),
      ('6', 'SILVER', '111111111', '111111112', 6, False, False),
      ('7', 'no label', '111111111', '111111112', 6, False, False),
      ('8', 'SILVER', None, '111111112', 6, False, False),
      ('9', 'no label', None, '111111112', 6, False, False),
      ('10', 'GOLD', '111111112', '111111112', 1, False, False),
      ('11', 'SILVER', '111111112', '111111112', 1, False, False),
      ('12', 'no label', '111111112', '111111112', 1, False, False),
    ],
    ['test_id', 'LABEL_TYPE', 'LABEL', 'NHS_NO_pds', 'match_weight', 'splink_close_match', 'mps_multiple_pds_matches']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('1', '', '', '', 0, False, False, False, False, False, False, False, False),
      ('2', 'GOLD', '111111111', '111111111', 6, False, False, True, False, False, False, False, False),
      ('3', 'SILVER', '111111111', '111111111', 6, False, False, True, False, False, False, False, False),
      ('4', 'silver', '111111111', '111111111', 6, False, False, False, False, False, False, False, False),
      ('5', 'GOLD', '111111111', '111111112', 6, False, False, False, True, False, False, False, False),
      ('6', 'SILVER', '111111111', '111111112', 6, False, False, False, False, True, False, False, False),
      ('7', 'no label', '111111111', '111111112', 6, False, False, False, False, True, False, False, False),
      ('8', 'SILVER', None, '111111112', 6, False, False, False, False, True, False, False, False),
      ('9', 'no label', None, '111111112', 6, False, False, False, False, True, False, False, False),
      ('10', 'GOLD', '111111112', '111111112', 1, False, False, False, False, False, True, False, False),
      ('11', 'SILVER', '111111112', '111111112', 1, False, False, False, False, False, False, True, False),
      ('12', 'no label', '111111112', '111111112', 1, False, False, False, False, False, False, False, True),
    ],
    ['test_id', 'LABEL_TYPE', 'LABEL', 'NHS_NO_pds', 'match_weight', 'splink_close_match', 'mps_multiple_pds_matches', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
    
  df_output = add_confusion_matrix_to_df(df_input)
  
  assert compare_results(df_output, df_expected, ['test_id'])

# COMMAND ----------

@suite_add_confusion_matrix.add_test
def manual_match_weight():
  ''' Test that the confusion matrix columns are added correctly with a submitted match weight threshold
  '''
  
  df_input = spark.createDataFrame(
    [
      ('1', '', '', '', 11, False, False),
      ('2', 'GOLD', '111111111', '111111111', 11, False, False),
      ('3', 'SILVER', '111111111', '111111111', 11, False, False),
      ('4', 'SILVER', '111111111', '111111111', 11, False, False),
      ('5', 'GOLD', '111111111', '111111112', 11, False, False),
      ('6', 'SILVER', '111111111', '111111112', 11, False, False),
      ('7', 'no label', '111111111', '111111112', 11, False, False),
      ('8', 'SILVER', None, '111111112', 11, False, False),
      ('9', 'no label', None, '111111112', 11, False, False),
      ('10', 'GOLD', '111111112', '111111112', 7, False, False),
      ('11', 'SILVER', '111111112', '111111112', 7, False, False),
      ('12', 'no label', '111111112', '111111112', 7, False, False),
    ],
    ['test_id', 'LABEL_TYPE', 'LABEL', 'NHS_NO_pds', 'match_weight', 'splink_close_match', 'mps_multiple_pds_matches']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('1', '', '', '', 11, False, False, False, False, False, False, False, False),
      ('2', 'GOLD', '111111111', '111111111', 11, False, False, True, False, False, False, False, False),
      ('3', 'SILVER', '111111111', '111111111', 11, False, False, True, False, False, False, False, False),
      ('4', 'SILVER', '111111111', '111111111', 11, False, False, True, False, False, False, False, False),
      ('5', 'GOLD', '111111111', '111111112', 11, False, False, False, True, False, False, False, False),
      ('6', 'SILVER', '111111111', '111111112', 11, False, False, False, False, True, False, False, False),
      ('7', 'no label', '111111111', '111111112', 11, False, False, False, False, True, False, False, False),
      ('8', 'SILVER', None, '111111112', 11, False, False, False, False, True, False, False, False),
      ('9', 'no label', None, '111111112', 11, False, False, False, False, True, False, False, False),
      ('10', 'GOLD', '111111112', '111111112', 7, False, False, False, False, False, True, False, False),
      ('11', 'SILVER', '111111112', '111111112', 7, False, False, False, False, False, False, True, False),
      ('12', 'no label', '111111112', '111111112', 7, False, False, False, False, False, False, False, True),
    ],
    ['test_id', 'LABEL_TYPE', 'LABEL', 'NHS_NO_pds', 'match_weight',  'splink_close_match', 'mps_multiple_pds_matches', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
    
  df_output = add_confusion_matrix_to_df(df_input, 10)
  
  assert compare_results(df_output, df_expected, ['test_id'])

# COMMAND ----------

@suite_add_confusion_matrix.add_test
def close_match():
  ''' Test that confusion matrix columns are derived correctly when some rows are flagged as having close matches.
  '''
  
  df_input = spark.createDataFrame(
    [
      ('1', '', '', '', 0, False, True),
      ('2', 'GOLD', '111111111', '111111111', 20, True, False),
      ('3', 'SILVER', '111111111', '111111111', 20, True, False),
      ('4', 'GOLD', '111111111', '111111112', 20, True, False),
      ('5', 'SILVER', '111111111', '111111112', 20, True, False),
      ('6', 'no label', '111111111', '111111112', 20, True, False),
      ('7', 'no label', '111111111', '111111112', 20, True, True),
      ('8', 'no label', '111111111', '111111111', 20, False, True),
    ],
    ['test_id', 'LABEL_TYPE', 'LABEL', 'NHS_NO_pds', 'match_weight', 'splink_close_match', 'mps_multiple_pds_matches']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('1', '', '', '', 0, False, True, False, False, False, False, False, False),
      ('2', 'GOLD', '111111111', '111111111', 20, True, False, False, False, False, True, False, False),
      ('3', 'SILVER', '111111111', '111111111', 20, True, False, False, False, False, False, True, False),
      ('4', 'GOLD', '111111111', '111111112', 20, True, False, False, False, False, True, False, False),
      ('5', 'SILVER', '111111111', '111111112', 20, True, False, False, False, False, False, True, False),
      ('6', 'no label', '111111111', '111111112', 20, True, False, False, False, False, False, False, True),
      ('7', 'no label', '111111111', '111111112', 20, True, True, False, False, False, False, False, True),
      ('8', 'no label', '111111111', '111111111', 20, False, True, False, False, True, False, False, False),

    ],
    ['test_id', 'LABEL_TYPE', 'LABEL', 'NHS_NO_pds', 'match_weight',  'splink_close_match', 'mps_multiple_pds_matches', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
    
  df_output = add_confusion_matrix_to_df(df_input, 5)
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
suite_add_confusion_matrix.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # filter_confusion_matrix

# COMMAND ----------

suite_filter_confusion_matrix = FunctionTestSuite()

@suite_filter_confusion_matrix.add_test
def true_positives():
  ''' Test filtering the confusion matrix for true positives 
  '''
  
  df_input = spark.createDataFrame(
    [
      ('1', False, False, False, False, False, False),
      ('2', True, False, False, False, False, False),
      ('3', True, False, False, False, False, False),
      ('4', False, False, False, False, False, False),
      ('5', False, True, False, False, False, False),
      ('6', False, False, True, False, False, False),
      ('7', False, False, True, False, False, False),
      ('8', False, False, True, False, False, False),
      ('9', False, False, True, False, False, False),
      ('10', False, False, False, True, False, False),
      ('11', False, False, False, False, True, False),
      ('12', False, False, False, False, False, True),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
  df_expected = spark.createDataFrame(
    [
      ('2', True, False, False, False, False, False),
      ('3', True, False, False, False, False, False),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
    
  df_output = filter_confusion_matrix(df_input,'true_positives')
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
  
@suite_filter_confusion_matrix.add_test
def false_positives():
  ''' Test filtering the confusion matrix for false positives
  '''
  
  df_input = spark.createDataFrame(
    [
      ('1', False, False, False, False, False, False),
      ('2', True, False, False, False, False, False),
      ('3', True, False, False, False, False, False),
      ('4', False, False, False, False, False, False),
      ('5', False, True, False, False, False, False),
      ('6', False, False, True, False, False, False),
      ('7', False, False, True, False, False, False),
      ('8', False, False, True, False, False, False),
      ('9', False, False, True, False, False, False),
      ('10', False, False, False, True, False, False),
      ('11', False, False, False, False, True, False),
      ('12', False, False, False, False, False, True),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
  df_expected = spark.createDataFrame(
    [
      ('5', False, True, False, False, False, False),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
    
  df_output = filter_confusion_matrix(df_input,'false_positives')
  
  assert compare_results(df_output, df_expected, ['test_id'])

# COMMAND ----------

@suite_filter_confusion_matrix.add_test
def possible_false_positives():
  ''' Test filtering the confusion matrix for possible false positives
  '''

  df_input = spark.createDataFrame(
    [
      ('1', False, False, False, False, False, False),
      ('2', True, False, False, False, False, False),
      ('3', True, False, False, False, False, False),
      ('4', False, False, False, False, False, False),
      ('5', False, True, False, False, False, False),
      ('6', False, False, True, False, False, False),
      ('7', False, False, True, False, False, False),
      ('8', False, False, True, False, False, False),
      ('9', False, False, True, False, False, False),
      ('10', False, False, False, True, False, False),
      ('11', False, False, False, False, True, False),
      ('12', False, False, False, False, False, True),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
  df_expected = spark.createDataFrame(
    [
      ('6', False, False, True, False, False, False),
      ('7', False, False, True, False, False, False),
      ('8', False, False, True, False, False, False),
      ('9', False, False, True, False, False, False),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
    
  df_output = filter_confusion_matrix(df_input,'possible_false_positives')
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
  
@suite_filter_confusion_matrix.add_test
def false_negatives():
  ''' Test filtering the confusion matrix for false negatives
  '''

  df_input = spark.createDataFrame(
    [
      ('1', False, False, False, False, False, False),
      ('2', True, False, False, False, False, False),
      ('3', True, False, False, False, False, False),
      ('4', False, False, False, False, False, False),
      ('5', False, True, False, False, False, False),
      ('6', False, False, True, False, False, False),
      ('7', False, False, True, False, False, False),
      ('8', False, False, True, False, False, False),
      ('9', False, False, True, False, False, False),
      ('10', False, False, False, True, False, False),
      ('11', False, False, False, False, True, False),
      ('12', False, False, False, False, False, True),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
  df_expected = spark.createDataFrame(
    [
      ('10', False, False, False, True, False, False),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
    
  df_output = filter_confusion_matrix(df_input,'false_negatives')
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
  
@suite_filter_confusion_matrix.add_test
def possible_false_negatives():
  ''' Test filtering the confusion matrix for possible false negatives
  '''

  df_input = spark.createDataFrame(
    [
      ('1', False, False, False, False, False, False),
      ('2', True, False, False, False, False, False),
      ('3', True, False, False, False, False, False),
      ('4', False, False, False, False, False, False),
      ('5', False, True, False, False, False, False),
      ('6', False, False, True, False, False, False),
      ('7', False, False, True, False, False, False),
      ('8', False, False, True, False, False, False),
      ('9', False, False, True, False, False, False),
      ('10', False, False, False, True, False, False),
      ('11', False, False, False, False, True, False),
      ('12', False, False, False, False, False, True),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
  df_expected = spark.createDataFrame(
    [
      ('11', False, False, False, False, True, False),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
    
  df_output = filter_confusion_matrix(df_input,'possible_false_negatives')
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
  
@suite_filter_confusion_matrix.add_test
def false_negatives():
  ''' Test filtering the confusion matrix for false negatives
  '''

  df_input = spark.createDataFrame(
    [
      ('1', False, False, False, False, False, False),
      ('2', True, False, False, False, False, False),
      ('3', True, False, False, False, False, False),
      ('4', False, False, False, False, False, False),
      ('5', False, True, False, False, False, False),
      ('6', False, False, True, False, False, False),
      ('7', False, False, True, False, False, False),
      ('8', False, False, True, False, False, False),
      ('9', False, False, True, False, False, False),
      ('10', False, False, False, True, False, False),
      ('11', False, False, False, False, True, False),
      ('12', False, False, False, False, False, True),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
  df_expected = spark.createDataFrame(
    [
      ('10', False, False, False, True, False, False),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
    
  df_output = filter_confusion_matrix(df_input,'false_negatives')
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
  
@suite_filter_confusion_matrix.add_test
def true_negatives():
  ''' Test filtering the confusion matrix for true negatives
  '''

  df_input = spark.createDataFrame(
    [
      ('1', False, False, False, False, False, False),
      ('2', True, False, False, False, False, False),
      ('3', True, False, False, False, False, False),
      ('4', False, False, False, False, False, False),
      ('5', False, True, False, False, False, False),
      ('6', False, False, True, False, False, False),
      ('7', False, False, True, False, False, False),
      ('8', False, False, True, False, False, False),
      ('9', False, False, True, False, False, False),
      ('10', False, False, False, True, False, False),
      ('11', False, False, False, False, True, False),
      ('12', False, False, False, False, False, True),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
  df_expected = spark.createDataFrame(
    [
      ('12', False, False, False, False, False, True),
    ],
    ['test_id', 'true_positives', 'false_positives', 'possible_false_positives', 'false_negatives', 'possible_false_negatives', 'true_negatives']
  )
    
  df_output = filter_confusion_matrix(df_input,'true_negatives')
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
suite_filter_confusion_matrix.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # Load MPS responses

# COMMAND ----------

suite_load_mps_responses = FunctionTestSuite()

@suite_load_mps_responses.add_test
def mps_successful_step_tests():
  ''' Test adding the MPS successful step column, based on the values of MATCHED_ALGORITHM_INDICATOR, MATCHED_CONFIDENCE_PERCENTAGE, and ALGORITHMIC_TRACE_DOB_SCORE_PERC
  '''
  
  df_input = spark.createDataFrame(
    [
      (1, 0, 0, None),
      (2, 0, 0, 0),
      (3, 3, 0, 0),
      (4, 4, 0, 0),
      (5, 1, 100, 0),
      (6, 1, 100, None),
      (7, 3, 100, 0),
      (8, 3, 100, None),
      (9, 4, 50, 100),
      (10, 4, 49, 33),
      (11, 5, 100, 100),
      (12, 1000, 1, 100),
      (13, 2, 50, 0) 
    ],
    ['UNIQUE_REFERENCE', 'MATCHED_ALGORITHM_INDICATOR', 'MATCHED_CONFIDENCE_PERCENTAGE', 'ALGORITHMIC_TRACE_DOB_SCORE_PERC']
  )
  
  df_pds_replaced_by = spark.createDataFrame(
    [('123', '456')], 
    ['nhs_number', 'replaced_by']
  )

  df_expected = spark.createDataFrame(
    [
      (1, None, 'No_PDS_tracing_run'),
      (2, 0, 'No_PDS_tracing_run'),
      (3, 0, 'No_PDS_match_found'),
      (4, 0, 'No_PDS_match_found'),
      (5, 0, 'CCT_live'),
      (6, None, 'CCT_cached'),
      (7, 0, 'alphanumeric_trace_live'),
      (8, None, 'alphanumeric_trace_live'),
      (9, 100, 'algorithmic_trace_live'),
      (10, 33, None),
      (11, 100, None),
      (12, 100, None),
      (13, 0, None)      
    ],
    ['UNIQUE_REFERENCE', 'ALGORITHMIC_TRACE_DOB_SCORE_PERC', 'mps_successful_step']
  )
  
  # several columns must be present but their values are irrelevant to the tests
  # they are added here with blank strings so that the schema match test will pass
  df_input = (
    add_untested_string_cols(
      df_input, 
      ['MATCHED_NHS_NO', 'GIVEN_NAME', 'FAMILY_NAME', 'DATE_OF_BIRTH', 'POSTCODE', 'GENDER', 'created', 'ERROR_SUCCESS_CODE',
       'ALGORITHMIC_TRACE_FAMILY_NAME_SCORE_PERC', 'ALGORITHMIC_TRACE_GIVEN_NAME_SCORE_PERC', 'ALGORITHMIC_TRACE_GENDER_SCORE_PERC', 'ALGORITHMIC_TRACE_POSTCODE_SCORE_PERC']    
    )
  )
  
  df_expected = (
    add_untested_string_cols(
      df_expected, 
      ['NHS_NO_mps', 'GIVEN_NAME_mps', 'FAMILY_NAME_mps', 'DATE_OF_BIRTH_mps', 'POSTCODE_mps', 'GENDER_mps',
       'ALGORITHMIC_TRACE_FAMILY_NAME_SCORE_PERC', 'ALGORITHMIC_TRACE_GIVEN_NAME_SCORE_PERC', 'ALGORITHMIC_TRACE_GENDER_SCORE_PERC', 'ALGORITHMIC_TRACE_POSTCODE_SCORE_PERC']
    )
    # these two cols also need to be added but are not strings
    .withColumn('mps_run_date', F.lit('').cast('Date'))
    .withColumn('mps_multiple_pds_matches', F.lit(False))
    # the columns must be in this order
    .select('UNIQUE_REFERENCE', 'NHS_NO_mps', 'GIVEN_NAME_mps', 'FAMILY_NAME_mps', 'DATE_OF_BIRTH_mps', 'POSTCODE_mps', 'GENDER_mps', 'mps_run_date', 
      'mps_successful_step', 'mps_multiple_pds_matches', 'ALGORITHMIC_TRACE_FAMILY_NAME_SCORE_PERC', 'ALGORITHMIC_TRACE_GIVEN_NAME_SCORE_PERC', 
      'ALGORITHMIC_TRACE_DOB_SCORE_PERC', 'ALGORITHMIC_TRACE_GENDER_SCORE_PERC', 'ALGORITHMIC_TRACE_POSTCODE_SCORE_PERC')
  )
  
  df_output = load_mps_responses(df_input, df_pds_replaced_by)
  
  assert compare_results(df_output, df_expected, join_columns = ['UNIQUE_REFERENCE'])
  

@suite_load_mps_responses.add_test
def mps_multiple_pds_matches_tests():
  ''' Test that the mps_multiple_pds_matches flag is populated correctly, based on the ERROR_SUCCESS_CODE
  '''
  
  df_input = spark.createDataFrame(
    [
      (1, 1, 100, 0, 96),
      (2, 1, 100, 0, 97),
      (3, 1, 100, 0, None)
    ],
    ['UNIQUE_REFERENCE', 'MATCHED_ALGORITHM_INDICATOR', 'MATCHED_CONFIDENCE_PERCENTAGE', 'ALGORITHMIC_TRACE_DOB_SCORE_PERC', 'ERROR_SUCCESS_CODE']
  )
  
  df_pds_replaced_by = spark.createDataFrame(
    [('123', '456')], 
    ['nhs_number', 'replaced_by']
  )

  df_expected = spark.createDataFrame(
    [
      (1, 0, 'CCT_live', False),
      (2, 0, 'CCT_live', True),
      (3, 0, 'CCT_live', False)
    ],
    ['UNIQUE_REFERENCE', 'ALGORITHMIC_TRACE_DOB_SCORE_PERC', 'mps_successful_step', 'mps_multiple_pds_matches']
  )
  
  # several columns must be present but their values are irrelevant to the tests
  # they are added here with blank strings so that the schema match test will pass
  df_input = (
    add_untested_string_cols(
      df_input, 
      ['MATCHED_NHS_NO', 'GIVEN_NAME', 'FAMILY_NAME', 'DATE_OF_BIRTH', 'POSTCODE', 'GENDER', 'created',
       'ALGORITHMIC_TRACE_FAMILY_NAME_SCORE_PERC', 'ALGORITHMIC_TRACE_GIVEN_NAME_SCORE_PERC', 'ALGORITHMIC_TRACE_GENDER_SCORE_PERC', 'ALGORITHMIC_TRACE_POSTCODE_SCORE_PERC']    
    )
  )
  
  df_expected = (
    add_untested_string_cols(
      df_expected, 
      ['NHS_NO_mps', 'GIVEN_NAME_mps', 'FAMILY_NAME_mps', 'DATE_OF_BIRTH_mps', 'POSTCODE_mps', 'GENDER_mps',
       'ALGORITHMIC_TRACE_FAMILY_NAME_SCORE_PERC', 'ALGORITHMIC_TRACE_GIVEN_NAME_SCORE_PERC', 'ALGORITHMIC_TRACE_GENDER_SCORE_PERC', 'ALGORITHMIC_TRACE_POSTCODE_SCORE_PERC']
    )
    # this col also needs to be added but is not a string
    .withColumn('mps_run_date', F.lit('').cast('Date'))
    # the columns must be in this order
    .select('UNIQUE_REFERENCE', 'NHS_NO_mps', 'GIVEN_NAME_mps', 'FAMILY_NAME_mps', 'DATE_OF_BIRTH_mps', 'POSTCODE_mps', 'GENDER_mps', 'mps_run_date', 
      'mps_successful_step', 'mps_multiple_pds_matches', 'ALGORITHMIC_TRACE_FAMILY_NAME_SCORE_PERC', 'ALGORITHMIC_TRACE_GIVEN_NAME_SCORE_PERC', 
      'ALGORITHMIC_TRACE_DOB_SCORE_PERC', 'ALGORITHMIC_TRACE_GENDER_SCORE_PERC', 'ALGORITHMIC_TRACE_POSTCODE_SCORE_PERC')
  )
  
  df_output = load_mps_responses(df_input, df_pds_replaced_by)
  
  assert compare_results(df_output, df_expected, join_columns = ['UNIQUE_REFERENCE'])
   
suite_load_mps_responses.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # add_agreement_pattern

# COMMAND ----------

suite_add_agreement_pattern = FunctionTestSuite()

@suite_add_agreement_pattern.add_test
def no_gammas():
  ''' Test that a blank agreement pattern is returned if there are no gammas
  '''
  df_input = spark.createDataFrame(
    [
      ('1',)
    ],
    ['test_id']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('1','')
    ],
    ['test_id','agreement_patterns']
  )
  
  df_output = add_agreement_pattern(df_input)
  
  assert compare_results(df_output, df_expected, ['test_id'])

  
@suite_add_agreement_pattern.add_test
def gamma_columns():
  ''' Test the agreement pattern if there are gammas
  '''
  df_input = spark.createDataFrame(
    [
      ('1',1,0,2,1),
      ('2',1,1,2,1)
    ],
    ['test_id','gamma_dob','gamma_given_name','gamma_family_name','gamma_postcode']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('1',1,0,2,1,'1021'),
      ('2',1,1,2,1,'1121')
    ],
    ['test_id','gamma_dob','gamma_given_name','gamma_family_name','gamma_postcode','agreement_patterns']
  )
  
  df_output = add_agreement_pattern(df_input)
  
  assert compare_results(df_output, df_expected, ['test_id'])  
  
  
@suite_add_agreement_pattern.add_test
def extra_columns():
  ''' Test that only columns named gamma_x are picked up in the agreement pattern
  '''
  
  df_input = spark.createDataFrame(
    [
      ('1',1,0,2,1),
      ('2',1,1,2,1)
    ],
    ['test_id','gamma_dob','gammi_given_name','gamma_family_name','gamma_postcode']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('1',1,0,2,1,'121'),
      ('2',1,1,2,1,'121')
    ],
    ['test_id','gamma_dob','gammi_given_name','gamma_family_name','gamma_postcode','agreement_patterns']
  )
  
  df_output = add_agreement_pattern(df_input)
  
  assert compare_results(df_output, df_expected, ['test_id'])  

  
@suite_add_agreement_pattern.add_test
def gamma_of_minus_one():
  ''' Test that gammas of -1 are included in agreement patterns
  '''
  
  df_input = spark.createDataFrame(
    [
      ('1',-1,0,2,1),
      ('2',1,1,2,-1)
    ],
    ['test_id','gamma_dob','gamma_given_name','gamma_family_name','gamma_postcode']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('1',-1,0,2,1,'-1021'),
      ('2',1,1,2,-1,'112-1')
    ],
    ['test_id','gamma_dob','gamma_given_name','gamma_family_name','gamma_postcode','agreement_patterns']
  )
  
  df_output = add_agreement_pattern(df_input)
   
  assert compare_results(df_output, df_expected, ['test_id'])

suite_add_agreement_pattern.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # Clerical Review Confusion Matrix

# COMMAND ----------

suite_clerical_review_cm = FunctionTestSuite()

@suite_clerical_review_cm.add_test
def clerical_review_cm():
  ''' Test that the correct confusion matrix columns are created when a results dataframe is compared to a ground truth dataframe
  '''
  
  df_input_match_probabilities = spark.createDataFrame(
    [
      ('1', '1'),
      ('2', '2'),
      ('3', None),
      ('4', None),
      ('5', '5'),
      ('6', None),
      ('7', '2')
    ],
    ['UNIQUE_REFERENCE_other_table', 'NHS_no_pds']
  )
  
  df_input_eval = spark.createDataFrame(
    [
      ('1','1', True),
      ('2', '3', True),
      ('3', '3', True),
      ('4', None, True),
      ('5', '5', False),
      ('6', '6', False),
      ('7', None, True)
    ],
    ['training_unique_id','pds_nhs_no', 'true_match']
  )
  
  df_expected = spark.createDataFrame(
  [
    ('1', '1', '1', True, False, False, False),
    ('2', '2', '3', False, True, False, False),
    ('3', None, '3', False, False, True, False),
    ('4', None, None, False, False, False, True),
    ('5', '5', '5', False, True, False, False),
    ('6', None, '6', False, False, False, False),
    ('7', '2', None, False, True, False, False)
  ],
  ['UNIQUE_REFERENCE_other_table', 'NHS_no_pds', 'pds_nhs_no', 'TP', 'FP', 'FN', 'TN']
  )

  df_output = create_clerical_review_confusion_matrix(df_input_match_probabilities, df_input_eval)
  
  assert compare_results(df_output, df_expected, ['UNIQUE_REFERENCE_other_table'])
  
suite_clerical_review_cm.run()

# COMMAND ----------

