# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import *
from pyspark.sql import DataFrame
from pyspark.sql import Row

import numpy as np
import itertools
from splink.spark.spark_linker import SparkLinker
import pandas as pd

import os
if not os.getcwd().startswith("/home/spark"):
  import sys

  current_file_path = os.path.abspath(__file__)
  current_dir_path = os.path.dirname(current_file_path)
  parent_dir_path = os.path.dirname(current_dir_path)
  sys.path.append(parent_dir_path)

  os.environ["PYSPARK_PYTHON"] = "python"
  from utils.dataset_ingestion_utils import *

# COMMAND ----------

def add_confusion_matrix_to_df(df, match_weight_threshold = 5):
  ''' Compares splink results to MPS results.
  In this context, MPS is considered to be the ground truth, with GOLD LABEL_TYPE representing an MPS match we are confident of, and SILVER representing a less confident MPS match.
  The Splink result is then designated as a true_positive, false_positive, etc, based on several rules.
  The allocation of confusion label is stored as boolean in several columns. For example the column true_positives contains True or False.
  '''
  confusion_matrix = (df
    .withColumn('true_positives',
                F.when(
                  ( (F.col('LABEL_TYPE') == 'GOLD') | (F.col('LABEL_TYPE') == 'SILVER') ) & 
                  (F.col('LABEL') == F.col('NHS_NO_pds')) & 
                  (F.col('match_weight') > match_weight_threshold) &
                  (F.col('splink_close_match') == False), 
                  True
                ).otherwise(False)
               )
    .withColumn('false_positives',
               F.when(
                 (F.col('LABEL_TYPE') == 'GOLD') &
                 (F.col('LABEL') != F.col('NHS_NO_pds')) &
                 (F.col('match_weight') > match_weight_threshold) &
                 (F.col('splink_close_match') == False), 
                  True
               ).otherwise(False)
               )
    .withColumn('possible_false_positives',
               F.when(
                 (((F.col('LABEL_TYPE') == 'SILVER') | (F.col('LABEL_TYPE') == 'no label') ) &
                 (F.col('match_weight') > match_weight_threshold) &
                 ((F.col('LABEL') != F.col('NHS_NO_pds')) | F.col('LABEL').isNull()) &
                 (F.col('splink_close_match') == False)) |
                 ((F.col('mps_multiple_pds_matches') == True) & 
                  (F.col('match_weight') > match_weight_threshold) &
                   (F.col('splink_close_match') == False)
                 ), 
                 True
               ).otherwise(False)
               )
    .withColumn('false_negatives',
               F.when(
                 (F.col('LABEL_TYPE') == 'GOLD') &
                 ((F.col('match_weight') <= match_weight_threshold) | ( F.col('splink_close_match') == True )), 
                  True
               ).otherwise(False)
               )
    .withColumn('possible_false_negatives',
               F.when(
                 (F.col('LABEL_TYPE') == 'SILVER') &
                 ((F.col('match_weight') <= match_weight_threshold) | (F.col('splink_close_match') == True)), 
                 True
               ).otherwise(False)
               )
    .withColumn('true_negatives',
               F.when(
                 ((F.col('LABEL_TYPE') == 'no label') &
                 ((F.col('match_weight') <= match_weight_threshold) 
                   | (F.col('splink_close_match') == True))) | 
                 ((F.col('mps_multiple_pds_matches') ==  True) &
                 (F.col('splink_close_match') == True)),
                  True
               ).otherwise(False)
               )
  )
  return confusion_matrix

# COMMAND ----------

def filter_confusion_matrix(df:DataFrame, subset:str):
  ''' Filter the result of add_confusion_matrix_to_df
  So that we can display and clerically review all false_positives, for example.
  '''
  df = df.filter(F.col(subset)==True)
  return df

# COMMAND ----------

def load_mps_responses(df_mps_responses, df_pds_replaced_by):
  '''Take MPS responses and:
    - derive the mps_successful_step
    - flag if MPS had multiple matches
    - replace superseded NHS numbers with their current NHS number (based on df_pds_replaced_by)
  '''
  df_mps_responses = (
    df_mps_responses
    .withColumn('mps_successful_step',
      F.when(
        (F.col('MATCHED_ALGORITHM_INDICATOR') == 1) & 
        (F.col('MATCHED_CONFIDENCE_PERCENTAGE') == 100) & 
        (F.col('ALGORITHMIC_TRACE_DOB_SCORE_PERC').isNull()), 'CCT_cached'
      )
      .when(
        (F.col('MATCHED_ALGORITHM_INDICATOR') == 1) & 
        (F.col('MATCHED_CONFIDENCE_PERCENTAGE') == 100) & 
        (F.col('ALGORITHMIC_TRACE_DOB_SCORE_PERC') == 0), 'CCT_live'
      )
      .when(                                            
        (F.col('MATCHED_ALGORITHM_INDICATOR') == 3) & 
        (F.col('MATCHED_CONFIDENCE_PERCENTAGE') == 100), 'alphanumeric_trace_live'
      )
      .when(
        (F.col('MATCHED_ALGORITHM_INDICATOR') == 4) & 
        (F.col('MATCHED_CONFIDENCE_PERCENTAGE') >= 50), 'algorithmic_trace_live'
      )
      .when(
        (F.col('MATCHED_ALGORITHM_INDICATOR') == 0), 'No_PDS_tracing_run'
      ) 
      .when(
        (F.col('MATCHED_ALGORITHM_INDICATOR') != 0) & 
        (F.col('MATCHED_CONFIDENCE_PERCENTAGE') == 0), 'No_PDS_match_found'
      )                                      
      .otherwise(None)
    )
    
    .withColumn('mps_multiple_pds_matches', 
      F.when(
        (F.col('ERROR_SUCCESS_CODE') == '97'),
        True
      )
      .otherwise(
        False
      )
    )
    
    .select(
      'UNIQUE_REFERENCE',
      F.col('MATCHED_NHS_NO').alias('NHS_NO_mps'),
      F.col('GIVEN_NAME').alias('GIVEN_NAME_mps'),
      F.col('FAMILY_NAME').alias('FAMILY_NAME_mps'),
      F.col('DATE_OF_BIRTH').alias('DATE_OF_BIRTH_mps'),
      F.col('POSTCODE').alias('POSTCODE_mps'),
      F.col('GENDER').alias('GENDER_mps'),
      F.col('created').alias('mps_run_date').cast('Date'),
      'mps_successful_step',
      'mps_multiple_pds_matches',
      'ALGORITHMIC_TRACE_FAMILY_NAME_SCORE_PERC',
      'ALGORITHMIC_TRACE_GIVEN_NAME_SCORE_PERC',
      'ALGORITHMIC_TRACE_DOB_SCORE_PERC',
      'ALGORITHMIC_TRACE_GENDER_SCORE_PERC',
      'ALGORITHMIC_TRACE_POSTCODE_SCORE_PERC',
    )
  )
  
  # Replace superseded NHS numbers with current NHS numbers, so that there are no superseded NHS numbers in the mps data.
  df_mps_responses = update_superseded_nhs_numbers(df_mps_responses, df_pds_replaced_by, 'NHS_NO_mps')
  
  return df_mps_responses

# COMMAND ----------

def join_with_mps_responses(df, df_response):
  ''' For evaluation comparing to the matches found by MPS, we need to retrive the Person ID returned by MPS.
  This function joins the Splink result to the MPS response, for comparison.
  It also returns some of the MPS metadata, so that it can be evaluated how MPS got to the result.
  '''
  df = ( df
    .join(
      df_response,
      df['UNIQUE_REFERENCE_other_table']==df_response['UNIQUE_REFERENCE'],
      'left'
    )
    .select(
      'UNIQUE_REFERENCE', 'LABEL_TYPE', 'LABEL', 'mps_run_date', 'match_weight', 'match_probability', 
      'source_dataset_pds', 'UNIQUE_REFERENCE_pds', 'source_dataset_other_table', 'UNIQUE_REFERENCE_other_table', 
      'NHS_NO_pds', 'NHS_NO_other_table', 'NHS_NO_mps', 
      'GIVEN_NAME_pds', 'GIVEN_NAME_other_table', 'GIVEN_NAME_mps', 'gamma_NAME', 
      'FAMILY_NAME_pds', 'FAMILY_NAME_other_table', 'FAMILY_NAME_mps', 'gamma_NAME', 
      'DATE_OF_BIRTH_pds', 'DATE_OF_BIRTH_other_table', 'DATE_OF_BIRTH_mps', 'gamma_DATE_OF_BIRTH', 
      'POSTCODE_pds', 'POSTCODE_other_table', 'POSTCODE_mps', 'gamma_POSTCODE', 
      'GENDER_pds', 'GENDER_other_table', 'GENDER_mps', 'gamma_GENDER',
      'match_key', 'no_splink_comparisons', 'agreement_patterns', 'mps_agreement_pattern',
      'mps_successful_step', 'splink_close_match', 'close_nhs_nums', 'mps_multiple_pds_matches', 
      'ALGORITHMIC_TRACE_FAMILY_NAME_SCORE_PERC', 'ALGORITHMIC_TRACE_GIVEN_NAME_SCORE_PERC', 'ALGORITHMIC_TRACE_DOB_SCORE_PERC',
      'ALGORITHMIC_TRACE_GENDER_SCORE_PERC', 'ALGORITHMIC_TRACE_POSTCODE_SCORE_PERC',
    )
  )
  
  return df

# COMMAND ----------

def add_agreement_pattern(df):
  '''Combine the values in gamma columns into an agreement pattern.
  For example if gamma_given_name = 2, gamma_family_name = 1, gamma_postcode = 5, then agreement pattern = 215
  '''
  # Extract gamma columns
  gamma_columns = [col for col in df.columns if col.startswith('gamma')]
  # Apply the function to concatenate gamma columns
  df = df.withColumn('agreement_patterns', F.concat_ws('', *[df[col] for col in gamma_columns]))
  return df

# COMMAND ----------

def add_to_labelled_data(new_rows, override = False):
  '''
  new_rows: List of lists, each list in the list is a row that you want to add to the labelled data, with the list items being as follows:
    index 0: training_unique_id, so the unique id of the NPEX records in the comparison you are registering a clerical review for
    index 1: pds_nhs_no, the matching PDS NHS number for the record you are registering a clerical review for, this could be set as None if you believe that the training record has DQ that should be too bad to ever be able to find a match for 
    index 2: True or False. True if you are recording that index 0 and index 1 records are a true match, False if you are recording that they are being incorrectly matched. If you left index 1 as None for the reasons described above, then set index 2 to True
    
    Examples: 
    add_to_labelled_data([['1', '1', True]]) - in this situation you are saying that the record with unique ID of 1 is a true match to NHS number '1' after clerical review
    
    add_to_labelled_data(([['1', '1', True], ['2', '2', True]])) - this will add several rows 
  '''
  
  clerical_labels = spark.table('mps_enhancement_collab.clerical_review_labels')
  
  if not override:
    for row in new_rows:
      
      check_if_present_different_nhs_no = clerical_labels.filter((clerical_labels.training_unique_id == row[0]) & (clerical_labels.pds_nhs_no != row[1])).limit(1).first()
      check_if_present_true_match = None
      check_if_present_false_match = None
      
      if (check_if_present_different_nhs_no is not None) & (row[2] == True):
        print(f'TRAINING_UNIQUE_REFERENCE {row[0]} is already present in the labelled data, matched to a different NHS number ({check_if_present_different_nhs_no.pds_nhs_no}) to the one you\'ve given \n')
        print(f'Run your query again without the row {row} or override this check by giving the function override=True \n')
        print('!! Be aware that none of the records in the current function call have been saved due to this duplication issue. !! \n')
        return clerical_labels
      
      if row[2] == False:
        check_if_present_true_match = clerical_labels.filter((clerical_labels.training_unique_id == row[0]) & (clerical_labels.pds_nhs_no == row[1]) & (clerical_labels.true_match == True)).limit(1).first()
      
      if (check_if_present_true_match is not None):
        print(f'training unique reference {row[0]} is already present in the labelled data, matched to the same NHS number')
        print('you previously labelled this as a True match, whereas now you are labelling as a false match \n')
        print(f'Run your query again without the row {row} or override this check by giving the function override=True \n')
        print('!! Be aware that none of the records in the current function call have been saved due to this duplication issue. !! \n')
        return clerical_labels
      
      if row[2] == True:
        check_if_present_false_match = clerical_labels.filter((clerical_labels.training_unique_id == row[0]) & (clerical_labels.pds_nhs_no == row[1]) & (clerical_labels.true_match == False)).limit(1).first()
      
      if (check_if_present_false_match is not None):
        print(f'training unique reference {row[0]} is already present in the labelled data, matched to the same NHS number')
        print('you previously labelled this as a False match, whereas now you are labelling as a True match \n')
        print(f'Run your query again without the row {row} or override this check by giving the function override=True \n')
        print('!! Be aware that none of the records in the current function call have been saved due to this duplication issue. !! \n')
        return clerical_labels
      
  
  formatted_rows = [Row(training_unique_id=row[0], pds_nhs_no=row[1], true_match=row[2]) for row in new_rows]
  
  rows_to_add = spark.createDataFrame(formatted_rows)
  
  updated_clerical_labels = clerical_labels.union(rows_to_add)
  
  updated_clerical_labels = updated_clerical_labels.dropDuplicates()
  
  updated_clerical_labels.write.mode('overwrite').saveAsTable('mps_enhancement_collab.clerical_review_labels')
  
  return updated_clerical_labels

# COMMAND ----------

def create_clerical_review_confusion_matrix(match_probabilities_df, clerical_labels_df):
  '''
  creates a confusion matrix using the evaluation table created using add_to_labelled_data. 
  '''
  
  mp_df = match_probabilities_df.join(clerical_labels_df, match_probabilities_df.UNIQUE_REFERENCE_other_table == clerical_labels_df.training_unique_id, 'left')

  confusion_matrix_df = (mp_df
                            .withColumn('TP', F.when((F.col('NHS_no_pds')==F.col('pds_nhs_no')) & 
                                                    (F.col('true_match')==True), True
                                                    ).otherwise(False)
                                       )
                            .withColumn('FP', F.when(((F.col('NHS_no_pds')!=F.col('pds_nhs_no')) & 
                                                    (F.col('true_match')==True)) |
                                                     ((F.col('NHS_no_pds')==F.col('pds_nhs_no')) &
                                                     (F.col('true_match')==False)) | 
                                                     ((F.col('pds_nhs_no').isNull()) & 
                                                      (F.col('true_match')==True) & 
                                                      (F.col('NHS_no_pds').isNotNull())), True
                                                    ).otherwise(False)
                                       )
                            .withColumn('FN', F.when(((F.col('NHS_no_pds').isNull()) &
                                                     (F.col('true_match')==True) &
                                                     (F.col('pds_nhs_no').isNotNull())), True
                                                    ).otherwise(False)
                                       )
                            .withColumn('TN', F.when((F.col('pds_nhs_no').isNull()) &
                                                    (F.col('true_match')==True) &
                                                    (F.col('NHS_no_pds').isNull()), True
                                                    ).otherwise(False)
                                       )
                           )
  
  confusion_matrix_df = confusion_matrix_df.drop('true_match', 'training_unique_id')
  
  
  return confusion_matrix_df


def expand_a_listed_column(df:DataFrame, list_column: str) -> DataFrame:
    """
    This function takes a column with a list type, and expands out the items in this list into seperate columns,
    to the maximum length of a value in the listed column.
    
    When say two values are given in a list, but the maximum is 4, the first value of the list is repeated.
    
    Args:
      df (DataFrame): Dataframe with a listed column with values you want to expand out.
      list_column (str): This is a column that has a list of string values that you want to expand out into it's own set of column.
    
    Returns:
      A new DataFrame with the expanded columns and null values filled with the first value provided.
    """
    # Step 1: Add a column to count the length of the list in each row
    df = df.withColumn("list_count", F.size(F.col(list_column)))

    # Step 2: Initialize an empty result DataFrame
    result_df = None

    # Step 3: Start the iteration over possible list counts (starting from 1)
    count = 2
    column_names = set()  # To keep track of all column names we create

    while True:
        # Filter rows that have the list_count equal to the current count
        filtered_df = df.filter(F.col("list_count") == count)

        # Check if the filtered DataFrame is empty using .count()
        if filtered_df.count() == 0:
            # Stop if the DataFrame is empty
            break

        # Step 4: Create columns linked_nhs_no_{1}, linked_nhs_no_{2}, etc.
        for i in range(count):
            col_name = f'linked_nhs_no_{i+1}'
            column_names.add(col_name)
            filtered_df = filtered_df.withColumn(col_name, 
                                                 F.when(F.size(F.col(list_column)) > i, F.col(list_column)[i]).otherwise(None))

        # Add missing columns with null values to the filtered DataFrame
        for col_name in column_names:
            if col_name not in filtered_df.columns:
                filtered_df = filtered_df.withColumn(col_name, F.lit(None))

        # Append the current filtered_df to the result_df
        if result_df is None:
            result_df = filtered_df
        else:
            # Add missing columns to result_df
            for col_name in column_names:
                if col_name not in result_df.columns:
                    result_df = result_df.withColumn(col_name, F.lit(None))
            
            result_df = result_df.union(filtered_df)

        # Increment count for the next iteration
        count += 1

    # Step 5: Fill nulls with the first column (linked_nhs_no_1) after the union
    max_cols = count - 1  # Since the last count increment happens after the break
    for i in range(2, max_cols + 1):
        result_df = result_df.withColumn(f'linked_nhs_no_{i}', 
                                         F.when(F.col(f'linked_nhs_no_{i}').isNull(), F.col("linked_nhs_no_1")).otherwise(F.col(f'linked_nhs_no_{i}')))

    # Step 6: Drop the helper list_count column (optional)
    result_df = result_df.drop("list_count")

    return result_df


# COMMAND ----------

def create_gamma_pattern_df_from_linked_duplicates(df_evaluation: DataFrame, comparisons: List[Any]) -> DataFrame:
  """
  This function uses a linker model and only gets out the agreement patterns for each duplicate combination in the dataframe provided.

  Args:
    df_evaluation (DataFrame): Dataframe where the listed column has been expanded out, and there is a set of columns with "linked_nhs_no_" in them.
    comparisons (List[Any]): This is a list of all the comparisons used to generate the agreement patterns for the model you are evaluation.

  Returns:
    A DataFrame with the gamma pattern for each duplicated define extracted out.
  """
  # Create columns to represent all NHS numbers that are superseded for each individual.
  df_evaluation_expands = expand_a_listed_column(df_evaluation, "LINKED_NHS_NO")
  # Extract out the number of values with seeded individuals   
  linked_columns = [col for col in df_evaluation_expands.columns if col.startswith("linked_nhs_no_")]
  # Generate the gamma patterns needed to get out the agreement pattern for each duplicate.   
  gamma_pattern_br = [f"l.NHS_NO = r.{col}"for col in linked_columns]
  
  # Define settings
  settings = {
  'link_type': 'dedupe_only',
  'unique_id_column_name': params.TRAINING_UNIQUE_REFERENCE_COLUMN,
  'comparisons': comparisons,
  'blocking_rules_to_generate_predictions': gamma_pattern_br,
  "retain_matching_columns": True,
  "retain_intermediate_calculation_columns": False,
  "max_iterations": 1,
  "em_convergence": 0.01,
  }
  
  try:
    clean_up(params.DATABASE)
  except Exception as e:
    None
  
  # Define Linker
  linker_evaluation = SparkLinker(
    df_evaluation_expands,
    settings,
    database = params.DATABASE,
    break_lineage_method = 'persist',
    register_udfs_automatically = False
  )
  
  gamma_pairwise = linker_evaluation.predict()
  gamma_pattern_df = spark.table(gamma_pairwise.physical_name)
  gamma_pattern_df = change_column_l_and_r_names(gamma_pattern_df)
  
  #Adds agreement patterns    
  gamma_pattern_df = add_agreement_pattern(gamma_pattern_df)
  gamma_pattern_df = gamma_pattern_df.select("NHS_NO_pds", "NHS_NO_other_table", "agreement_patterns")
  gamma_pattern_df = gamma_pattern_df.withColumnRenamed("agreement_patterns", "all_agreement_patterns")
  
  return gamma_pattern_df

# COMMAND ----------

def combine_predictions_and_gamma_pattern_df(df_predictions:DataFrame, gamma_pattern_df:DataFrame, df_evaluation:DataFrame, thresholds: List[float]) -> DataFrame:
  """
  Takes the predictions extracted out from the model and merges it with the duplicates expected to be identified from the model.

  Args:
    df_predictions (DataFrame): Dataframe produced from the predictions out of a linker.predict function. This dataframe also needs the agreement_patterns attached.
    gamma_pattern_df (DataFrame): DataFrame with the gamma pattern for each duplicated define extracted out.
    df_evaluation (DataFrame): Dataframe where the listed column has been expanded out, and there is a set of columns with "linked_nhs_no_" in them.
    thresholds (List[float]): This is a list of thresholds you want to set to test for the threshold of match probabilities.

  Returns:
    A DataFrame that has combined information from df_predictions, gamma_pattern_df, and df_evaluation, with defined values depending on the threshold used.

  """
  # Join all expected predictions (gamma_pattern_df) with all predictions created from the model (df_predictions)   
  combine_df= df_predictions.join(gamma_pattern_df, on=["NHS_NO_pds", "NHS_NO_other_table"], how="outer")
  # Extract out the NHS_NO with their LINKED_NHS_NO from the evaluation dataframe, and rejoin this on to the combine_df.    
  linked_nhs_no = df_evaluation.select("NHS_NO", "LINKED_NHS_NO")
  combine_df = combine_df.join(linked_nhs_no, combine_df.NHS_NO_pds == linked_nhs_no.NHS_NO, how="left")
  
  # Label true duplicates and those missed from blocking rules.   
  combine_df = combine_df.withColumn("true_duplicate", F.array_contains(F.col("LINKED_NHS_NO"), F.col("NHS_NO_other_table")))
  combine_df = combine_df.withColumn("missed_duplicate", (F.col("true_duplicate") & F.col("match_probability").isNull()))
  
  # Create a set of columns which determines whether a duplicate fulls above or below the threshold from the threshold defined.   
  for threshold in thresholds:
    thresh_str = "_".join(str(threshold).split("."))
    combine_df = combine_df.withColumn(f"above_threshold_{thresh_str}", F.col("match_probability") >= threshold)
    combine_df = combine_df.withColumn(f"below_threshold_{thresh_str}", F.col("match_probability") < threshold)
    
  return combine_df

# COMMAND ----------

def create_confusion_matrix_per_threshold(combine_df:DataFrame, thresholds:List[float])-> pd.DataFrame:
  """
  Takes a dataframe with different values for different thresholds

  Args:
    combine_df (DataFrame): DataFrame that has combined information from df_predictions, gamma_pattern_df, and df_evaluation, with counts of idenfied duplicates or not depending on the thresholds used.
    thresholds (List[float]): This is a list of thresholds you want to set to test for the threshold of match probabilities.

  Returns:
    A DataFrame that generates the TP, FP, TN, FN rates from the data provided. (This can take a long time to run.)

  """
  
  threshold_values = list()

  for threshold in thresholds:
    thresh_str = "_".join(str(threshold).split("."))
    tp_count = combine_df.filter(F.col(f"above_threshold_{thresh_str}") & F.col("true_duplicate")).count()
    fp_count = combine_df.filter(F.col(f"above_threshold_{thresh_str}") & ~F.col("true_duplicate")).count()
    tn_count = combine_df.filter(F.col(f"below_threshold_{thresh_str}") & ~F.col("true_duplicate")).count()
    fn_count = combine_df.filter(F.col(f"below_threshold_{thresh_str}") & F.col("true_duplicate")).count()

    threshold_values.append({
      "threshold": threshold,
      "tp": tp_count,
      "fp": fp_count,
      "tn": tn_count,
      "fn": fn_count
    })

  threshold_df = pd.DataFrame(threshold_values)

  threshold_df["tpr"] = threshold_df.tp/(threshold_df.tp + threshold_df.fn)
  threshold_df["fpr"] = threshold_df.fp/(threshold_df.fp + threshold_df.tn)
  
  return threshold_df

# COMMAND ----------

def categorise_by_count_and_aggregate(
  df:DataFrame, 
  column_name:str, 
  agg_column:str, 
  thresholds:List[int], 
  exact_values=None, 
  label_column:str="category", 
  sort_column:str="threshold_sort"
) -> DataFrame:
    """
    Groups a column of integer values into categories based on thresholds, then aggregates another column by category.
    
    Args:
      df (DataFrame): The input DataFrame.
      column_name (str) : The column name that contains the integer values to be categorised.
      agg_column (str) : The column whose values will be concatenated after grouping by category.
      thresholds List[int] : list of integers you want to create category ranges from. i.e. [0, 100] -> [0 - 99, 100+]
      exact_values List[iint] : A list of exact values to match separately (e.g., [1, 2]).
      label_column (str) : The name of the new column that will contain the categories.
      sort_column (str) : The name of the column to store threshold values for sorting.
    
    Returns:
      df (DataFrame): The DataFrame with categories, aggregated, and sorted by thresholds.
    """
    # List to hold conditions for categorical labels
    conditions = []
    sort_conditions = []
    
    # Handle exact values if provided
    if exact_values:
        for exact_value in exact_values:
            conditions.append(F.when(F.col(column_name) == exact_value, f"== {exact_value}"))
            sort_conditions.append(F.when(F.col(column_name) == exact_value, exact_value))

    # Handle less than the first threshold
    conditions.append(F.when(F.col(column_name) < thresholds[0], f"< {thresholds[0]}"))
    sort_conditions.append(F.when(F.col(column_name) < thresholds[0], thresholds[0] - 1))

    # Iterate over pairs of thresholds to define ranges (e.g., 10-99)
    for i in range(1, len(thresholds)):
        conditions.append(
            F.when((F.col(column_name) >= thresholds[i-1]) & (F.col(column_name) < thresholds[i]), 
                   f"{thresholds[i-1]} - {thresholds[i]-1}")
        )
        sort_conditions.append(
            F.when((F.col(column_name) >= thresholds[i-1]) & (F.col(column_name) < thresholds[i]), thresholds[i-1])
        )
    
    # Handle values greater than or equal to the last threshold
    conditions.append(F.when(F.col(column_name) >= thresholds[-1], f">= {thresholds[-1]}"))
    sort_conditions.append(F.when(F.col(column_name) >= thresholds[-1], thresholds[-1]))

    # Chain the conditions using coalesce to form the final categorized column and sorting column
    categorized_column = F.coalesce(*conditions)
    threshold_sort_column = F.coalesce(*sort_conditions)

    # Add the categorized and sorting columns to the DataFrame
    df = df.withColumn(label_column, categorized_column).withColumn(sort_column, threshold_sort_column)
    
    # Group by the category and concatenate the 'agg_column' values
    df_grouped = df.groupBy(label_column, sort_column) \
                   .agg(F.concat_ws(", ", F.collect_list(agg_column)).alias("aggregated_values")) \
                   .orderBy(F.col(sort_column).desc())

    return df_grouped