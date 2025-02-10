# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import *
from pyspark.sql import DataFrame, SparkSession
import pandas as pd
import statistics

# COMMAND ----------

def get_m_and_u_probabilities(model_as_json):
  '''
  This function creates a dataframe with the m and u probabilities for a specified linker for each variable and each comparison level given to the model for linking. If you have a linker object, the input will be: model_as_json = linker.save_model_to_json()
  '''
  spark = SparkSession.builder \
      .appName("Format Model M and Us") \
      .getOrCreate()
  
  m_and_us = []
  for variable in model_as_json['comparisons']:
    for level in variable['comparison_levels']:
      if 'm_probability' in level:
          m_and_us.append({'variable':variable['output_column_name'],
                           'sql_condition':level['label_for_charts'],'m_probability':level['m_probability']}) 
      if 'u_probability' in level:
        m_and_us[-1]['u_probability'] = level['u_probability']
  m_and_us = pd.DataFrame(m_and_us)

  comparisons_df = spark.createDataFrame(m_and_us)

  return comparisons_df

# COMMAND ----------

def get_average_m_values_from_models(feature_name, df_comparisons, models):
  
  '''
  This function takes the comparisons dictionary for a feature from several models, extracts their M values, takes the average M value, and writes it to the master model.
  
  Parameters
  ----------
    feature_name:str
      for example: "NAME"
    df_comparisons: pandas dataframe
      tabular representation of the comparison levels dictionary for e.g. NAME
    models: Array(dict)
      array of several models, of which we want to take the average
  
  Returns
  -------
    df_comparison_levels: pandas dataframe
      updated table with only the average m value for each level
  
  Overview
  --------
    For the given feature_name, e.g. NAME
    Loop through all the comparison levels for NAME
    Loop through each model (that was trained with different blocking rules)
    Loop through all the comparisons to find the one for NAME
    Loop through all the comparison levels to find the one for the comparison level we want
    Store the m value in m_values
    Take the mean of the m values
    Repeat for the next comparison level
    Various post-processing steps to make the format acceptable to the linker model
  '''

  # extract the comparison levels for this feature name
  df_comparison_levels = pd.DataFrame.from_dict(df_comparisons.loc[df_comparisons['output_column_name'] == feature_name, 'comparison_levels'].item())

  # add a column for m values
  df_comparison_levels['m_probability'] = 0
  
  for level_name in df_comparison_levels['label_for_charts'].tolist():
    m_values = []
    for model in models:
      for comparison in model['comparisons']:
        if comparison['output_column_name'] == feature_name:
          for comparison_level in comparison['comparison_levels']:
            if comparison_level['label_for_charts'] == level_name:
              if 'm_probability' in comparison_level:
                m_values.append(comparison_level['m_probability'])
 
    if(len(m_values) == 0): m_values = [0]
    df_comparison_levels.loc[df_comparison_levels['label_for_charts'] == level_name, ['m_probability']] = statistics.mean(m_values)
  
  # replace NaN with False
  df_comparison_levels['is_null_level'] = df_comparison_levels['is_null_level'].fillna(False)
  # replace zero m values with 0.000001 (otherwise bayes factors don't work)
  df_comparison_levels['m_probability'] = df_comparison_levels['m_probability'].replace(0, 0.000001)
  # ...except the null level... keep the null level's m value as 0
  df_comparison_levels.at[0, 'm_probability'] = 0
  #replace zero or NaN u values with 0.000000001
  df_comparison_levels['u_probability'] = df_comparison_levels['u_probability'].replace(0, 0.000000001)
  df_comparison_levels['u_probability'] = df_comparison_levels['u_probability'].fillna(0.000000001)
  # make the m values sum to 1
  df_comparison_levels['m_probability'] = df_comparison_levels['m_probability']/df_comparison_levels['m_probability'].sum()
  # convert from pandas dataframe back to dictionary
  dict_comparison_levels = pd.DataFrame.to_dict(df_comparison_levels, orient='records')
  # drop m and u values from the Null levels
  del dict_comparison_levels[0]['u_probability']
  del dict_comparison_levels[0]['m_probability']
  
  return dict_comparison_levels

# COMMAND ----------

def save_model(dictionary, description:str, database:str, table:str):
  '''
  This function converts a dictionary to json and writes it to a table.
  '''
  
  json_object = json.dumps(dictionary)
  
  data = [
    (datetime.now(), json_object, description)
  ]

  df = spark.createDataFrame(data=data, schema=['datetime', 'json', 'description'])

  (df
   .write
   .mode('append')
   .format('delta')
   .saveAsTable(f'{database}.{table}')
  )

# COMMAND ----------

def get_model(database:str, table:str, description:str=''):
  '''
  This function returns a trained Splink model based on its description. If no description is given the latest Splink model to have been saved is returned. 
  '''
  if description == '':
    df_splink_models = spark.table(f'{database}.{table}')

    row_object = (
      df_splink_models
      .orderBy('datetime', ascending=False)
      .select('json')
      .head()
    )

    model_latest = row_object[0]

    return json.loads(model_latest)
  
  df_splink_models = spark.table(f'{database}.{table}')
  
  row_object = (
    df_splink_models
    .where(F.col('description') == description)
    .select('json')
    .head()
  )
  
  model = row_object[0]
  
  return json.loads(model)

# COMMAND ----------

def match_probabilities_output(splink_predictions_df:DataFrame, full_evaluation_df:DataFrame):
  '''
  Parameters
    ----------
    splink_predictions_df : dataframe
      The dataframe that splink outputs with all of the match predictions.
    full_evaluation_df : dataframe 
      The dataframe that has all of the original data fed to splink
 
    Returns
    -------
    dataframe
       Dataframe with added rows for those that arent compared by splink, and an added flag for those that did not, to allow evaluation for which records get missed due to too strict blocking rules. 
  '''
  df_predictions = splink_predictions_df.join(full_evaluation_df, full_evaluation_df['UNIQUE_REFERENCE']==splink_predictions_df['UNIQUE_REFERENCE_other_table'], "outer")
  
  df_predictions = df_predictions.withColumn('no_splink_comparisons', F.when(F.col('match_weight').isNull(), True).otherwise(False))

  df_predictions = df_predictions.withColumn('DATE_OF_BIRTH_other_table', F.coalesce(df_predictions['DATE_OF_BIRTH_other_table'], df_predictions['DATE_OF_BIRTH']))
  df_predictions = df_predictions.withColumn('GENDER_other_table', F.coalesce(df_predictions['GENDER_other_table'], df_predictions['GENDER']))
  df_predictions = df_predictions.withColumn('GIVEN_NAME_other_table', F.coalesce(df_predictions['GIVEN_NAME_other_table'], df_predictions['GIVEN_NAME']))
  df_predictions = df_predictions.withColumn('NHS_NO_other_table', F.coalesce(df_predictions['NHS_NO_other_table'], df_predictions['NHS_NO']))
  df_predictions = df_predictions.withColumn('FAMILY_NAME_other_table', F.coalesce(df_predictions['FAMILY_NAME_other_table'], df_predictions['FAMILY_NAME']))
  df_predictions = df_predictions.withColumn('POSTCODE_other_table', F.coalesce(df_predictions['POSTCODE_other_table'], df_predictions['POSTCODE']))
  df_predictions = df_predictions.withColumn('UNIQUE_REFERENCE_other_table', F.coalesce(df_predictions['UNIQUE_REFERENCE_other_table'], df_predictions['UNIQUE_REFERENCE']))

  df_predictions = df_predictions.drop('DATE_OF_BIRTH', 'UNIQUE_REFERENCE', 'GENDER', 'GIVEN_NAME', 'NHS_NO', 'FAMILY_NAME', 'POSTCODE')
  
  return df_predictions

# COMMAND ----------

def get_best_match(
  df_predictions:DataFrame, 
  close_matches_threshold, 
  match_weight_threshold
) -> DataFrame:
  ''' The output of a linker contains all candidate matches with their match_weight.
  For most use cases we would be interested in the single best PDS match for each input record.
  This function chooses the NHS number with the highest match_weight for each input record.
  
  Inputs:
    - df_predictions (DataFrame): dataframe containing all candidate matches and their match weight
    - close_matches_threshold (float): if the NHS number with the highest match_weight has a match_weight close (within this threshold) to the match_weight of a different NHS number, then we consider this a close match. 
  
  '''

   # partition by unique reference NHS no so that if theres exploded data only the exploded version with the best match weight gets kept 
  window_UR_and_NHS_NO_pds = Window.partitionBy("UNIQUE_REFERENCE_other_table", "NHS_NO_pds").orderBy(F.col("match_weight").desc())
  df_predictions_windowed = df_predictions.withColumn("row_number", F.row_number().over(window_UR_and_NHS_NO_pds)).filter(F.col('row_number') == 1).drop("row_number")
  
  window_UR_ordered = Window.partitionBy("UNIQUE_REFERENCE_other_table").orderBy(F.col("match_weight").asc())
  window_UR = Window.partitionBy("UNIQUE_REFERENCE_other_table")
  
  df_predictions_top_weights = (df_predictions_windowed
                                      # Column of maximum match weight for each unique reference
                                      .withColumn('max_weight', F.max('match_weight').over(window_UR))
                                      # Difference between that records mp + the max 
                                      .withColumn('mp_difference', F.round((F.col("max_weight") - F.col("match_weight")), 4)) 
                                      .withColumn('splink_close_match', F.when(((F.col("mp_difference") < close_matches_threshold))
                                                                               , True).otherwise(False)) 
                                      # List of NHS numbers with close matches
                                      .withColumn('close_nhs_nums', 
                                                  F.collect_list(
                                                    F.when(F.col('splink_close_match')==True, F.col('NHS_NO_pds'))
                                                    .otherwise(F.lit(None))).over(window_UR_ordered)
                                                 ) 
                                      #next three commands deal with the fact that F.collectlist only has the full list in the bottom record but thats the one we want 
                                      .withColumn('arr_len', F.size(F.col('close_nhs_nums'))) 
                                      .withColumn("max_size", F.max("arr_len").over(window_UR))
                                      )
  
  df_predictions_top_weights = df_predictions_top_weights.filter(((F.col('arr_len')==F.col('max_size')) & 
                                                                             (F.col('max_weight')==F.col('match_weight')))
                                                                            | (F.col('match_weight').isNull()))
  
  df_predictions_top_weights = (df_predictions_top_weights.withColumn("row_number", 
                                                                                 F.row_number().over(window_UR_ordered))
                                                                     .filter(F.col('row_number') == 1)
                                                                     .drop("row_number")
                                     )
  
#   flag for close match when the list of close matches has more than just itself in it 
  df_predictions_top_weights = (df_predictions_top_weights.withColumn('splink_close_match', 
                                                                                  F.when((F.size(F.col('close_nhs_nums'))>1 ) & 
                                                                                         (F.col('max_weight') >= match_weight_threshold ), True)
                                                                                  .otherwise(False))
                                     )

  df_predictions_top_weights = df_predictions_top_weights.drop('max_weight', 'mp_difference','arr_len','max_size')
  

  return df_predictions_top_weights

# COMMAND ----------

def clean_up(database_name:str):
  '''
  This function drops tables generated from a splink run.
  '''
  
  tables = spark.sql(f'SHOW TABLES IN {database_name} LIKE "__splink*"')

  for row in tables.collect():
    database = row[0]
    table = row[1]

    if database == database_name:
      spark.sql(f'DROP TABLE {database_name}.{table}')

    else:
      spark.sql(f'DROP TABLE {database_name}')

# COMMAND ----------

def change_column_l_and_r_names(df:DataFrame):
  '''
  Parameters
    ----------
    df : dataframe
      Any dataframe where you want to change the _l and _r tables to _pds and _other_table

    Returns
    -------
    dataframe
       Dataframe with columns renamed from _l and _r to _pds and _other_table respectively  
  '''
  current_columns = df.columns
  
  column_mapping = {
    col_name: col_name[:-2] + "_pds" if col_name.endswith("_l") else
               col_name[:-2] + "_other_table" if col_name.endswith("_r") else col_name
    for col_name in current_columns
  }
  
  for old_name, new_name in column_mapping.items():
    df = df.withColumnRenamed(old_name, new_name)
    
  return df