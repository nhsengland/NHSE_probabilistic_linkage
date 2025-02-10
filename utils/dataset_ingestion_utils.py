# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from typing import *
from collections import defaultdict, deque

# COMMAND ----------

#%run ../parameters_linking

# COMMAND ----------

def remove_confidential_from_pds(
  df_pds_full: DataFrame,
) -> DataFrame:
  '''Remove records from PDS if their confidentiality code is I, S, or Y.
  
  Args:
    df_pds_full (DataFrame): Input PDS dataframe.
    
  Returns:
    df_pds_full_without_confidential (DataFrame): PDS dataframe with confidential records removed.
    
  '''  
  df_pds_full_without_confidential = (
    df_pds_full
    .filter(
      (~(F.col('ConfidentialityCode.code')[0].isin(['I', 'S', 'Y']))) |
      F.col('ConfidentialityCode.code')[0].isNull()
    )
  )
  return df_pds_full_without_confidential

# COMMAND ----------

def remove_superseded_from_pds(
  df_pds_full: DataFrame,
  df_pds_replaced_by: DataFrame
) -> DataFrame:
  '''Remove records from PDS if they are superseded.
  
  A record is superseded if its NHS number appears in the pds.replaced_by table.
  
  Args:
    df_pds_full (DataFrame): Input PDS dataframe.
    
  Returns:
    df_pds_full_without_superseded (DataFrame): PDS dataframe with superseded records removed.
  
  '''  

  df_pds_full_without_superseded = (
    df_pds_full
    .join(
      df_pds_replaced_by,
      (df_pds_full['nhs_number'] == df_pds_replaced_by['nhs_number']),
      'left_anti'
    )
  )
  return df_pds_full_without_superseded 

# COMMAND ----------

def update_superseded_nhs_numbers(
  df_input: DataFrame, 
  df_pds_replaced_by: DataFrame, 
  col_to_update: str,
) -> DataFrame:
  ''' Update superseded NHS numbers to their current NHS number.
  
  This function performs a lookup of NHS numbers in the pds.replaced_by table.
  If the NHS number appears in pds.replaced_by, it will be updated.
  If the NHS number does not appear in pds.replaced_by, then it remains the same.

  Args:
    df_input (DataFrame): Input dataframe.
    df_pds_replaced_by (DataFrame): PDS replaced by table.
    col_to_update (str): The name of a column in df_input which contains NHS numbers that we want to update.
    
  Returns:
    df_output (DataFrame): Dataframe with NHS numbers updated.
    
  '''
  
  df_pds_replaced_by = df_pds_replaced_by.select('nhs_number', 'replaced_by')
  
  df_output = (
    df_input
    .join(
      df_pds_replaced_by,
      df_input[col_to_update] == df_pds_replaced_by['nhs_number'],
      'left'
    )
    .withColumn(
      col_to_update,
      F.when(F.col('replaced_by').isNotNull(), F.col('replaced_by'))
       .otherwise(F.col(col_to_update))
    )
    .drop('nhs_number', 'replaced_by')
  )
  return df_output

# COMMAND ----------

def preprocess_full_pds(
  df_pds_full: DataFrame,
  df_pds_replaced_by: DataFrame = [],
  remove_confidential_records: bool = True,
  remove_superseded_records: bool = True,
) -> DataFrame:
  '''Perform some preprocessing specific to PDS.
  
  Specific preprocessing:
    - Remove confidential records (optional).
    - Remove superseded records (optional).
    - Change column names to the names used in the splink linkage model.
    - Add a unique reference, as required by the splink linkage model.
    - Add a label, which can be used in splink model training and/or evaluation. The label is the NHS number.
    
  Args:
    df_pds_full (DataFrame): PDS full dataframe, to be preprocessed. Must contain at least the nhs_number, dob, gender_code, preferred_name, and home_address columns.
    df_pds_replaced_by (DataFrame, optional): Dataframe which maps NHS numbers that have been superseded to their superseding value. Only required if remove_superseded_records is True.
    remove_confidential_records (bool): flag to indicate whether confidential records should be removed.
    remove_superseded_records (bool): flag to indicate whether superseded records should be removed.
    
  Returns:
    df_pds_full (DataFrame): PDS dataframe preprocessed for the splink linkage model.
  
  '''
  
  if remove_confidential_records:
    df_pds_full = remove_confidential_from_pds(df_pds_full)
    
  if remove_superseded_records:
    df_pds_full = remove_superseded_from_pds(df_pds_full,df_pds_replaced_by)
  
  df_pds_full = (
    df_pds_full
    .withColumn('UNIQUE_REFERENCE', F.monotonically_increasing_id())
    .select(
      'UNIQUE_REFERENCE',
      F.col('nhs_number').alias('LABEL'),
      F.col('nhs_number').alias('NHS_NO'),
      F.lower(F.concat_ws(' ', F.col('preferred_name.givenNames'))).alias('GIVEN_NAME'),
      F.lower(F.col('preferred_name.familyName')).alias('FAMILY_NAME'),
      F.col('gender_code').alias('GENDER'),
      F.col('dob').alias('DATE_OF_BIRTH'),
      F.upper(F.col('home_address.postalCode')).alias('POSTCODE'),
    )
  )
  
  return df_pds_full

# COMMAND ----------

def load_data_to_link(
  DATABASE, 
  TABLE_NAME, 
  TRAINING_GIVEN_NAME_COLUMN, 
  TRAINING_FAMILY_NAME_COLUMN,
  TRAINING_GENDER_COLUMN,
  TRAINING_POSTCODE_COLUMN,
  TRAINING_DOB_COLUMN,
  TRAINING_UNIQUE_REFERENCE
) -> DataFrame:
  ''' Loads training data from a table into a dataframe and assigns the correct column names 
  
  Args:
    - DATABASE, TABLE_NAME (str): the location of the table to be loaded
    - TRAINING_xxx (str): the correct names to assign to the columns
    
  Returns:
    - df_training (DataFrame): dataframe with the correct column names to be used in training
  '''
  
  df_training = (
    spark.table(f'{DATABASE}.{TABLE_NAME}')
    .withColumn(TRAINING_GIVEN_NAME_COLUMN, F.lower(F.col('GIVEN_NAME')))
    .withColumn(TRAINING_FAMILY_NAME_COLUMN, F.lower(F.col('FAMILY_NAME')))
    .withColumnRenamed(TRAINING_GENDER_COLUMN, 'GENDER')
    .withColumnRenamed(TRAINING_POSTCODE_COLUMN, 'POSTCODE')
    .withColumnRenamed(TRAINING_DOB_COLUMN, 'DATE_OF_BIRTH')
    .withColumnRenamed(TRAINING_UNIQUE_REFERENCE,'UNIQUE_REFERENCE')
  )
  return df_training

# COMMAND ----------

def explode_historical_values(
  df_input: DataFrame,
  column_name: str,
  case: str = 'lower'
) -> DataFrame:
  ''' Explodes the family name or address history from PDS.

  Specifics:
    - Deduplicates if there are repeats of the same value.
    - Standardises all values to the same case (default lower case) so that this deduplication is case-insensitive.
  
  Args:
    df_input (DataFrame): dataframe from PDS, which must contain at least the nhs_number and family_name_history or address_history columns.
    column_name (str): name of the column to be returned. For example, if column_name is "address", this function will explode the column "address_history" into a new column named "address".
    case (str): Case (lower or upper) to standardise values to. Defaults to 'lower'.
  
  Returns:
    df_exploded (DataFrame): PDS dataframe exploded, i.e. with one row for each historical value.
    
  Example:
    Input: 
      [None, 'name_1', 'name_2', 'NAME_2']
    Output: 
      'name_1'
      'name_2'

  '''
  
  column_historical_name = column_name + '_history'
  
  df_exploded = (
    df_input
    .select('nhs_number', column_historical_name)
    
    # remove duplicates and nulls from the arrays of historical values.
    .withColumn(
      'null_array_elements_removed',
      F.when(
        F.col(column_historical_name).isNotNull(),
        F.array_except(F.col(column_historical_name), F.array(F.lit(None)))
      )
      .otherwise(None)
    )
    
    # explode to get a row for each historical value.
    # this step uses explode_outer, so that if all the historical values for this patient are null, it retains one row with null value.
    # we do want to keep at least one row for each patient, even if there were no non-null values.
    .withColumn('exploded', F.explode_outer('null_array_elements_removed')) 
    
    # standardise case
    .withColumn(
      column_name,
      F.when(
        (F.col('exploded').isNotNull()) & (F.lit(case == 'lower')),
        F.lower(F.col('exploded'))
      )
      .when(
        (F.col('exploded').isNotNull()) & (F.lit(case == 'upper')),
        F.upper(F.col('exploded'))
      )
      .otherwise(None)
    )
    .select('nhs_number', column_name)
    
    # after the previous transformations, some rows may now be identical, so drop duplicates.
    .dropDuplicates() 
  )
  
  return df_exploded

# COMMAND ----------

def explode_historical_given_names(
  df_input: DataFrame, 
  case: str = 'lower',
) -> DataFrame:
  ''' Explodes the given name history from PDS.

  In PDS, given name history is different to family name history and address history, in that each historical entry in given name is itself an array of multiple names, i.e. the first name and middle names of a patient.
  This is why the get_historical_values function will not work on given names, so we need this different function: get_historical_given_names, in which the steps are in a different sequence.
  An accompanying document (explaining pds explode) provides examples of how the two functions operate.
  
  Other specifics:
    - Removes null and empty names from the arrays of first and middle names.
    - Deduplicates if there are repeats of the same given name.
    - Standardises all given names to the same case (default lower case) so that this deduplication is case-insensitive.
    - In the output, each array of multiple given names is concatenated with whitespace into a single string.
    - Note: Does not deduplicate names within the array of first and middle names.
  
  Args:
    df_input (DataFrame): dataframe from PDS, which must contain at least the nhs_number and given_name_history columns.
    case (str): Case (lower or upper) to standardise given names to.
  
  Returns:
    df_exploded (DataFrame): PDS dataframe exploded on given name history. i.e. with one row for each historical given name.
    
  Example:
    Input: 
      [[], ['name_1'], ['name_1', None], ['name_2_part_1', 'name_2_part_2', 'NAME_2_PART_2']]
    Output: 
      ['name_1']
      ['name_2_part_1 name_2_part_2 name_2_part_2']
      
  '''
  
  df_exploded = (
    df_input
    .select('nhs_number', 'given_name_history')
    
    # explode to get a row for each historical given name.
    # this step uses explode_outer, so that if all the historical given names for this patient are null, it retains one row with null given name.
    .withColumn('exploded', F.explode_outer('given_name_history')) 
    
    # each given name is an array containing first name and middle names.
    # this step uses array_except to remove nulls and duplicates from these arrays.
    .withColumn(
      'null_array_elements_removed',
      F.when(
        F.col('exploded').isNotNull(),
        F.array_except(
          F.col('exploded'), F.array(F.lit(None))
        )
      )
      .otherwise(None)
    )
    
    # if any of the arrays of first and middle names are now empty, transform then to nulls.
    .withColumn(
      'empty_arrays_removed',
      F.when(
        F.size('null_array_elements_removed') > 0,
        F.col('null_array_elements_removed')
      )
      .otherwise(None)
    )
    
    # on the arrays of first and middle names, standardise case and concatenate with whitespace.
    .withColumn(
      'given_name',
      F.when(
        (F.col('empty_arrays_removed').isNotNull()) & (F.lit(case == 'lower')),
        F.lower(F.concat_ws(' ', F.col('empty_arrays_removed')))
      )
      .when(
        (F.col('empty_arrays_removed').isNotNull()) & (F.lit(case == 'upper')),
        F.upper(F.concat_ws(' ', F.col('empty_arrays_removed')))
      )
      .otherwise(None)
    )
    
    # after the previous transformations, some rows may now be identical, so drop duplicates.
    .dropDuplicates(['nhs_number', 'given_name'])
    
    # remove null rows if there is at least one other non-null row for the same patient.
    # we do want to keep at least one row for each patient, even if there were no non-null given names.
    .withColumn(
      'count',
      F.count('nhs_number')
      .over(Window.partitionBy('nhs_number'))
    )   
    .filter(~((F.col('given_name').isNull()) & (F.col('count') > 1)))
    .select('nhs_number', 'given_name')
  )  
  return df_exploded

# COMMAND ----------

def explode_pds(
  df_pds_full: DataFrame, 
  df_pds_replaced_by: DataFrame = [],
  name_case: str = 'lower', 
  postcode_case: str = 'upper',
  remove_confidential_records: bool = True,
  remove_superseded_records: bool = True,
) -> DataFrame:
  ''' Explodes the historical values of PDS.
  So that there is a row for each combination of historical values from demographic fields.
  The intended use of this function is in the prediction step of probabilistic linkage. We want to consider each combination of historic values in PDS as a potential match for the input record.
  This function handles 5 demographic columns from PDS.
    - NHS number: the unique key for each patient. In the exploded dataframe there will be many rows for each NHS number.
    - Date of birth: PDS does have a date of birth history column, however, we assume this is for corrections rather than changes, since it is not possible to change date of birth.
    - Gender: PDS does have a gender history column but we assume it is for corrections. In the case of genuine changes to gender, we understand a new NHS number is issued.
    - Given name: PDS has a given name history column ("name_history.givenNames") which we explode.
    - Family name: PDS has a family name history column ("name_history.familyName") which we explode.
    - Postcode: PDS has a postcode history column ("address_history.postalCode") which we explode.
    
  Args:
    df_pds_full (DataFrame): PDS full dataframe, to be exploded. Must contain at least the nhs_number, dob, gender_code, name_history, and address_history columns.
    df_pds_replaced_by (DataFrame, optional): Dataframe which maps NHS numbers that have been superseded to their superseding value. Only required if remove_superseded_records is True.
    name_case (str): Case (upper or lower) to standardise given name and family name to. Defaults to 'lower'.
    postcode_case (str): Case (upper or lower) to standardise postcode to. Defaults to 'upper'.
    remove_confidential_records (bool): flag to indicate whether confidential records should be removed.
    remove_superseded_records (bool): flag to indicate whether superseded records should be removed.
    
  Returns:
    df_pds_exploded (DataFrame): PDS dataframe exploded, i.e. with a row for every combination of each historical value.
  
  '''

  if remove_confidential_records:
    df_pds_full = remove_confidential_from_pds(df_pds_full)
    
  if remove_superseded_records:
    df_pds_full = remove_superseded_from_pds(df_pds_full, df_pds_replaced_by)

  # We do not explode DOB or gender.
  # They do have historical columns in PDS, but we assume they are corrections rather than changes.
  df_pds_dob_and_gender = (
    df_pds_full
    .select('nhs_number', 'dob', 'gender_code')
  )
  
  # explode given name history
  df_pds_historical_given_names = (
    explode_historical_given_names(
      df_pds_full
      .select('nhs_number', F.col('name_history.givenNames').alias('given_name_history')),
      case = name_case
    )  
  )
  
  # explode family name history
  df_pds_historical_family_names = (
    explode_historical_values(
      df_pds_full
      .select('nhs_number', F.col('name_history.familyName').alias('family_name_history')),
      column_name = 'family_name',
      case = name_case
    )
  )  
  
  # explode postcode history
  df_pds_historical_postcodes = (
    explode_historical_values(
      df_pds_full
      .select('nhs_number', F.col('address_history.postalCode').alias('postcode_history')),
      column_name = 'postcode', 
      case = postcode_case
    )
  )
  
  # use outer joins to create a row for every combination of each historical value.
  df_pds_exploded = (
    df_pds_dob_and_gender
    .join(df_pds_historical_given_names, ['nhs_number'], 'outer')
    .join(df_pds_historical_family_names, ['nhs_number'], 'outer')
    .join(df_pds_historical_postcodes, ['nhs_number'], 'outer')

    # transform to the column names we use in the linkage model.
    .select(
      F.col('nhs_number').alias('NHS_NO'),
      F.col('given_name').alias('GIVEN_NAME'),
      F.col('family_name').alias('FAMILY_NAME'),
      F.col('gender_code').alias('GENDER'),
      F.col('dob').alias('DATE_OF_BIRTH'),
      F.col('postcode').alias('POSTCODE'),
      F.col('nhs_number').alias('LABEL')

    )
    .withColumn('UNIQUE_REFERENCE', F.monotonically_increasing_id())
  )

  return df_pds_exploded

# COMMAND ----------

def load_pds_full_or_exploded(
  pds_database, 
  pds_table, 
  exploded:bool=False
) -> DataFrame:
  """
  Load pds full or exploded.
  
  Args:
    - pds_database, pds_table (str): location of the PDS table to be loaded
    - exploded (bool): whether to return the exploded PDS (with multiple records for each NHS number showing historic values) or just full PDS
    
  Returns:
    - df_pds (DataFrame): PDS in a dataframe and preprocessed
  """
  
  df_pds_full = (spark.table(f'{pds_database}.{pds_table}'))
  df_pds_replaced_by = spark.table('pds.replaced_by')
  
  if exploded:
    df_pds= explode_pds(df_pds_full, df_pds_replaced_by, name_case='lower', postcode_case='upper', remove_confidential_records=True, remove_superseded_records=True)
  else:
    df_pds= (preprocess_full_pds(df_pds_full, df_pds_replaced_by)
              .select('UNIQUE_REFERENCE', 'GENDER', 'GIVEN_NAME', 'NHS_NO', 'FAMILY_NAME', 'DATE_OF_BIRTH', 'POSTCODE', 'LABEL')
              )
    
  return df_pds

# COMMAND ----------

def find_connected_ids(id_dict: Dict[str, str]) -> Dict[str, List[str]]:
    """
    This takes in a dictionary of id: id, and works out for every id across both keys and values,
    what other ids they are linked to.
    
    This is used when we know what the links are between two input values.
    This can then derive if their are any chained links, and return a list of all links for each id. 

    Args:
      id_dict (Dict[str, str]): Dictionary of keys: values which are known to be linked values.

    Returns:
      connected_components (Dict[str, List[str]]): Dictionary of every unique key and value in id_dict, to their linked value in id_dict.

    """
    # Build an adjacency list (graph representation)
    graph = defaultdict(list)
    
    # Add both directions because we want ids from both sides
    for id1, id2 in id_dict.items():
        graph[id1].append(id2)
        graph[id2].append(id1)

    # Perform BFS/DFS to find all connected components
    def bfs(start_node, visited):
        queue = deque([start_node])
        connected_component = set([start_node])
        visited.add(start_node)
        
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    connected_component.add(neighbor)
                    queue.append(neighbor)
                    
        return connected_component

    visited = set()
    connected_components = {}

    # Traverse each node in the graph
    for node in graph:
        if node not in visited:
            # Find the connected component for this node
            component = bfs(node, visited)
            # Assign the component to all nodes in the component
            for id_ in component:
                connected_components[id_] = list(component)
                
    connected_ids = [(key, value) for key, value in connected_components.items()]
    
    return connected_ids

# COMMAND ----------

def load_wider_test_data():
  ''' Load a specific dataset from MPS archive to train and test the linkage model on.
  Returns:
    - df_mps_responses (DataFrame): mps records that can be used for training and evaluation (still needs to be preprocessed to make suitable for linkage)
  '''
  df_request_response = spark.table('mps_archive.request_response')
    
  list_of_submission_ids = [9426034, 11769495, 11769492, 13065891, 13064302, 12297410, 10474124]

  df_request_response = (
    df_request_response
    .filter(F.col('dataset_id') == 'digitrials_dcm')
    .withColumn('submission_id', F.split(F.col('local_id'), '_')[0])
    .filter(F.col('submission_id').isin(list_of_submission_ids))
    .drop('submission_id')
  )

  # Remove duplicates from request_response. Where the ambiguous join has occured, keep the most recent based on req_CREATED.
  id_window = Window.partitionBy(['dataset_id', 'unique_reference'])
  id_window_ordered = Window.partitionBy(['dataset_id', 'unique_reference']).orderBy('req_CREATED')

  df_request_response = (
    df_request_response
    .dropDuplicates()
    .withColumn('row', F.row_number().over(id_window_ordered))
    .withColumn('max_row', F.max('row').over(id_window))
    .filter(F.col('max_row') == F.col('row'))
    .drop('row', 'max_row')
  )

  # ~10,000 records have NHS number. We don't know what these are so remove them.
  df_request_response = df_request_response.filter(F.col('req_NHS_NO').isNull())

 
  df_mps_responses = (
    df_request_response
    .select('local_id', 'dataset_id', 
            F.col('unique_reference').alias('UNIQUE_REFERENCE'),
            F.col('res_created').alias('created'), 
            F.col('res_REQ_NHS_NUMBER').alias('REQ_NHS_NUMBER'),
            F.col('res_MATCHED_NHS_NO').alias('MATCHED_NHS_NO'),
            F.col('res_FAMILY_NAME').alias('FAMILY_NAME'),
            F.col('res_GIVEN_NAME').alias('GIVEN_NAME'),
            F.col('res_GENDER').alias('GENDER'),
            F.col('res_DATE_OF_BIRTH').alias('DATE_OF_BIRTH'),
            F.col('res_POSTCODE').alias('POSTCODE'),
            F.col('res_MATCHED_ALGORITHM_INDICATOR').alias('MATCHED_ALGORITHM_INDICATOR'),
            F.col('res_MATCHED_CONFIDENCE_PERCENTAGE').alias('MATCHED_CONFIDENCE_PERCENTAGE'),
            F.col('res_ERROR_SUCCESS_CODE').alias('ERROR_SUCCESS_CODE'),
            F.col('res_ALGORITHMIC_TRACE_FAMILY_NAME_SCORE_PERC').alias('ALGORITHMIC_TRACE_FAMILY_NAME_SCORE_PERC'),
            F.col('res_ALGORITHMIC_TRACE_GIVEN_NAME_SCORE_PERC').alias('ALGORITHMIC_TRACE_GIVEN_NAME_SCORE_PERC'),
            F.col('res_ALGORITHMIC_TRACE_DOB_SCORE_PERC').alias('ALGORITHMIC_TRACE_DOB_SCORE_PERC'),
            F.col('res_ALGORITHMIC_TRACE_GENDER_SCORE_PERC').alias('ALGORITHMIC_TRACE_GENDER_SCORE_PERC'),
            F.col('res_ALGORITHMIC_TRACE_POSTCODE_SCORE_PERC').alias('ALGORITHMIC_TRACE_POSTCODE_SCORE_PERC'),
          )
  )
  return df_mps_responses