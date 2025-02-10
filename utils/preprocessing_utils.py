# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from pyspark.sql import Row
from typing import *
import pyspark.sql.types as T
import re
# import numpy as np
import abydos.phonetic as abyphon
# import abydos.distance as abydist

import os

# COMMAND ----------

def extract_outcode_from_postcode(postcode:str) -> str:
  ''' For example, extract LS1 from LS1 4DP
  '''

  if postcode==None:
    return None
  
  pattern = r'^([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d[A-Z]{2}$'

  match = re.match(pattern, postcode)
  
  if match:
    return match.group(1)
  else:
    return None

# COMMAND ----------

def extract_sector_from_postcode(postcode:str) -> str:
  ''' For example, extract LS1 4 from LS1 4DP
  '''
  
  if postcode==None:
    return None
  
  pattern = r"^([A-Za-z]{1,2}[0-9][A-Za-z0-9]?[0-9])"
  
  match = re.match(pattern, postcode)
  
  if match:
    return match.group(1)
  else:
    return None

# COMMAND ----------

def extract_region_from_postcode(postcode:str) -> str:
  ''' For example, extract LS from LS1 4DP
  '''
  
  if postcode==None:
    return None

  pattern = r'^([A-Za-z]{1,2})'

  match = re.match(pattern, postcode)
  
  if match:
    return match.group(1)
  else:
    return None

# COMMAND ----------

def preprocess_postcode(df_input:DataFrame, postcode_column:str, postcodes_to_ignore:List[str]=['']) -> DataFrame:
  '''
  This function removes whitespace from postcodes and transforms them to uppercase. It also replaces all 0s with Os to account for errors in writing postcodes when mistaking Os for 0s and vice versa. 
  
  There is an optional parameter to pass in an ignore list which transforms postcodes to ignore to null.
  
  Each postcode to ignore is expected to be in the format 'AA0AA0' i.e. uppercase with no whitespace.
  
  Example: 
  input: aA0 AA0 --> AAOAAO , aa0 aa0 --> AAOAAO
  if CC1CC2 is in the ignore list: cc1 cc2 --> None
  
  '''
  extract_outcode = F.udf(extract_outcode_from_postcode, T.StringType())
  extract_sector = F.udf(extract_sector_from_postcode, T.StringType())
  extract_region = F.udf(extract_region_from_postcode, T.StringType())
  
  df_input = df_input.withColumn(postcode_column, F.upper(F.regexp_replace(postcode_column,'\s','')))
  df_input = (
    df_input.withColumn(
      postcode_column,
      F.when(
        F.col(postcode_column).isin(postcodes_to_ignore),
        None
      )
      .otherwise(F.upper(F.col(postcode_column)))
    ))
  
  df_input = (df_input.withColumn('OUTCODE', extract_outcode(F.col(postcode_column)))
                      .withColumn('SECTOR', extract_sector(F.col(postcode_column)))
                      .withColumn('REGION', extract_region(F.col(postcode_column)))
             )
  
  df_output = (
    df_input
    .withColumn(
      postcode_column,
      F.upper(F.regexp_replace(F.regexp_replace(postcode_column,'0','O'), '\s', ''))
    )
    .withColumn(
      'OUTCODE',
      F.upper(F.regexp_replace(F.regexp_replace('OUTCODE','0','O'), '\s', ''))
    )
    .withColumn(
      'SECTOR',
      F.upper(F.regexp_replace(F.regexp_replace('SECTOR','0','O'), '\s', ''))
    )
    .withColumn(
      'REGION',
      F.upper(F.regexp_replace(F.regexp_replace('REGION','0','O'), '\s', ''))
    )
  )
  
  
  return df_output

# COMMAND ----------

def preprocess_name(
  df_input:DataFrame, 
  name_column:str, 
  split_names: bool = False, 
  names_to_ignore:List[str]=[], 
  apply_soundex: bool = False, 
  create_nicknames_col: bool = False, 
  nicknames_df:DataFrame = None
) -> DataFrame:
  '''
    Parameters
    ----------
    df_input : dataframe
      The dataframe to which the function is applied.
    name_column : str 
      The naming of the either family or given name column that requires preprocessing in the inputted dataframe.
    split_names (optional): boolean
      When true will split given names by dashes and spaces into three columns.
    names_to_ignore (optional): list of strings
      when list is given transforms given names to ignore to null. Each given name to ignore is expected to be a regular expression. Can ignore substrings of potential given names e.g. if you want to remove "miss" from when a name is entered as "Miss Alice" but dont want to lose the name Alice. If you want the ignore list items to only match if they are their own words (i.e. match 'dr' in 'Dr. Ben' but not in 'drake') and not contained in other words, add them to the list with a bracket around them. 
    apply_soundex (optional): boolean
        True if you would like to create a column of the soundex of name
    create_nicknames_col (optional): boolean
      True if you want to create a column that has all the derived nicknames for the given name(s)

    Returns
    -------
    dataframe
       This function removes any extra whitespace, non-alphabet characters, nulls or nones from given names. Also sets zero length strings in given name column to None. 

  '''
    
  special_characters_regex = '[^\p{L}. -]' # removes all non alphabet characters except for '.', '-' and spaces, which will be dealt with later 
  
  df_input = df_input.withColumn(name_column, F.regexp_replace(name_column, special_characters_regex, ''))
  df_input = df_input.withColumn(name_column, F.lower(F.col(name_column)))
  df_input = clean_name(df_input, name_column, names_to_ignore)
  
  #   removes periods and hyphens from the data to ensure consistency (e.g. ian b. smith -> ian b smith)
  df_input = df_input.withColumn(name_column, F.regexp_replace(name_column, r'([.*])|([-*])', ' '))

  df_input = standardise_white_space(df_input, name_column, 'one')
  
    
  df_input = df_input.withColumn('split_names', F.split(F.col(name_column), ' '))

#   remove duplicates in names 
  df_input = df_input.withColumn('split_names', F.array_distinct(F.col('split_names')))


#   creates a column for each of the first three names 
  df_input = (df_input
       .withColumn(f'{name_column}1', F.col('split_names').getItem(0))
       .withColumn(f'{name_column}2', F.col('split_names').getItem(1))
       .withColumn(f'{name_column}3', F.col('split_names').getItem(2))
      )

  df_input = (df_input
               .withColumn(f'{name_column}1',F.when(F.length(f'{name_column}1')==0, None).otherwise(F.col(f'{name_column}1')))
               .withColumn(f'{name_column}2',F.when(F.length(f'{name_column}2')==0, None).otherwise(F.col(f'{name_column}2')))
               .withColumn(f'{name_column}3',F.when(F.length(f'{name_column}3')==0, None).otherwise(F.col(f'{name_column}3')))
              )
  
  array_col_name = 'split_names'
  
  df_input = df_input.withColumn(name_column, 
                                  F.when(F.length(name_column)==0, None)
                                   .otherwise(F.col(name_column))
                                 )
    
  df_input = df_input.withColumn(array_col_name, F.array_remove(F.col(array_col_name), ""))
  
  df_input = df_input.withColumn(array_col_name, 
                                 F.when(F.col(array_col_name).isNull(), None)
                                 .when(F.size(F.col(array_col_name))==0, None)
                                 .otherwise(F.col(array_col_name)))
  
  df_output = df_input.withColumn(name_column, 
                                  F.when(F.col(name_column).isNull(), None)
                                   .otherwise(F.concat_ws(' ', F.col(array_col_name)))
                                 )
  
  if apply_soundex:
    df_output = df_output.withColumn(f'SOUNDEX_{name_column}', F.soundex(F.col(name_column)))
  
  if create_nicknames_col:
    df_output = create_nicknames_columns(df_output, nicknames_df, name_column)
    
  if split_names:
    df_output = df_output.withColumnRenamed('split_names', f'{name_column}_array'.lower())
    df_output = df_output.drop(name_column)
  else:
    df_output = df_output.withColumnRenamed('split_names', f'{name_column}_array'.lower())
    df_output = df_output.drop(f'{name_column}3', f'{name_column}2', f'{name_column}1')
         
  return df_output

# COMMAND ----------

def preprocess_full_name(
  df_input: DataFrame, 
  given_name_col:str, 
  family_name_col:str, 
  double_metaphone:bool = False, 
  alphabetised_full_name:bool = False
) -> DataFrame:
  
  ''' This function generates the derived names required for calculating some of the distance metrics:
    - DM_GIVEN_NAME: double metaphone of given name
    - DM_FAMILY_NAMAE: double metaphone of family name
    - ALPHABETISED_FULL_NAME: the tokens of the full name in alphabetical order
  '''
  
  create_full_name_col_udf = F.udf(create_full_name, T.StringType())
  
  df_output = df_input.withColumn('FULL_NAME',
                                 create_full_name_col_udf(F.col(given_name_col), F.col(family_name_col))
                                 )
  
  if double_metaphone:
    create_DM_udf = F.udf(double_metaphone_as_list, T.ArrayType(T.StringType()))
    df_output = (df_output.withColumn('DM_GIVEN_NAME', create_DM_udf(F.col(given_name_col)))
                      .withColumn('DM_FAMILY_NAME', create_DM_udf(F.col(family_name_col)))
                )
  
  if alphabetised_full_name:
    create_alphabetised_full_name_col_udf = F.udf(alphabetise_full_name, T.StringType())
    df_output = df_output.withColumn('ALPHABETISED_FULL_NAME', 
                                     create_alphabetised_full_name_col_udf(F.col('FULL_NAME'))
                                  )  
  
  return df_output

# COMMAND ----------

def clean_name(df, subset, components = []):
    '''
    from ONS repo
    
    Removes common forename prefixes from a specified column.

    Parameters
    ----------
    df : dataframe
      The dataframe to which the function is applied.
    subset : str or list of str
      The columns on which this function will be applied.
      
    components (optional): list of strings
      when list is given replaces those components in the given column to an empty string.

    Returns
    -------
    dataframe
      Dataframe with standardised forename variables.

    Raises
    -------
    None at present.

    '''

    forename_regex = "|".join([f"\\b{component}\\b"
                               for component in components])

    if not isinstance(subset, list):
        subset = [subset]

    for col in subset:

        df = df.withColumn(col, F.regexp_replace(F.col(col),
                                                 forename_regex,
                                                 ''))

    return df

# COMMAND ----------

def standardise_white_space(df, subset=None, wsl='one', fill=None):
    """
    from ONS github. 
    
    Alters the number of white spaces in the dataframe.

    This can be used to select specified
    columns within the dataframe and alter the number of
    white space characters within the values of those
    specified columns.


    Parameters
    ----------
    df : dataframe
      The dataframe to which the function is applied.
    subset : str or list of str, (default = None)
      The subset of columns that are having their
      white space characters changed. If this is left blank
      then it defaults to all columns.
    wsl : {'one','none'}, (default = 'one')
      wsl stands for white space level, which is used to
      indicate what the user would like white space to be
      replaced with. 'one' replaces multiple whitespces with single
      whitespace. 'none' removes all whitespace
    fill : str, (default = None)
      fill allows user to specify a string that
      would replace whitespace characters.

    Returns
    -------
    dataframe
      The function returns the new dataframe once
      the specified rules are applied.

    Raises
    -------
    None at present.
    """

    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    for col in subset:
        if fill is not None:
            if df.select(col).dtypes[0][1] == 'string':
                df = df.withColumn(col, F.trim(F.col(col)))
                df = df.withColumn(col, F.regexp_replace(
                    F.col(col), "\\s+", fill))

        elif wsl == 'none':
            if df.select(col).dtypes[0][1] == 'string':
                df = df.withColumn(col, F.trim(F.col(col)))
                df = df.withColumn(
                    col, F.regexp_replace(F.col(col), "\\s+", ""))
        elif wsl == 'one':
            if df.select(col).dtypes[0][1] == 'string':
                df = df.withColumn(col, F.trim(F.col(col)))
                df = df.withColumn(
                    col, F.regexp_replace(F.col(col), "\\s+", " "))

    return df

# COMMAND ----------

def preprocess_dob(df_input:DataFrame, dob_column:str, subset_types:list = []) -> DataFrame:
  '''
  This function should contain all DOB preprocessing - which currently entails creating helper columns (DDYYYY column for blocking rules). The input dob column must be in 'YYYYMMDD' format.
  
  '''
  df_dob = df_input
  year = F.substring(F.col(dob_column), 1, 4)
  month = F.substring(F.col(dob_column), 5, 2)
  day = F.substring(F.col(dob_column), 7, 2)
  
  if 'YYYYDD' in subset_types:
    df_dob = df_dob.withColumn('YYYYDD', F.concat(year, day))
  if 'YYYYMM' in subset_types:
    df_dob = df_dob.withColumn('YYYYMM', F.concat(year, month))
  if 'MMDD' in subset_types:
    df_dob = df_dob.withColumn('MMDD', F.concat(month, day))
  if 'YYYYDDMM' in subset_types:
    df_dob = df_dob.withColumn('YYYYDDMM', F.concat(year, day, month))
  if 'MMDDYYYY' in subset_types:
    df_dob = df_dob.withColumn('MMDDYYYY', F.concat(month, day, year))
    
  return df_dob

# COMMAND ----------

def create_full_name(name:str, surname:str) -> str:
  ''' concatenate name with surname to create full_name
  '''
  
  if name and surname:
    tokens = name.split() + surname.split()
  elif name:
    tokens = name.split()
  elif surname:
    tokens = surname.split()
  else:
    return None
     
  return ' '.join(tokens)


def alphabetise_full_name(full_name:str) -> str:
  ''' Arrange the tokens of a string in alphabetical order
  '''

  tokens = full_name.split()
  tokens.sort()
    
  return ' '.join(tokens)

# COMMAND ----------

def double_metaphone_as_list(name: str) -> List:
  ''' Wrapper for the abydos phonetic double metaphone function, so that the two possible metaphones are returned as a list.
  '''
  if name:
      return list(filter(None, abyphon.DoubleMetaphone().encode(name)))
  else:
      return []


# COMMAND ----------

# def damerau_levenshtein_as_int(left: str, right: str) -> int:
#   dl_dist = abydist.DamerauLevenshtein().dist_abs(left, right)
#   if isinstance(dl_dist, np.generic):
#     return dl_dist.item()
#   else:
#     return dl_dist

# COMMAND ----------

if os.getcwd().startswith("/home/spark"):
  def preprocess_all_demographics(df: DataFrame,
                                  preprocess_postcode_args,
                                  preprocess_givenname_args,
                                  preprocess_dob_args,
                                  preprocess_familyname_args,
                                  preprocess_fullname_args
                                  ) -> DataFrame:
    """
    Cleaning the columns we need to train a splink model.  
    """
    df = preprocess_postcode(df, **preprocess_postcode_args)

    df = preprocess_name(df, **preprocess_givenname_args)

    df = preprocess_dob(df , **preprocess_dob_args)

    df = preprocess_name(df , **preprocess_familyname_args)
    
    df = preprocess_full_name(df, **preprocess_fullname_args)
  
    return df

# COMMAND ----------

def add_to_nicknames_dict(name:str, nickname:str):
  '''
  function to add to the nicknames table. Checks if the name exists and makes sure you're not adding duplicate nicknames
  
  name: str: name that you want to add a nickname to 
  nickname: str: nickname you want to add
  
  '''
  all_nicknames = spark.table('mps_enhancement_collab.nicknames_data')

  chosen_name_nicknames = all_nicknames.where(F.col('Full_Name')==name).first()

  if chosen_name_nicknames is not None:
    current_nicknames = chosen_name_nicknames['Nicknames']
    current_nicknames.append(nickname)
    current_nicknames = list(set(current_nicknames))
  else:
    current_nicknames = [nickname]

  rest_of_names = all_nicknames.where(F.col('Full_Name')!=name) 

  schema = T.StructType([
      T.StructField("Full_Name", T.StringType(), False),
      T.StructField("Nicknames", T.ArrayType(T.StringType()), False)
  ])

  new_row = spark.createDataFrame([Row(Full_Name=name, Nicknames=current_nicknames)], schema = schema)

  new_nicknames_df = rest_of_names.union(new_row)

  new_nicknames_df.write.mode('overwrite').saveAsTable('mps_enhancement_collab.nicknames_data')

  return new_row

# COMMAND ----------

def create_nicknames_columns(df_input:DataFrame, df_nicknames:DataFrame, given_name_col:str):
  ''' Adds a column which is an array of all possible nicknames for up to 3 of the given names.
  For example, Robert William -> [Rob, Robbie, Bob, Bertie, Bill, Billy, Bill]  
  '''
  
  df_input_with_nicknames = df_input.withColumn('split_names', F.split(F.col(given_name_col), ' '))
  
  df_input_with_nicknames = (df_input_with_nicknames
         .withColumn(f'{given_name_col}1', F.col('split_names').getItem(0))
         .withColumn(f'{given_name_col}2', F.col('split_names').getItem(1))
         .withColumn(f'{given_name_col}3', F.col('split_names').getItem(2))
        )
  
  df_input_with_nicknames = df_input_with_nicknames.join(df_nicknames.alias('1'), df_input_with_nicknames[f'{given_name_col}1']==df_nicknames.Full_Name, 'left')
  df_input_with_nicknames = df_input_with_nicknames.drop('Full_Name')
  
  df_input_with_nicknames = df_input_with_nicknames.join(df_nicknames.alias('2'), df_input_with_nicknames[f'{given_name_col}2']==df_nicknames.Full_Name, 'left')
  df_input_with_nicknames = df_input_with_nicknames.drop(F.col('Full_Name'))
  
  df_input_with_nicknames = df_input_with_nicknames.join(df_nicknames.alias('3'), df_input_with_nicknames[f'{given_name_col}3']==df_nicknames.Full_Name, 'left')
  df_input_with_nicknames = df_input_with_nicknames.drop(F.col('Full_Name'))
  
  df_input_with_nicknames = df_input_with_nicknames.withColumn("all_nicknames", 
                                                               F.concat(
                                                                 F.coalesce(F.col("1.Nicknames"), F.array()),  
                                                                 F.coalesce(F.col("2.Nicknames"), F.array()), 
                                                                 F.coalesce(F.col("3.Nicknames"), F.array())
                                                               )
                                                              )
  
  df_input_with_nicknames = df_input_with_nicknames.drop("Nicknames",f'{given_name_col}1',f'{given_name_col}2',f'{given_name_col}3')
  
  return df_input_with_nicknames
