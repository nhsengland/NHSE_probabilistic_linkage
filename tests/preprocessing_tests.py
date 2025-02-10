# Databricks notebook source
import os
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

if not os.getcwd().startswith("/home/spark"):
    import sys

    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)
    parent_dir_path = os.path.dirname(current_dir_path)
    sys.path.append(parent_dir_path)

    os.environ["PYSPARK_PYTHON"] = "python"

    from pyspark.sql import SparkSession
    from function_test_suite import *
    from utils.preprocessing_utils import *

    spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %run ./function_test_suite

# COMMAND ----------

# MAGIC %run ../parameters_linking

# COMMAND ----------

# MAGIC %run ../utils/preprocessing_utils

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # preprocess_postcode

# COMMAND ----------

suite_preprocess_postcode = FunctionTestSuite()

@suite_preprocess_postcode.add_test
def ignore_list():
  ''' Test that postcodes to ignore are ignored
  '''
  
  postcodes_to_ignore = ['ZZ9ZZ9']
  
  df_input = spark.createDataFrame(
    [
      ('10', None),
      ('11', 'PE13DS'),
      ('12', 'gu46 6hp'),
      ('13', 'EC2V7QR'),
      ('14', 'ZZ9 ZZ9'),
    ],
    ['test_id', 'postcode']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', None, None, None, None),
      ('11', 'PE13DS', 'PE1','PE13','PE'),
      ('12', 'GU466HP', 'GU46','GU466','GU'),
      ('13', 'EC2V7QR', 'EC2V','EC2V7','EC'),
      ('14', None, None, None, None),
    ],
    ['test_id', 'postcode', 'OUTCODE','SECTOR','REGION']
  )
  
  df_output = preprocess_postcode(df_input,'postcode', postcodes_to_ignore)

  assert compare_results(df_output, df_expected, ['test_id'])
  
  
@suite_preprocess_postcode.add_test
def no_ignore_list():
  ''' Test that postcodes are preprocessed correctly
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', None),
      ('11', 'AA1 AA1'),
      ('12', 'EC4N 8AR'),
      ('13', 'RG72NA'),
      ('14', 'TN36 4LQ'),
    ],
    ['test_id', 'postcode']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', None, None, None, None),
      ('11', 'AA1AA1', None, None,'AA'),
      ('12', 'EC4N8AR', 'EC4N', 'EC4N8', 'EC'),
      ('13', 'RG72NA', 'RG7','RG72', 'RG'),
      ('14', 'TN364LQ', 'TN36','TN364','TN'),
    ],
    ['test_id', 'postcode', 'OUTCODE','SECTOR','REGION']
  )
  
  df_output = preprocess_postcode(df_input,'postcode')
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
  
@suite_preprocess_postcode.add_test
def zeros_and_os():
  ''' test that zeros are coverted to "o"
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', None),
      ('11', 'EC4N8AR'),
      ('12', 'O01A AA1'),
      ('13', '000000'),
      ('14', 'DT101QU'),
    ],
    ['test_id', 'postcode']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', None, None, None, None),
      ('11', 'EC4N8AR', 'EC4N','EC4N8','EC'),
      ('12', 'OO1AAA1', None, 'OO1', 'O'),
      ('13', 'OOOOOO',None, None, None),
      ('14', 'DT1O1QU','DT1O','DT1O1','DT'),
    ],
    ['test_id', 'postcode', 'OUTCODE','SECTOR','REGION']
  )
  
  df_output = preprocess_postcode(df_input,'postcode')
  assert compare_results(df_output, df_expected, ['test_id'])
  
suite_preprocess_postcode.run()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # preprocess_name

# COMMAND ----------

suite_preprocess_name = FunctionTestSuite()

@suite_preprocess_name.add_test
def no_ignore_list_and_no_splitting():
  ''' Test that given names are preprocessed correctly
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', 'given_name'),
      ('11', None, ),
      ('12', ''), 
      ('16', 'anne--marie--claire'),
      ('17', 'anne...marie..claire'),
      ('18', 'anne   marie-claire'),
      ('19','123anne'),
      ('20', ' '),
      ('21', '... '), 
    ],
    ['test_id', 'given_name']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', 'givenname'),
      ('11', None),
      ('12', None),
      ('16','anne marie claire'),
      ('17','anne marie claire'),
      ('18','anne marie claire'),
      ('19','anne'),
      ('20', None),
      ('21', None)
    ],
    ['test_id', 'given_name']
  )
 
  
  df_output = preprocess_name(df_input,'given_name').drop('given_name_array')
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
  
@suite_preprocess_name.add_test
def ignore_list():
  ''' Test that given names to ignore are ignored
  '''
  
  given_names_to_ignore = [
    '^\W+$',
    '\d',
    'baby'
  ]
  
  df_input = spark.createDataFrame(
    [
      ('10', 'given_name'),
      ('11', None),
      ('12', ''), 
      ('13', '------------'),
      ('14', 'given_name 1'),
      ('15', 'baby of'),
      ('16', 'baby baby')
    ],
    ['test_id', 'given_name']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', 'givenname'),
      ('11', None),
      ('12', None),
      ('13', None),
      ('14', 'givenname'),
      ('15', 'of'),
      ('16', None)
    ],
    ['test_id', 'given_name']
  )
  
  df_output = preprocess_name(df_input,'given_name', False, given_names_to_ignore).drop('given_name_array')
  
  assert compare_results(df_output, df_expected, ['test_id'])
 

@suite_preprocess_name.add_test
def split_names():
  ''' Test that multiple given names are split into separate columns (before they are combined in an array)
  '''

  df_input = spark.createDataFrame(
    [
      ('0','amaia'),
      ('1','anne-marie claire'),
      ('2','anne marie claire'),
      ('3','a. m. claire'),
      ('4','a..m.c.'),
      ('5','anne anne-marie claire'),
      ('6','anne.anne.marie.claire'),
      ('7','miss amaia'),
      ('8','miss. anne-marie'),
      ('9','dr. anne'),
      ('10', 'none'),
      ('11','null none'),
      ('12', 'anne known as anna'),
    ],
    ['test_id', 'given_name']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('0', 'amaia',None,None),
      ('1', 'anne','marie','claire'),
      ('2', 'anne','marie','claire'),
      ('3','a', 'm', 'claire'),
      ('4','a','m','c'),
      ('5','anne', 'marie','claire'),
      ('6','anne','marie','claire'),
      ('7', 'miss','amaia',None),
      ('8', 'miss','anne','marie'),
      ('9', 'dr','anne',None),
      ('10','none',None,None),
      ('11','null','none',None),
      ('12','anne','known','as'),
    ],
    ['test_id', 'given_name1','given_name2','given_name3']
  )
  
  df_output = preprocess_name(df_input,'given_name', True,[]).drop('given_name_array')
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
  
@suite_preprocess_name.add_test
def ignore_list_and_split():
  ''' Test the combination of splitting given names and ignoring names from the ignore list
  '''
  
  df_input = spark.createDataFrame(
    [
      ('0','amaia'),
      ('1','anne-marie claire'),
      ('2','anne marie claire'),
      ('3','a. m. claire'),
      ('4','a.m.c.'),
      ('5','anne anne-marie claire'),
      ('6','anne.anne.marie.claire'),
      ('7','miss amaia'),
      ('8','miss. anne--marie'),
      ('9','dr. anne'),
      ('10', 'mr adam'),
      ('11','annie amissa'),
      ('12', 'anne known as anna'),
      ('13', 'anne marie claire susana'),
      ('14', 'drake'),
      ('15','dr anne'),
      ('16', 'anne--marie--claire'),
      ('17', 'anne...marie..claire'),
      ('18', 'anne   marie-claire')
    ],
    ['test_id', 'given_name']
  )
  
  things_to_remove = ['use to be known as','also known as','likes to be known as','wants to be known as','prefers to be known as','known as', '(^miss)', '(^mrs)', '(^mr)', '(^dr)']
  
  df_expected = spark.createDataFrame(
    [
      ('0','amaia',None,None),
      ('1','anne','marie', 'claire'),
      ('2','anne', 'marie', 'claire'),
      ('3','a', 'm', 'claire'),
      ('4','a','m','c'),
      ('5','anne', 'marie','claire'),
      ('6','anne','marie','claire'),
      ('7','amaia',None,None),
      ('8','anne','marie',None),
      ('9','anne',None,None),
      ('10','adam',None,None),
      ('11','annie','amissa',None),
      ('12','anne','anna',None),
      ('13', 'anne', 'marie', 'claire'),
      ('14', 'drake', None,None),
      ('15','anne',None,None),
      ('16','anne','marie', 'claire'),
      ('17','anne','marie', 'claire'),
      ('18','anne','marie', 'claire')
    ],
    ['test_id', 'given_name1', 'given_name2', 'given_name3']
  )
  
  df_output = preprocess_name(df_input,'given_name',True,things_to_remove).drop('given_name_array')
  assert compare_results(df_output, df_expected, ['test_id'])

  
@suite_preprocess_name.add_test
def given_name_array():
  ''' Test that multiple given names, once split, are then correctly collected in an array
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', 'given_name'),
      ('11', None, ),
      ('12', ''), 
      ('16', 'anne- anne -marie--claire'),
      ('17', 'anne...marie..claire'),
      ('18', 'anne   marie-claire'),
      ('19','123anne')
    ],
    ['test_id', 'given_name']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', 'givenname',['givenname']),
      ('11', None,None),
      ('12', None,None),
      ('16','anne marie claire',['anne','marie','claire']),
      ('17','anne marie claire',['anne','marie','claire']),
      ('18','anne marie claire',['anne','marie','claire']),
      ('19','anne',['anne'])
    ],
    ['test_id', 'given_name','given_name_array']
  )
  
  df_output = preprocess_name(df_input,'given_name')
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
  
@suite_preprocess_name.add_test
def ignore_list_split_and_array():
  ''' Test the combination of splitting names into an array and ignoring names from the ignore list
  '''
  
  df_input = spark.createDataFrame(
    [
      ('0','amaia'),
      ('1','anne-marie claire'),
      ('2','anne marie claire'),
      ('3','a. m. claire'),
      ('4','a.m.c.'),
      ('5','anne anne-marie claire'),
      ('6','anne.anne.marie.claire'),
      ('7','miss amaia'),
      ('8','miss. anne--marie'),
      ('9','dr. anne'),
      ('10', 'mr adam'),
      ('11','annie amissa'),
      ('12', 'anne known as anna'),
      ('13', 'anne marie claire susana'),
      ('14', 'drake'),
      ('15','dr anne'),
      ('16', 'anne--marie--claire'),
      ('17', 'anne...marie..claire'),
      ('18', 'anne   marie-claire')
    ],
    ['test_id', 'given_name']
  )
  
  things_to_remove = ['use to be known as','also known as','likes to be known as','wants to be known as','prefers to be known as','known as', '(^miss)', '(^mrs)', '(^mr)', '(^dr)']
  
  df_expected = spark.createDataFrame(
    [
      ('0','amaia',None,None,['amaia']),
      ('1','anne','marie', 'claire',['anne','marie','claire']),
      ('2','anne', 'marie', 'claire',['anne','marie','claire']),
      ('3','a', 'm', 'claire',['a','m','claire']),
      ('4','a','m','c',['a','m','c']),
      ('5','anne', 'marie','claire',['anne','marie','claire']),
      ('6','anne','marie','claire',['anne','marie','claire']),
      ('7','amaia',None,None,['amaia']),
      ('8','anne','marie',None,['anne','marie']),
      ('9','anne',None,None,['anne']),
      ('10','adam',None,None,['adam']),
      ('11','annie','amissa',None,['annie','amissa']),
      ('12','anne','anna',None,['anne','anna']),
      ('13', 'anne', 'marie', 'claire',['anne','marie','claire','susana']),
      ('14', 'drake', None,None,['drake']),
      ('15','anne',None,None,['anne']),
      ('16','anne','marie', 'claire',['anne','marie','claire']),
      ('17','anne','marie', 'claire',['anne','marie','claire']),
      ('18','anne','marie', 'claire',['anne','marie','claire'])
    ],
    ['test_id', 'given_name1', 'given_name2', 'given_name3','given_name_array']
  )
  
  df_output = preprocess_name(df_input,'given_name',True,things_to_remove).select('test_id', 'given_name1', 'given_name2', 'given_name3','given_name_array')
  
  assert compare_results(df_output, df_expected, ['test_id'])
 

@suite_preprocess_name.add_test
def family_name():
  ''' Test the preprocessing of family name, including splitting names into an array
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', 'family_name'),
      ('11', None, ),
      ('12', ''), 
      ('16', 'smith--davies--stclaire'),
      ('17', 'davies...stclaire..smith'),
      ('18', 'smith    davies'),
      ('19','123smith')
    ],
    ['test_id', 'family_name']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', 'familyname',['familyname']),
      ('11', None, None),
      ('12', None, None),
      ('16','smith davies stclaire', ['smith','davies','stclaire']),
      ('17','davies stclaire smith',['davies','stclaire','smith']),
      ('18','smith davies',['smith','davies']),
      ('19','smith',['smith'])
    ],
    ['test_id', 'family_name','family_name_array']
  )
  
  df_output = preprocess_name(df_input,'family_name')
  assert compare_results(df_output, df_expected, ['test_id'])
  
@suite_preprocess_name.add_test
def family_name_soundex():
  ''' Test adding a soundex version of family name
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', 'family_name'),
      ('11', None, ),
      ('12', ''), 
      ('16', 'smith--davies--stclaire'),
      ('17', 'davies...stclaire..smith'),
      ('18', 'smith    davies'),
      ('19','123smith')
    ],
    ['test_id', 'family_name']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', 'familyname',['familyname'],'F554'),
      ('11', None,[],''),
      ('12', None,[],''),
      ('16','smith davies stclaire',[],''),
      ('17','davies stclaire smith',[],''),
      ('18','smith davies',[],''),
      ('19','smith',[],'')
    ],
    ['test_id', 'family_name','family_name_array','SOUNDEX_family_name']
  )
 
  
  df_output = preprocess_name(df_input,'family_name', apply_soundex = True)
  assert check_schemas_match(df_output, df_expected, ['test_id'])
  
suite_preprocess_name.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # clean_forename

# COMMAND ----------

suite_clean_name = FunctionTestSuite()

@suite_clean_name.add_test
def no_components():
  ''' Test that clean_name with no components parameter makes no difference
  '''
  
  df_input = spark.createDataFrame(
      [
          ("0", "amaia"),
          ("1", "anne-marie claire"),
          ("2", "anne marie claire"),
          ("3", "a. m. claire"),
          ("4", "a.m.c."),
          ("5", "anne anne-marie claire"),
          ("6", "anne.anne.marie.claire"),
          ("7", "miss amaia"),
          ("8", "miss. anne--marie"),
          ("9", "dr. anne"),
          ("10", "mr adam"),
          ("11", "annie amissa"),
          ("12", "anne known as anna"),
          ("13", "anne marie claire susana"),
          ("14", "drake"),
          ("15", "dr anne"),
          ("16", "anne--marie--claire"),
          ("17", "anne...marie..claire"),
          ("18", "anne   marie-claire"),
      ],
      ["test_id", "given_name"],
  )

  df_output = clean_name(df_input, "given_name", [])

  assert compare_results(df_output, df_input, ["test_id"])


@suite_clean_name.add_test
def with_components():
  ''' Test that components_to_remove are removed when running clean_name
  '''
  
  df_input = spark.createDataFrame(
      [
          ("0", "amaia"),
          ("1", "anne-marie claire"),
          ("2", "anne marie claire"),
          ("3", "miss amaia"),
          ("4", "miss anne--marie"),
          ("5", "dr anne"),
          ("6", "mr adam"),
          ("7", "anne known as anna"),
          ("8", "dr anne"),
          ("9", "drake"),
          ("10", "amr"),
          ("11", "amissa"),
      ],
      ["test_id", "given_name"],
  )

  components_to_remove = [
      "known as",
      "(^miss)",
      "(^mr)",
      "(^dr)",
  ]

  df_expected = spark.createDataFrame(
      [
          ("0", "amaia"),
          ("1", "anne-marie claire"),
          ("2", "anne marie claire"),
          ("3", " amaia"),
          ("4", " anne--marie"),
          ("5", " anne"),
          ("6", " adam"),
          ("7", "anne  anna"),
          ("8", " anne"),
          ("9", "drake"),
          ("10", "amr"),
          ("11", "amissa"),
      ],
      ["test_id", "given_name"],
  )

  df_output = clean_name(df_input, "given_name", components_to_remove)

  assert compare_results(df_output, df_expected, ["test_id"])


@suite_clean_name.add_test
def multiple_columns():
  ''' Test that components_to_remove are removed from all names (if multiple names have been split)
  '''
  
  df_input = spark.createDataFrame(
      [
          ("0", "amaia", "anne-marie claire", "anne marie claire"),
          ("1", "miss amaia", "miss anne--marie", "dr anne"),
      ],
      ["test_id", "given_name", "given_name2", "given_name3"],
  )

  components_to_remove = [
      "(^miss)",
      "(^dr)",
  ]

  df_expected = spark.createDataFrame(
      [
          ("0", "amaia", "anne-marie claire", "anne marie claire"),
          ("1", " amaia", " anne--marie", " anne"),
      ],
      ["test_id", "given_name", "given_name2", "given_name3"],
  )

  df_output = clean_name(
      df_input, ["given_name", "given_name2", "given_name3"], components_to_remove
  )

  assert compare_results(df_output, df_expected, ["test_id"])


suite_clean_name.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # standardise_white_space

# COMMAND ----------

suite_standardise_white_space = FunctionTestSuite()

@suite_standardise_white_space.add_test
def default_values():
  ''' Test that default (one) whitespace formatting is applied
  '''
  
  df_input = spark.createDataFrame(
      [
          ("0", "am  aia", "anne   marie claire", "anne marie claire  "),
          ("1", " amaia", " anne--marie", " anne"),
      ],
      ["test_id", "given_name", "given_name2", "given_name3"],
  )

  df_expected = spark.createDataFrame(
      [
          ("0", "am aia", "anne marie claire", "anne marie claire"),
          ("1", "amaia", "anne--marie", "anne"),
      ],
      ["test_id", "given_name", "given_name2", "given_name3"],
  )

  df_output = standardise_white_space(df_input)

  assert compare_results(df_output, df_expected, ["test_id"])


@suite_standardise_white_space.add_test
def single_column():
  ''' test that whitespace removal is applied to only the specified column
  '''
  
  df_input = spark.createDataFrame(
      [
          ("0", "ama  ia", "anne   marie claire"),
          ("1", " amaia", " anne--marie"),
      ],
      ["test_id", "given_name", "given_name2"],
  )

  df_expected = spark.createDataFrame(
      [
          ("0", "ama ia", "anne   marie claire"),
          ("1", "amaia", " anne--marie"),
      ],
      ["test_id", "given_name", "given_name2"],
  )

  df_output = standardise_white_space(df_input, "given_name")

  assert compare_results(df_output, df_expected, ["test_id"])


@suite_standardise_white_space.add_test
def with_wsl_none():
  ''' Test that all whitespace is removed
  '''
  
  df_input = spark.createDataFrame(
      [
          ("0", "anne   marie claire"),
          ("1", " anne--marie"),
      ],
      ["test_id", "given_name"],
  )

  df_expected = spark.createDataFrame(
      [
          ("0", "annemarieclaire"),
          ("1", "anne--marie"),
      ],
      ["test_id", "given_name"],
  )

  df_output = standardise_white_space(df_input, wsl="none")

  assert compare_results(df_output, df_expected, ["test_id"])


@suite_standardise_white_space.add_test
def with_fill():
  ''' Test that whitespace is replaced with a fill string
  '''
  
  df_input = spark.createDataFrame(
      [
          ("0", "anne marie claire"),
          ("1", "anne--marie"),
      ],
      ["test_id", "given_name"],
  )

  df_expected = spark.createDataFrame(
      [
          ("0", "anne*marie*claire"),
          ("1", "anne--marie"),
      ],
      ["test_id", "given_name"],
  )

  df_output = standardise_white_space(df_input, fill="*")

  assert compare_results(df_output, df_expected, ["test_id"])


suite_standardise_white_space.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocess DOB

# COMMAND ----------

suite_preprocess_dob = FunctionTestSuite()

@suite_preprocess_dob.add_test
def day_year():
  ''' Test that day and year are extracted from a date of birth
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', None),
      ('11', '19990212'),
      ('12', '199902'),
      ('13', '20001103'),
    ],
    ['test_id', 'dob']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', None, None),
      ('11', '19990212', '199912'),
      ('12', '199902', '1999'),
      ('13', '20001103', '200003'),
    ],
    ['test_id', 'dob', 'YYYYDD']
  )
  
  df_output = preprocess_dob(df_input,'dob', ['YYYYDD'])

  assert compare_results(df_output, df_expected, ['test_id'])

  
@suite_preprocess_dob.add_test
def month_year():
  ''' Test that month and year are extracted from a date of birth
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', None),
      ('11', '19990212'),
      ('12', '199902'),
      ('13', '20001103'),
    ],
    ['test_id', 'dob']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', None, None),
      ('11', '19990212', '199902'),
      ('12', '199902', '199902'),
      ('13', '20001103', '200011'),
    ],
    ['test_id', 'dob', 'YYYYMM']
  )
  
  df_output = preprocess_dob(df_input,'dob', ['YYYYMM'])

  assert compare_results(df_output, df_expected, ['test_id'])
  
  
@suite_preprocess_dob.add_test
def day_month():
  ''' Test that day and month are extracted from a date of birth
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', None),
      ('11', '19990212'),
      ('12', '199902'),
      ('13', '20001103'),
    ],
    ['test_id', 'dob']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', None, None),
      ('11', '19990212', '0212'),
      ('12', '199902', '02'),
      ('13', '20001103', '1103'),
    ],
    ['test_id', 'dob', 'MMDD']
  )
  
  df_output = preprocess_dob(df_input, 'dob', ['MMDD'])

  assert compare_results(df_output, df_expected, ['test_id'])

  
@suite_preprocess_dob.add_test
def no_change():
  ''' Test preprocessing if no additional derived columns are requested
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', None),
      ('11', '19990212'),
      ('12', '199902'),
      ('13', '20001103'),
    ],
    ['test_id', 'dob']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', None),
      ('11', '19990212'),
      ('12', '199902'),
      ('13', '20001103'),
    ],
    ['test_id', 'dob']
  )
  
  df_output = preprocess_dob(df_input, 'dob')

  assert compare_results(df_output, df_expected, ['test_id'])
           
    
@suite_preprocess_dob.add_test
def several_subcolumns():
  ''' Test date of birth preprocessing if several derived columns are requested
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', None),
      ('11', '19990212'),
      ('12', '199902'),
      ('13', '20001103'),
    ],
    ['test_id', 'dob']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', None, None, None,None,None),
      ('11', '19990212', '199902','0212','19991202','02121999'),
      ('12', '199902', '199902','02','199902','021999'),
      ('13', '20001103', '200011','1103','20000311','11032000'),
    ],
    ['test_id', 'dob', 'YYYYMM', 'MMDD','YYYYDDMM','MMDDYYYY']
  )
  
  df_output = preprocess_dob(df_input, 'dob', ['YYYYMM', 'MMDD', 'YYYYDDMM', 'MMDDYYYY'])
  assert compare_results(df_output, df_expected, ['test_id'])

  
suite_preprocess_dob.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # preprocess_full_name
# MAGIC

# COMMAND ----------

suite_preprocess_full_name = FunctionTestSuite()

@suite_preprocess_full_name.add_test
def check_dm():
  ''' Test that preprocess_full_name adds double metaphone for given name and family name, full name, and alphabetised full name
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', 'givenname','smith'),
      ('11', None, 'smith'),
      ('12', 'john', None), 
      ('16', 'anne marie claire','davies smith'),
      ('17', 'anne marie claire','davies'),
      ('18', 'anne marie claire','john'),
      ('19','anne','smith davies')
    ],
    ['test_id', 'given_name','family_name']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', 'givenname','smith',['JFNM', 'KFNM'],['SM0', 'XMT'],'givenname smith', 'givenname smith'),
      ('11', None, 'smith',[''],[''],'smith', 'smith'),
      ('12', 'john', None,[''],[''],'john', 'john'), 
      ('16', 'anne marie claire','davies smith',[''],[''],'anne claire davies marie smith', 'anne marie claire davies smith'),
      ('17', 'anne marie claire','davies',[''],[''],'anne claire davies marie', 'anne marie claire davies'),
      ('18', 'anne marie claire','john',[''],[''],'anne claire john marie', 'anne marie claire john'),
      ('19','anne','smith davies',[''],[''],'anne davies smith', 'anne smith davies')
    ],
    ['test_id', 'given_name','family_name','DM_GIVEN_NAME','DM_FAMILY_NAME','ALPHABETISED_FULL_NAME', 'FULL_NAME']
  ).select('test_id', 'given_name','family_name', 'FULL_NAME', 'DM_GIVEN_NAME','DM_FAMILY_NAME','ALPHABETISED_FULL_NAME')
  
  df_output = preprocess_full_name(df_input,'given_name','family_name', True, True)
  
  assert check_schemas_match(df_output, df_expected, ['test_id'])
  
  df_expected = df_expected.select('test_id','given_name','family_name','ALPHABETISED_FULL_NAME', 'FULL_NAME')
  df_output = df_output.select('test_id','given_name','family_name','ALPHABETISED_FULL_NAME', 'FULL_NAME')
  
  assert compare_results(df_output, df_expected, ['test_id'])

suite_preprocess_full_name.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # Add_Nicknames_column

# COMMAND ----------

suite_add_nicknames = FunctionTestSuite()
                             
@suite_add_nicknames.add_test
def add_nicknames_col():
  ''' Test that nicknames are added
  '''
  
  df_input = spark.createDataFrame(
    [
      ('0','alexander'),
      ('1', 'alexander claire'),
      ('2','anne marie claire'),
    ],
    ['test_id', 'given_name']
  )
  df_nicknames = spark.createDataFrame(
    [
      ('alexander',['alex', 'alec', 'al']),
      ('anne',['annie']),
      ('marie', ['m', 'mae']),
      ('claire', ['clara'])
    ],
    ['Full_Name', 'Nicknames']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('0','alexander', ['alexander'], ['alex', 'alec', 'al']),
      ('1', 'alexander claire', ['alexander', 'claire'], ['alex','alec', 'al','clara']),
      ('2','anne marie claire',['anne', 'marie','claire'], ['annie', 'm', 'mae', 'clara']),
    ],
    ['test_id', 'given_name', 'split_names', 'all_nicknames']
  )
  
  df_output = create_nicknames_columns(df_input, df_nicknames, 'given_name')
  
  assert compare_results(df_output, df_expected, ['test_id'])

  
suite_add_nicknames.run()

# COMMAND ----------


