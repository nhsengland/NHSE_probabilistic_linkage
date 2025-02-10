# Databricks notebook source
import os

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

    spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %run ./function_test_suite

# COMMAND ----------

# MAGIC %run ../utils/dataset_ingestion_utils

# COMMAND ----------

# MAGIC %md 
# MAGIC # remove_confidential_from_pds

# COMMAND ----------

suite_remove_confidential_from_pds = FunctionTestSuite()

@suite_remove_confidential_from_pds.add_test
def all_scenarios():
  ''' Test if confidentiality codes are handled correctly.
  Records with 'I', 'S', or 'Y' should be removed.
  '''
  
  input_schema = (
    T.StructType([
      T.StructField('test_id', T.StringType()),
      T.StructField('confidentialityCode', T.ArrayType(
        T.StructType([
          T.StructField('code', T.StringType()),
        ])
      ))
    ])
  )
  
  df_input = spark.createDataFrame(
    [
      ('10', [{'code': 'I'}]),
      ('11', [{'code': 'S'}]),
      ('12', [{'code': 'N'}]),
      ('13', [{'code': 'Y'}]),
      ('14', [{'code': None}]),
    ],
    input_schema
  )
  
  df_expected = spark.createDataFrame(
    [
      ('12', [{'code': 'N'}]),
      ('14', [{'code': None}]),
    ],
    input_schema
  )
  
  df_output = remove_confidential_from_pds(df_input)
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
suite_remove_confidential_from_pds.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # update_superseded_nhs_numbers

# COMMAND ----------

suite_update_superseded_nhs_numbers = FunctionTestSuite()

@suite_update_superseded_nhs_numbers.add_test
def all_scenarios():
  ''' Test if superseded nhs numbers are updated correctly.
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', '1111111111', '1111111111'),
      ('11', '1111111111', '2222222222'),
      ('12', '1111111111', '5555555555'),
      ('13', '5555555555', '3333333333'),
      ('14', '5555555555', '4444444444'),
    ],
    ['test_id', 'req_NHS_NO', 'res_NHS_NO']
  )
  
  df_pds_replaced_by = spark.createDataFrame(
    [
      ('1111111111', '2222222222'),
      ('3333333333', '4444444444')
    ],
    ['nhs_number', 'replaced_by']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', '2222222222', '2222222222'),
      ('11', '2222222222', '2222222222'),
      ('12', '2222222222', '5555555555'),
      ('13', '5555555555', '4444444444'),
      ('14', '5555555555', '4444444444'),
    ],
    ['test_id', 'req_NHS_NO', 'res_NHS_NO']  )
  
  df_input = update_superseded_nhs_numbers(df_input, df_pds_replaced_by, 'req_NHS_NO')
  df_output = update_superseded_nhs_numbers(df_input, df_pds_replaced_by, 'res_NHS_NO')
  
  assert compare_results(df_output, df_expected, ['test_id'])
  
suite_update_superseded_nhs_numbers.run()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # explode_historical_given_names

# COMMAND ----------

suite_explode_historical_given_names = FunctionTestSuite()

@suite_explode_historical_given_names.add_test
def all_scenarios():
  ''' test if PDS records with multiple entries in given_name_history are successfully exploded into multiple records.
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', [['preferred_given_name_part_0', 'preferred_given_name_part_1'], ['given_name_historical_part_0', 'given_name_historical_part_1']]),
      ('11', None),
      ('12', []),
      ('13', [[]]),
      ('14', [[None]]),
      ('15',[['preferred_given_name']]),
      ('16', [[], []]),
      ('17', [[], [None]]),
      ('18', [[], ['given_name_historical']]),
      ('19', [[], [None, None]]),
      ('20', [['preferred_given_name_part_0', 'preferred_given_name_part_1'], ['preferred_given_name_part_0', 'preferred_given_name_part_1']]),
      ('21', [['preferred_given_name', 'preferred_given_name'], ['given_name_historical_part_0', 'given_name_historical_part_1']]),
    ],
    ['nhs_number', 'given_name_history']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', 'preferred_given_name_part_0 preferred_given_name_part_1'),
      ('10', 'given_name_historical_part_0 given_name_historical_part_1'),
      ('11', None),
      ('12', None),
      ('13', None),
      ('14', None),
      ('15', 'preferred_given_name'),
      ('16', None),
      ('17', None),
      ('18', 'given_name_historical'),
      ('19', None),
      ('20', 'preferred_given_name_part_0 preferred_given_name_part_1'),
      ('21', 'preferred_given_name'),
      ('21', 'given_name_historical_part_0 given_name_historical_part_1'),
    ],
    ['nhs_number', 'given_name']
  )
  
  df_output = explode_historical_given_names(df_input, 'lower')
  
  assert compare_results(df_output, df_expected, ['nhs_number', 'given_name'])
  
suite_explode_historical_given_names.run()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # explode_historical_values

# COMMAND ----------

suite_explode_historical_values = FunctionTestSuite()

@suite_explode_historical_values.add_test
def all_scenarios():
  ''' Test if PDS records with multiple entries in family_name_history are successfully exploded into multiple records.
  address_history has the same structure as family_name_history, so does not need to be tested separately.
  '''

  df_input = spark.createDataFrame(
    [
      ('10', ['preferred_family_name', 'family_name_historical']),
      ('11', None),
      ('12', []),
      ('13', [None]),
      ('14', [None, None]),
      ('15', ['family_name_historical']),
      ('16', ['family_name_historical', None]),
      ('17', ['family_name_historical', 'family_name_historical']),
      ('18', ['family_name_historical', 'FAMILY_NAME_HISTORICAL']),
      ('19', [None, 'family_name_historical_0', 'family_name_historical_1', 'family_name_historical_1', 'family_name_historical_2', 'FAMILY_NAME_HISTORICAL_2']),
    ],
    ['nhs_number', 'family_name_history']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('10', 'preferred_family_name'),
      ('10', 'family_name_historical'),
      ('11', None),
      ('12', None),
      ('13', None),
      ('14', None),
      ('15', 'family_name_historical'),
      ('16', 'family_name_historical'),
      ('17', 'family_name_historical'),
      ('18', 'family_name_historical'),
      ('19', 'family_name_historical_0'),
      ('19', 'family_name_historical_1'),
      ('19', 'family_name_historical_2'),
    ],
    ['nhs_number', 'family_name']
  )
  
  df_output = explode_historical_values(df_input, 'family_name', 'lower')
  
  assert compare_results(df_output, df_expected, ['nhs_number', 'family_name'])
  
suite_explode_historical_values.run()

# COMMAND ----------

suite_explode_pds = FunctionTestSuite()

@suite_explode_pds.add_test
def no_removing():
  ''' Test the explosion of given_name, family_name, and address, simultaneously.
  '''
  
  input_schema = (
    T.StructType([
      T.StructField('nhs_number', T.StringType()),
      T.StructField('dob', T.IntegerType()),
      T.StructField('gender_code', T.StringType()),
      T.StructField('name_history', T.ArrayType(
        T.StructType([
          T.StructField('givenNames', T.ArrayType(T.StringType())),
          T.StructField('familyName', T.StringType())
        ])
      )),
      T.StructField('address_history', T.ArrayType(
        T.StructType([
          T.StructField('postalCode', T.StringType())
        ])
      ))
    ])
  )
  
  output_schema = (
    T.StructType([
      T.StructField('NHS_NO', T.StringType()),
      T.StructField('GIVEN_NAME', T.StringType()),
      T.StructField('FAMILY_NAME', T.StringType()),
      T.StructField('GENDER', T.StringType()),
      T.StructField('DATE_OF_BIRTH', T.IntegerType()),
      T.StructField('POSTCODE', T.StringType())
    ])
  )
    
  df_input = spark.createDataFrame(
    [('1111111111', 11012001, '1', 
      [{'givenNames': ['John', 'Mark'], 'familyName': 'Smith'}, {'givenNames': ['Henry', 'James'], 'familyName': 'Jones'}],
      [{'postalCode': 'LS1 1AA'},{'postalCode': 'LS2 2BB'}]
    )],
    input_schema
  )
  
  
  df_expected = spark.createDataFrame(
    [
      ('1111111111', 'john mark', 'smith', '1', 11012001, 'LS1 1AA'),
      ('1111111111', 'john mark', 'smith', '1', 11012001, 'LS2 2BB'),
      ('1111111111', 'john mark', 'jones', '1', 11012001, 'LS1 1AA'),
      ('1111111111', 'john mark', 'jones', '1', 11012001, 'LS2 2BB'),
      ('1111111111', 'henry james', 'smith', '1', 11012001, 'LS1 1AA'),
      ('1111111111', 'henry james', 'smith', '1', 11012001, 'LS2 2BB'),
      ('1111111111', 'henry james', 'jones', '1', 11012001, 'LS1 1AA'),
      ('1111111111', 'henry james', 'jones', '1', 11012001, 'LS2 2BB'),
    ],
    output_schema
  )
  
  df_output = explode_pds(df_input, name_case='lower', postcode_case='upper', remove_confidential_records=False, remove_superseded_records=False)
  
  assert compare_results(df_output.drop('UNIQUE_REFERENCE', 'LABEL'), df_expected, ['NHS_NO', 'GIVEN_NAME', 'FAMILY_NAME', 'POSTCODE'])

  
@suite_explode_pds.add_test
def confidential_but_not_superseded():
  ''' test the removal of confidential records in conjunction with exploding PDS
  '''
  
  input_schema = (
    T.StructType([
      T.StructField('nhs_number', T.StringType()),
      T.StructField('dob', T.IntegerType()),
      T.StructField('gender_code', T.StringType()),
      T.StructField('name_history', T.ArrayType(
        T.StructType([
          T.StructField('givenNames', T.ArrayType(T.StringType())),
          T.StructField('familyName', T.StringType())
        ])
      )),
      T.StructField('address_history', T.ArrayType(
        T.StructType([
          T.StructField('postalCode', T.StringType())
        ])
      )),
      T.StructField('confidentialityCode', T.ArrayType(
        T.StructType([
          T.StructField('code', T.StringType()),
        ])
      ))
    ])
  )
  
  output_schema = (
    T.StructType([
      T.StructField('NHS_NO', T.StringType()),
      T.StructField('GIVEN_NAME', T.StringType()),
      T.StructField('FAMILY_NAME', T.StringType()),
      T.StructField('GENDER', T.StringType()),
      T.StructField('DATE_OF_BIRTH', T.IntegerType()),
      T.StructField('POSTCODE', T.StringType())
    ])
  )
    
  df_input = spark.createDataFrame(
    [
      ('1111111111', 11012001, '1', [{'givenNames': ['John', 'Mark'], 'familyName': 'Smith'}], [{'postalCode': 'LS1 1AA'}], [{'code': 'N'}]),
      ('3333333333', 11012001, '1', [{'givenNames': ['John', 'Mark'], 'familyName': 'Smith'}], [{'postalCode': 'LS1 1AA'}], [{'code': 'I'}]),
      ('5555555555', 11012001, '1', [{'givenNames': ['John', 'Mark'], 'familyName': 'Smith'}], [{'postalCode': 'LS1 1AA'}], [{'code': 'N'}])
    ],
    input_schema
  )
  
  df_expected = spark.createDataFrame(
    [
      ('1111111111', 'john mark', 'smith', '1', 11012001, 'LS1 1AA'),
      ('5555555555', 'john mark', 'smith', '1', 11012001, 'LS1 1AA'),
    ],
    output_schema
  )
  
  df_output = explode_pds(df_input, name_case='lower', postcode_case='upper', remove_confidential_records=True, remove_superseded_records=False)
  
  assert compare_results(df_output.drop('UNIQUE_REFERENCE', 'LABEL'), df_expected, ['NHS_NO', 'GIVEN_NAME', 'FAMILY_NAME', 'POSTCODE'])
  
@suite_explode_pds.add_test
def superseded_but_not_confidential():
  ''' Test the updating of superseded NHS numbers in conjunction with exploding PDS.
  '''
  
  input_schema = (
    T.StructType([
      T.StructField('nhs_number', T.StringType()),
      T.StructField('dob', T.IntegerType()),
      T.StructField('gender_code', T.StringType()),
      T.StructField('name_history', T.ArrayType(
        T.StructType([
          T.StructField('givenNames', T.ArrayType(T.StringType())),
          T.StructField('familyName', T.StringType())
        ])
      )),
      T.StructField('address_history', T.ArrayType(
        T.StructType([
          T.StructField('postalCode', T.StringType())
        ])
      )),
      T.StructField('confidentialityCode', T.ArrayType(
        T.StructType([
          T.StructField('code', T.StringType()),
        ])
      ))
    ])
  )
  
  output_schema = (
    T.StructType([
      T.StructField('NHS_NO', T.StringType()),
      T.StructField('GIVEN_NAME', T.StringType()),
      T.StructField('FAMILY_NAME', T.StringType()),
      T.StructField('GENDER', T.StringType()),
      T.StructField('DATE_OF_BIRTH', T.IntegerType()),
      T.StructField('POSTCODE', T.StringType())
    ])
  )
  
  df_pds_replaced_by = spark.createDataFrame(
    [
      ('1111111111', '2222222222'),
    ],
    ['nhs_number', 'replaced_by']
  )
    
  df_input = spark.createDataFrame(
    [
      ('1111111111', 11012001, '1', [{'givenNames': ['John', 'Mark'], 'familyName': 'Smith'}], [{'postalCode': 'LS1 1AA'}], [{'code': 'N'}]),
      ('3333333333', 11012001, '1', [{'givenNames': ['John', 'Mark'], 'familyName': 'Smith'}], [{'postalCode': 'LS1 1AA'}], [{'code': 'I'}]),
      ('5555555555', 11012001, '1', [{'givenNames': ['John', 'Mark'], 'familyName': 'Smith'}], [{'postalCode': 'LS1 1AA'}], [{'code': 'N'}])
    ],
    input_schema
  )
  
  df_expected = spark.createDataFrame(
    [
      ('3333333333', 'john mark', 'smith', '1', 11012001, 'LS1 1AA'),
      ('5555555555', 'john mark', 'smith', '1', 11012001, 'LS1 1AA'),
    ],
    output_schema
  )
  
  df_output = explode_pds(df_input, df_pds_replaced_by, name_case='lower', postcode_case='upper', remove_confidential_records=False, remove_superseded_records=True)
  
  assert compare_results(df_output.drop('UNIQUE_REFERENCE', 'LABEL'), df_expected, ['NHS_NO', 'GIVEN_NAME', 'FAMILY_NAME', 'POSTCODE'])
  
  
@suite_explode_pds.add_test
def superseded_and_confidential():
  ''' Test the updating of superseded NHS numbers and the removal of confidential records in conjunction with exploding PDS.
  '''
  
  input_schema = (
    T.StructType([
      T.StructField('nhs_number', T.StringType()),
      T.StructField('dob', T.IntegerType()),
      T.StructField('gender_code', T.StringType()),
      T.StructField('name_history', T.ArrayType(
        T.StructType([
          T.StructField('givenNames', T.ArrayType(T.StringType())),
          T.StructField('familyName', T.StringType())
        ])
      )),
      T.StructField('address_history', T.ArrayType(
        T.StructType([
          T.StructField('postalCode', T.StringType())
        ])
      )),
      T.StructField('confidentialityCode', T.ArrayType(
        T.StructType([
          T.StructField('code', T.StringType()),
        ])
      ))
    ])
  )
  
  output_schema = (
    T.StructType([
      T.StructField('NHS_NO', T.StringType()),
      T.StructField('GIVEN_NAME', T.StringType()),
      T.StructField('FAMILY_NAME', T.StringType()),
      T.StructField('GENDER', T.StringType()),
      T.StructField('DATE_OF_BIRTH', T.IntegerType()),
      T.StructField('POSTCODE', T.StringType())
    ])
  )
  
  df_pds_replaced_by = spark.createDataFrame(
    [
      ('1111111111', '2222222222'),
    ],
    ['nhs_number', 'replaced_by']
  )
    
  df_input = spark.createDataFrame(
    [
      ('1111111111', 11012001, '1', [{'givenNames': ['John', 'Mark'], 'familyName': 'Smith'}], [{'postalCode': 'LS1 1AA'}], [{'code': 'N'}]),
      ('3333333333', 11012001, '1', [{'givenNames': ['John', 'Mark'], 'familyName': 'Smith'}], [{'postalCode': 'LS1 1AA'}], [{'code': 'I'}]),
      ('5555555555', 11012001, '1', [{'givenNames': ['John', 'Mark'], 'familyName': 'Smith'}], [{'postalCode': 'LS1 1AA'}], [{'code': 'N'}])
    ],
    input_schema
  )
  
  df_expected = spark.createDataFrame(
    [
      ('5555555555', 'john mark', 'smith', '1', 11012001, 'LS1 1AA'),
    ],
    output_schema
  )
  
  df_output = explode_pds(df_input, df_pds_replaced_by, name_case='lower', postcode_case='upper', remove_confidential_records=True, remove_superseded_records=True)
  
  assert compare_results(df_output.drop('UNIQUE_REFERENCE', 'LABEL'), df_expected, ['NHS_NO', 'GIVEN_NAME', 'FAMILY_NAME', 'POSTCODE'])
  
  
suite_explode_pds.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # remove_superseded_from_pds

# COMMAND ----------

suite_remove_superseded_nhs_numbers = FunctionTestSuite()

@suite_remove_superseded_nhs_numbers.add_test
def all_scenarios():
  ''' test the removal of records with superseded NHS numbers
  '''
  
  df_input = spark.createDataFrame(
    [
      ('10', '1111111111'),
      ('11', '1111111111'),
      ('12', '3333333333'),
      ('13', '5555555555'),
      ('14', '1234567890'),
    ],
    ['test_id', 'nhs_number']
  )
  
  df_pds_replaced_by = spark.createDataFrame(
    [
      ('1111111111', '2222222222'),
      ('3333333333', '4444444444')
    ],
    ['nhs_number', 'replaced_by']
  )
  
  df_expected = spark.createDataFrame(
    [
      ('13', '5555555555'),
      ('14', '1234567890')
    ],
    ['test_id', 'nhs_number']  )
  
  df_output = remove_superseded_from_pds(df_input, df_pds_replaced_by)

  assert compare_results(df_output, df_expected, ['test_id'])
  
suite_remove_superseded_nhs_numbers.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # preprocess_full_pds

# COMMAND ----------

suite_preprocess_full_pds = FunctionTestSuite()


# this input_schema, output_schema, and df_pds_replaced_by, are used by the subsequent four tests.
input_schema = (
      T.StructType(
      [T.StructField('nhs_number', T.StringType(),True),  
       T.StructField('preferred_name',
       T.StructType([T.StructField('givenNames' ,T.ArrayType(T.StringType()), True), 
                     T.StructField('familyName', T.StringType(), True)]), True),  
       T.StructField('gender_code', T.StringType(), True),  
       T.StructField('dob', T.IntegerType(), True),  
       T.StructField('home_address', T.StructType([T.StructField('postalCode', T.StringType(), True)]), True),
       T.StructField('ConfidentialityCode', T.ArrayType(T.StructType([T.StructField('code', T.StringType(), True)])), True)])
               )

output_schema = (
    T.StructType([
        T.StructField('LABEL', T.StringType()),
        T.StructField('NHS_NO', T.StringType()),
        T.StructField('GIVEN_NAME', T.StringType()),
        T.StructField('FAMILY_NAME', T.StringType()),
        T.StructField('GENDER', T.StringType()),
        T.StructField('DATE_OF_BIRTH', T.IntegerType()),
        T.StructField('POSTCODE', T.StringType())
    ])
)

df_pds_replaced_by = spark.createDataFrame(
    [
      ('1111111111', '2222222222'),
      ('3333333333', '4444444444')
    ],
    ['nhs_number', 'replaced_by']
  )

@suite_preprocess_full_pds.add_test
def base_load():
  ''' Test preprocessing of PDS full
  '''
  
  df_input = spark.createDataFrame(
                [('12345', {'givenNames': ['given_name','given_name2'], 'familyName':'family_name'},
                   '1',19810602,{'postalCode':'AA11AA'},[{'code':[None]}]),
                 ('11111', {'givenNames': ['given_name'], 'familyName':'family_name'},
                  '1',19810602,{'postalCode':'AA11AA'},[{'code':[None]}]),
                 ('3333333333', {'givenNames': ['given_name','given_name2'], 'familyName':'family_name'},
                  '1',19810602,{'postalCode':'AA11AA'},[{'code':[None]}]),
                ],
                input_schema
                )
  df_expected = spark.createDataFrame(
                [
                  ('12345','12345','given_name given_name2','family_name','1',19810602,'AA11AA'),
                  ('11111','11111','given_name','family_name','1',19810602,'AA11AA'),
                  ('3333333333','3333333333','given_name given_name2','family_name','1',19810602,'AA11AA'),
                ],
                  output_schema
                )
    
  df_output = preprocess_full_pds(df_input, df_pds_replaced_by, False, False).drop('UNIQUE_REFERENCE')
  
  assert compare_results(df_expected,df_output,['NHS_NO'])
  
  
@suite_preprocess_full_pds.add_test
def remove_confidential_records():
  ''' Test removal of confidential records in the context of preprocessing PDS full
  '''
  
  df_input = spark.createDataFrame(
  [ 
    ('12345', {'givenNames': ['given_name','given_name2'], 'familyName':'family_name'},
       '1',19810602,{'postalCode':'AA11AA'},[{'code':None}]),
    ('11111', {'givenNames': ['given_name3','given_name4'], 'familyName':'family_name2'},
       '1',19810602,{'postalCode':'AA11AA'},[{'code':'I'}]),
    ('22222', {'givenNames': ['given_name4','given_name5'], 'familyName':'family_name3'},
       '1',19810602,{'postalCode':'AA11AA'},[{'code':'Y'}]),
    ('33333', {'givenNames': ['given_name5','given_name'], 'familyName':'family_name'},
       '1',19810602,{'postalCode':'AA11AA'},[{'code':'S'}]),
    ('3333333333', {'givenNames': ['given_name','given_name2'], 'familyName':'family_name'},
       '1',19810602,{'postalCode':'AA11AA'},[{'code':[None]}]),
    ('444444444', {'givenNames': ['given_name','given_name2'], 'familyName':'family_name'},
       '1',19810602,{'postalCode':'AA11AA'},[{'code':'N'}]),
  ],
    input_schema
  )

  df_expected = spark.createDataFrame(
  [
    ('12345','12345','given_name given_name2','family_name','1',19810602,'AA11AA'),
    ('3333333333','3333333333','given_name given_name2','family_name','1',19810602,'AA11AA'),
    ('444444444','444444444','given_name given_name2','family_name','1',19810602,'AA11AA'),
  ],
    output_schema
  )
  
  df_output = preprocess_full_pds(df_input, df_pds_replaced_by, True, False).drop('UNIQUE_REFERENCE')
  
  assert compare_results(df_expected,df_output,['NHS_NO'])

  
@suite_preprocess_full_pds.add_test
def remove_superseeded_records():
  ''' Test removal of superseded records, in the context of preprocessing PDS full
  '''
  
  df_input = spark.createDataFrame(
  [ ('12345', {'givenNames': ['given_name','given_name2'], 'familyName':'family_name'},'1',19810602,{'postalCode':'AA11AA'},[{'code':[None]}]),
   ('11111', {'givenNames': ['given_name'], 'familyName':'family_name'},'1',19810602,{'postalCode':'AA11AA'},[{'code':[None]}]),
   ('3333333333', {'givenNames': ['given_name','given_name2'], 'familyName':'family_name'},'1',19810602,{'postalCode':'AA11AA'},[{'code':[None]}]),
  ],
    input_schema
  )
  df_expected = spark.createDataFrame(
  [
    ('12345','12345','given_name given_name2','family_name','1',19810602,'AA11AA'),
    ('11111','11111','given_name','family_name','1',19810602,'AA11AA'),
  ],
    output_schema
  )
    
  df_output = preprocess_full_pds(df_input, df_pds_replaced_by, False, True).drop('UNIQUE_REFERENCE')
  
  assert compare_results(df_expected,df_output,['NHS_NO'])

  
@suite_preprocess_full_pds.add_test
def remove_confidential_and_superseeded():
  ''' Test the removal of both confidential and superseded records, in the context of preprocessing PDS full
  '''
  
  df_input = spark.createDataFrame(
  [ ('12345', {'givenNames': ['given_name','given_name2'], 'familyName':'family_name'},'1',19810602,{'postalCode':'AA11AA'},[{'code':[None]}]),
   ('11111', {'givenNames': ['given_name'], 'familyName':'family_name'},'1',19810602,{'postalCode':'AA11AA'},[{'code':[None]}]),
   ('3333333333', {'givenNames': ['given_name','given_name2'], 'familyName':'family_name'},'1',19810602,{'postalCode':'AA11AA'},[{'code':[None]}]),
   ('44444', {'givenNames': ['given_name3','given_name4'], 'familyName':'family_name2'},'1',19810602,{'postalCode':'AA11AA'},[{'code':'I'}]),
    ('55555', {'givenNames': ['given_name4','given_name5'], 'familyName':'family_name3'},'1',19810602,{'postalCode':'AA11AA'},[{'code':'Y'}]),
   ('1111111111', {'givenNames': ['given_name5','given_name'], 'familyName':'family_name'},'1',19810602,{'postalCode':'AA11AA'},[{'code':'S'}]),
  ],
    input_schema
  )
  
  df_expected = spark.createDataFrame(
  [
    ('12345','12345','given_name given_name2','family_name','1',19810602,'AA11AA'),
    ('11111','11111','given_name','family_name','1',19810602,'AA11AA'),
  ],
    output_schema
  )
    
  df_output = preprocess_full_pds(df_input, df_pds_replaced_by, True, True).drop('UNIQUE_REFERENCE')
  
  assert compare_results(df_expected,df_output,['NHS_NO'])
  
suite_preprocess_full_pds.run()
