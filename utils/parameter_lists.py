# Databricks notebook source
name_comparisons = {
  'output_column_name': 'NAME', 
  'comparison_description': 'Name comparisons',  
  'comparison_levels': [
    { 
      # null level
      'sql_condition': '(GIVEN_NAME_l IS NULL AND FAMILY_NAME_l is NULL) OR (GIVEN_NAME_r IS NULL AND FAMILY_NAME_r is NULL)',
      'label_for_charts': 'Null',
      'is_null_level': True,
      'tf_adjustment_column': 'FULL_NAME',
      'tf_adjustment_weight': 0,
      'tf_minimum_u_value': 0.001,
    },
    {
      # level 1
      'sql_condition': 'FULL_NAME_l = FULL_NAME_r',
      'label_for_charts': 'Exact match',
      'tf_adjustment_column': 'FULL_NAME',
      'tf_adjustment_weight': 1.0,
      'tf_minimum_u_value': 0.001,
    },
    {
      #level 2
      'sql_condition': 'jaro_winkler_udf(GIVEN_NAME_l, GIVEN_NAME_r) > 0.88 AND jaro_winkler_udf(FAMILY_NAME_l, FAMILY_NAME_r) > 0.88',
      'label_for_charts': 'jaro winkler > 0.88',
      'tf_adjustment_column': 'FULL_NAME',
      'tf_adjustment_weight': 0.5,
      'tf_minimum_u_value': 0.001,
    }, 
    {
      # level 3
      'sql_condition': 'size(array_intersect(given_name_array_l, given_name_array_r)) > 0 AND size(array_intersect(family_name_array_l, family_name_array_r)) > 0',
      'label_for_charts': 'One token from each of given and family name',
      'tf_adjustment_column': 'FULL_NAME',
      'tf_adjustment_weight': 0.5,
      'tf_minimum_u_value': 0.001,
    }, 
    {
      #level 4
      'sql_condition': '(size(array_intersect(all_nicknames_l, given_name_array_r)) > 0 OR size(array_intersect(all_nicknames_r, given_name_array_l)) > 0) AND (size(array_intersect(family_name_array_l,  family_name_array_r))>0)',
      'label_for_charts': 'One nickname token and one token family name ',
      'tf_adjustment_column': 'FULL_NAME',
      'tf_adjustment_weight': 0.5,
      'tf_minimum_u_value': 0.001,
    },
    {
      #level 5a
      'sql_condition': 'GIVEN_NAME_l = GIVEN_NAME_r',
      'label_for_charts': 'Exact match on given name only',
      'tf_adjustment_column': 'GIVEN_NAME',
      'tf_adjustment_weight': 1,
      'tf_minimum_u_value': 0.001,
    },
    {
      #level 5b
      'sql_condition': 'FAMILY_NAME_l = FAMILY_NAME_r',
      'label_for_charts': 'Exact match on family name only',
      'tf_adjustment_column': 'FAMILY_NAME',
      'tf_adjustment_weight': 1,
      'tf_minimum_u_value': 0.001,
    },

    {
      # else level
      'sql_condition': 'ELSE',
      'label_for_charts': 'All other comparisons',
      'tf_adjustment_column': 'FULL_NAME',
      'tf_adjustment_weight': 0,
      'tf_minimum_u_value': 0.001,
    },  
  ]
}

# COMMAND ----------

dob_comparisons = {
  'output_column_name': 'DATE_OF_BIRTH', 
  'comparison_description': 'Date of birth comparisons',  
  'comparison_levels': [
    {
      #null level
      'sql_condition': 'DATE_OF_BIRTH_l IS NULL OR DATE_OF_BIRTH_r IS NULL',
      'label_for_charts': 'Null',
      'is_null_level': True,
      'tf_adjustment_column': 'DATE_OF_BIRTH',
      'tf_adjustment_weight': 0,
      'tf_minimum_u_value': 0.001,
    },
    {
      #level 1
      'sql_condition': 'DATE_OF_BIRTH_l = DATE_OF_BIRTH_r',
      'label_for_charts': 'Exact match',
      'tf_adjustment_column': 'DATE_OF_BIRTH',
      'tf_adjustment_weight': 1.0,
      'tf_minimum_u_value': 0.001,
    },
    {
      #level 2
      'sql_condition': 'DATE_OF_BIRTH_l = YYYYDDMM_r',
      'label_for_charts': 'Month and day swapped',
      'tf_adjustment_column': 'DATE_OF_BIRTH',
      'tf_adjustment_weight': 0.5,
      'tf_minimum_u_value':0.001,
    },
    {
      #level 3
      'sql_condition': 'levenshtein(DATE_OF_BIRTH_l, DATE_OF_BIRTH_r) <= 1',
#      'sql_condition': 'damerau_levenshtein_udf(DATE_OF_BIRTH_l, DATE_OF_BIRTH_r) <= 1',
      'label_for_charts': 'Levenshtein <=1',
      'tf_adjustment_column': 'DATE_OF_BIRTH',
      'tf_adjustment_weight': 0.5,
      'tf_minimum_u_value':0.001,
    }, 
    {
      #level 4
      'sql_condition': 'YYYYDD_l = YYYYDD_r OR YYYYMM_l = YYYYMM_r OR MMDD_l = MMDD_r',
      'label_for_charts': 'Two elements match',
      'tf_adjustment_column': 'DATE_OF_BIRTH',
      'tf_adjustment_weight': 0.5,
      'tf_minimum_u_value':0.001,
    },
    {
      #else level
      'sql_condition': 'ELSE',
      'label_for_charts': 'All other comparisons',
      'tf_adjustment_column': 'DATE_OF_BIRTH',
      'tf_adjustment_weight': 0,
      'tf_minimum_u_value': 0.001,
    },  
  ]
}


# COMMAND ----------

postcode_comparisons = {
  'output_column_name': 'POSTCODE', 
  'comparison_description': 'Postcode comparisons',  
  'comparison_levels': [
    {
      #null level
      'sql_condition': 'POSTCODE_l IS NULL OR POSTCODE_r IS NULL',
      'label_for_charts': 'Null',
      'is_null_level': True,
      'tf_adjustment_column': 'POSTCODE',
      'tf_adjustment_weight': 0,
      'tf_minimum_u_value': 0.001,
    },
    {
      #level 1
      'sql_condition': 'POSTCODE_l = POSTCODE_r',
      'label_for_charts': 'Exact match',
      'tf_adjustment_column': 'POSTCODE',
      'tf_adjustment_weight': 1.0,
      'tf_minimum_u_value': 0.001,
    },
    {
      #level 2
      'sql_condition': 'levenshtein(POSTCODE_l, POSTCODE_r) <= 1',
#      'sql_condition': 'damerau_levenshtein_udf(POSTCODE_l, POSTCODE_r) <= 1',
      'label_for_charts': 'Levenshtein <=1',
      'tf_adjustment_column': 'POSTCODE',
      'tf_adjustment_weight': 0.5,
      'tf_minimum_u_value':0.001,
    }, 
    {
      # level 3
      'sql_condition': 'SECTOR_l = SECTOR_r',
      'label_for_charts': 'Sector match',
      'tf_adjustment_column': 'SECTOR',
      'tf_adjustment_weight': 1,
      'tf_minimum_u_value':0.001,
    },
    {
      # level 4
      'sql_condition': 'OUTCODE_l = OUTCODE_r',
      'label_for_charts': 'Outcode match',
      'tf_adjustment_column': 'OUTCODE',
      'tf_adjustment_weight': 1,
      'tf_minimum_u_value':0.001,
    },
    {
      # level 5
      'sql_condition': 'REGION_l = REGION_r',
      'label_for_charts': 'Region match',
      'tf_adjustment_column': 'REGION',
      'tf_adjustment_weight': 1,
      'tf_minimum_u_value':0.001,
    },
    {
      # else level
      'sql_condition': 'ELSE',
      'label_for_charts': 'All other comparisons',
      'tf_adjustment_column': 'POSTCODE',
      'tf_adjustment_weight': 0,
      'tf_minimum_u_value': 0.001,
    },  
  ]
}


# COMMAND ----------

gender_comparisons = {
    'output_column_name': 'GENDER', 
    'comparison_description': 'Gender comparisons',  
    'comparison_levels': [
      {
        #null level
        'sql_condition': 'GENDER_l IS NULL OR GENDER_r IS NULL',
        'label_for_charts': 'Null',
        'is_null_level': True,
        'tf_adjustment_column': 'FULL_NAME',
        'tf_adjustment_weight': 0,
        'tf_minimum_u_value': 0.001,
      },
      {
        #level 1
        'sql_condition': 'GENDER_l = GENDER_r',
        'label_for_charts': 'Exact match',
        'tf_adjustment_column': 'GENDER',
        'tf_adjustment_weight': 1,
        'tf_minimum_u_value':0.001,
      },
      {
        #level 2
        'sql_condition': 'GENDER_l = 0 OR GENDER_l = 9 OR GENDER_r = 0 OR GENDER_r = 9',
        'label_for_charts': 'Gender 0 or 9',
        'tf_adjustment_column': 'GENDER',
        'tf_adjustment_weight': 0.5,
        'tf_minimum_u_value':0.001,
      },
      {
        # else level
        'sql_condition': 'ELSE',
        'label_for_charts': 'All other comparisons',
        'tf_adjustment_column': 'GENDER',
        'tf_adjustment_weight': 0,
        'tf_minimum_u_value': 0.001,
      },  
    ]
  }

# COMMAND ----------

comparisons = {'name_comparisons': name_comparisons, 'dob_comparisons': dob_comparisons, 'gender_comparisons': gender_comparisons, 'postcode_comparisons': postcode_comparisons}