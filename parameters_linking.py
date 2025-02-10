# Databricks notebook source
import splink.spark.spark_comparison_library as cl

# COMMAND ----------

# MAGIC %run ./utils/pipeline_utils

# COMMAND ----------


class Params:
    def __init__(self, integration_tests=True, comparisons={}, hash_to_use = ''):
      
        self.integration_tests = integration_tests

        self.DATABASE = "mps_enhancement_collab"
        self.TABLE_MODEL_RUNS = "splink_model_log"

        self.POSTCODES_TO_IGNORE = []

        self.TRAINING_GIVEN_NAME_COLUMN = "GIVEN_NAME"
        self.TRAINING_FAMILY_NAME_COLUMN = "FAMILY_NAME"
        self.TRAINING_GENDER_COLUMN = "GENDER"
        self.TRAINING_POSTCODE_COLUMN = "POSTCODE"
        self.TRAINING_DOB_COLUMN = "DATE_OF_BIRTH"
        self.TRAINING_UNIQUE_REFERENCE_COLUMN = "UNIQUE_REFERENCE"

        self.PROPORTION_OF_LINKS_EXPECTED = 0.5  # due to the training sample containing approx 50% Scottish or Northern Irish
        self.MATCH_WEIGHT_THRESHOLD = 5
        self.CLOSE_MATCHES_THRESHOLD = 5

        self.COMPARISONS_LIST = list(comparisons.values())

        if self.integration_tests:
            self.TRAINING_TABLE = "temp_splink_integration_test_data"
            self.LINKING_TABLE = "temp_splink_integration_test_data"

            self.TRAINING_PDS_TABLE = "temp_splink_integration_test_data"
            self.TRAINING_LINKING_TABLE = "temp_splink_integration_test_data"

            self.DATABASE_PDS = "mps_enhancement_collab"
            self.TABLE_PDS = "temp_splink_integration_test_pds_subsample"

            self.MATCH_PROBABILITIES_TABLE = "temp_splink_integration_test_match_probabilities"
            self.BEST_MATCH_TABLE = "temp_splink_integration_test_best_match"

            self.MODEL_DESCRIPTION = "temp_splink_integration_test_model_linking" 

        elif not hash_to_use:
            self.MODEL_DESCRIPTION = ""  # setting for creating the hash
            self.TABLE_NAME = "splink_training_data_20240710"
            self.LINKING_TABLE = "splink_evaluation_data_20240710"

            self.BR_TRAINING_TABLE = "training_data_subset_20241210"
            self.BR_LINKING_TABLE = "pds_data_subset_20241210"

            self.TRAINING_PDS_TABLE = "temp_pds_full_preprocessed"
            self.TRAINING_LINKING_TABLE = "temp_training_preprocessed"

            self.DATABASE_PDS = "pds"
            self.TABLE_PDS = "full"

            self.MATCH_PROBABILITIES_TABLE = f"match_probabilities_{self.MODEL_DESCRIPTION}"
            self.BEST_MATCH_TABLE = f"best_matches_{self.MODEL_DESCRIPTION}"

        self.BR_COMPARISONS = [
            cl.exact_match("NHS_NO"),
            cl.exact_match("GIVEN_NAME"),
            cl.exact_match("FAMILY_NAME"),
            cl.exact_match("SOUNDEX_FAMILY_NAME"),
            cl.exact_match("DATE_OF_BIRTH"),
            cl.exact_match("YYYYDD"),
            cl.exact_match("POSTCODE"),
            cl.exact_match("OUTCODE"),
            cl.exact_match("GENDER"),
        ]
        
        self.BLOCKING_RULES = [
          'l.NHS_NO = r.NHS_NO',
          'l.DATE_OF_BIRTH = r.DATE_OF_BIRTH and l.GENDER = r.GENDER and l.OUTCODE = r.OUTCODE',
          'l.GENDER = r.GENDER and l.SOUNDEX_FAMILY_NAME = r.SOUNDEX_FAMILY_NAME and l.DATE_OF_BIRTH = r.DATE_OF_BIRTH',
          'l.GENDER = r.GENDER and l.FAMILY_NAME = r.FAMILY_NAME and l.YYYYDD = r.YYYYDD',
          'l.POSTCODE = r.POSTCODE',
          'l.GIVEN_NAME = r.GIVEN_NAME and l.DATE_OF_BIRTH = r.DATE_OF_BIRTH',
          'l.GIVEN_NAME = r.GIVEN_NAME and l.OUTCODE = r.OUTCODE',
          'l.FAMILY_NAME = r.FAMILY_NAME and l.DATE_OF_BIRTH = r.DATE_OF_BIRTH',
          'l.FAMILY_NAME = r.FAMILY_NAME and l.OUTCODE = r.OUTCODE',
        ]
        
        self.BLOCKING_RULES_FOR_TRAINING = [
          'l.DATE_OF_BIRTH = r.DATE_OF_BIRTH and l.POSTCODE = r.POSTCODE and l.GENDER = r.GENDER',
          'l.GIVEN_NAME = r.GIVEN_NAME and l.FAMILY_NAME = r.FAMILY_NAME and l.POSTCODE = r.POSTCODE and l.GENDER = r.GENDER',
          'l.GIVEN_NAME = r.GIVEN_NAME and l.FAMILY_NAME = r.FAMILY_NAME and l.DATE_OF_BIRTH = r.DATE_OF_BIRTH and l.GENDER = r.GENDER',
          'l.GIVEN_NAME = r.GIVEN_NAME and l.FAMILY_NAME = r.FAMILY_NAME and l.DATE_OF_BIRTH = r.DATE_OF_BIRTH and l.POSTCODE = r.POSTCODE',
          'l.GIVEN_NAME = r.GIVEN_NAME and l.FAMILY_NAME = r.FAMILY_NAME and l.DATE_OF_BIRTH = r.DATE_OF_BIRTH',
          'l.GIVEN_NAME = r.GIVEN_NAME and l.FAMILY_NAME = r.FAMILY_NAME and l.POSTCODE = r.POSTCODE',
          'l.DATE_OF_BIRTH = r.DATE_OF_BIRTH and l.POSTCODE = r.POSTCODE',
        ]
        
        self.GIVEN_NAMES_TO_IGNORE = [
                              'null', 'none', '^\W+$', '\d',
                              'used to be known as', 'also known as', 'likes to be known as', 'wants to be known as', 'prefers to be known as', 'known as',
                              'formerly known as', 'aka', 'alias', 'formerly', 'previously known as',
                              'unknown', 'undecided',
                              'one of',
                              'girl one', 'boy one', 'female one', 'male one', 'one girl', 'two', 'baby girl', 'baby boy', 'babyone', 'boy', 'girl',
                              '\b(twin|boy|girl|one|i|two|ii|baby girl|baby boy|second|female baby|male baby|baby|infant|inf|babyof) of\b.*',
                              'triplet', 'tripletone', 'triplettwo', 'tripletthree',
                              'mr', 'mrs', 'dr', 'miss', 'ms', 'mx', 'mstr', 'prof', 'sir', 'doctor', 'professor', 'count', 'countess', 'dame', 'reverend', 
                              'father', 'mother', 'sister', 'brother', 'pastor', 'archbishop',
                              'no name', 'name', 'forename',
                              'nee', 'ne', 'nickname', 'given name',
                              'female baby', 'male baby', 'baby', 'female', 'male',
                              '(?:twin(?:\s+(?:one|i|1|2|ii|two)?))', '(?:triplet(?:\s+(?:one|i|1|2|ii|two|three|iii|3")?))',
                              'two',
                              'infant'
                            ] # The regular expression matches the string "twin" optionally followed by whitespace and one of the options listed. 
                              # This allows for variations: "twin one", "twin i", "twin 1", "twin 2", "twin ii", or "twin two", also does this for triplets.
          
        self.PREPROCESS_POSTCODE_ARGS = {
            "postcode_column": self.TRAINING_POSTCODE_COLUMN
        }

        self.PREPROCESS_GIVEN_NAME_ARGS = {
              "name_column": self.TRAINING_GIVEN_NAME_COLUMN,
              "split_names": False,
              "names_to_ignore": self.GIVEN_NAMES_TO_IGNORE,
              "create_nicknames_col": True,
              "nicknames_df": spark.table("mps_enhancement_collab.nicknames_data"),
          }

        self.PREPROCESS_DOB_ARGS = {
              "dob_column": self.TRAINING_DOB_COLUMN,
              "subset_types": ["YYYYDD", "YYYYMM", "MMDD", "YYYYDDMM"],
        }

        self.PREPROCESS_FAMILY_NAME_ARGS = {
              "name_column": self.TRAINING_FAMILY_NAME_COLUMN,
              "apply_soundex": True,
        }

        self.PREPROCESS_FULL_NAME_ARGS = {
              "given_name_col": self.TRAINING_GIVEN_NAME_COLUMN,
              "family_name_col": self.TRAINING_FAMILY_NAME_COLUMN,
        }