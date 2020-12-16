'''
configurations, these should be set before running the pipeline
'''

'''
descriptions for parameters
'''

'''drop_semi_bedrock ->  boolean, if set to True then some bedrock sites deemed to be semi-bedrock sites are not used for classification

bedrock_only -> boolean, set to True if want to only classify into bedrock sites

drop_semi_bedrock -> string, 'BM' | 'BC', 

features_start -> int, column index of dataframe that equates to first feature column

features_end -> int, column index of dataframe that equates to last feature column

target -> string, name of column to be classified

known_idententifier_col -> string, name of column that identifies whether source of samples is known or not

known_identifier_value -> string, value in known_identifier_col that encodes that the source of the observation
is known

unknown_identifier_value -> string, value in known_identifier_col that encodes that the source of the observation
is unknown

random_seed_state -> int, number to set randomness
'''


'''
paramaters to be set
'''



data_input_path = './Data/Provincial_JC_to_Trajans_reform_final.csv'

features_start = 13
features_end = -2
target = 'target'

random_seed_state = 42
