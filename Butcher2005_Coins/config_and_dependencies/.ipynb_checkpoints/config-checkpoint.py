'''
configurations, these should be set before running the pipeline
'''

'''
descriptions for paramaters
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

#edrock_only = True
data_input_path = '../Butcher2005_Denarii.csv'
#drop_semi_bedrock = True
features_start = 6
features_end = -1
target = 'class'
#known_idententifier_col = 'Geology'
#known_identifier_value = 'samples'
#unknown_identifier_value = 'Artefacts'
random_seed_state = 42
