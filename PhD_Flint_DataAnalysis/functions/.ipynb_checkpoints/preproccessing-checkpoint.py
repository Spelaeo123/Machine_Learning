'''
Functions for preproccessing data
'''



def clean_columns(data, features_start, features_end):

    '''
    cleans strings and unwanted characters from feature data
    '''
    
    import swifter
    import pandas as pd
    import numpy as np
    
    
    for column_name in data.columns.values[features_start:features_end]:
        print(column_name)
        def fill_less_than(row):
            if 'DL' in  str(row[column_name]):
                return(np.nan)
            if '<' in str(row[column_name]):
                return(float(row[column_name].replace('<', '').replace(',', '')))
            else:
                return(float(row[column_name]))
        data[column_name] = data.swifter.apply(fill_less_than, axis = 1)
        
    return(data)


def split_data(data, known_identifier_col, unknown_identifier_value):
    
    '''
    splits data into two datasets, one contains observations for which the target is known
    the other for which the target is unknown
    
    Paramaters:
        
        data : pandas dataframe, dataset containing all observations
        
        known_identifier_column : string, column name for column that identifies rows as 
            known or not known
        
        unknown_identifier_value :string/int, value in known_identifier_column 
        that encodes target as unknown
        
    Returns:
        my_data_known : pandas dataframe, contains data for which target is known
        
        my_data_unknown : pandas dataframe, contains data for which target is unknown
            
    '''
    
    my_data_known = data[data[known_identifier_col] != unknown_identifier_value]
    my_data_unknown = data[data[known_identifier_col] == unknown_identifier_value]
    
    return(my_data_known, my_data_unknown)





def replace_outliers(data, features_start, features_end, num_stds):
    
    '''
    replace values more than num_stds standard deviations away from the feature mean with the mean. The function
    assumes teh features are all adjacent to eachother in the column ordering, if they are not this won't work
    
    Paramaters:
    
        data: data, should be either data for which the target is known or for which the target is unknown.
        Should not be a concatenation of the two otherwise there would be data leakage.
        
        features_start : integer, index in columns of data that contains first feature column
        
        features_end : integer, index in columns of data that contains last feature column
        
        num_stds : integer, the number of standard deviations away from the mean that a value is deemed as an outlier and is
        replaced    
        
    Returns : 
    
        data : pandas dataframe, data with outliers replaced with mean value for the associated column
    '''
    
    import swifter
    import pandas as pd
    import numpy as np
    
    std_dict_geo = {}
    mean_dict_geo = {}
    median_dict_geo = {}

    for col in data.columns.values[features_start:features_end]:
        std_dict_geo[col] = data[col].std()

    for col in data.columns.values[features_start:features_end]:
        mean_dict_geo[col] = data[col].mean()

    for col in data.columns.values[features_start:features_end]:
        median_dict_geo[col] = data[col].median()


    for col_name in data.columns.values[features_start:features_end]:
        def impute_outliers_geo(row):
            if np.abs(row[col_name] - mean_dict_geo[col_name]) > 2*(std_dict_geo[col_name]):
                return(mean_dict_geo[col_name])
            else:
                return(row[col_name])
        data[col_name]= data.swifter.apply(impute_outliers_geo, axis = 1)

    return(data)