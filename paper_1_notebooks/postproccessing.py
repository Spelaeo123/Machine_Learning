'''
Functions to proccess results
'''


def process_results(model, data, best_feats, uniques, identifiers):
    
    '''
    
    creates dataframe containing predictions from model, unique identifiers and probabilities for all classes
    
    Paramaters:
    
        model : sklearn random forest classifier
    
        data : pandas dataframe, dataset with features for unkown observations to be classified and indicator of whether outlier or
        not 
    
        best_feats : list, most predictive combination of features
        
        uniques : list, unique classes to be predicted
        
        identifiers : list, unique identifiers for samples to be classified
        
    Returns:
    
        final_pred_df : pandas dataframe, contains predictions from model, unique identifiers and probabilities for all classes
        
    '''
    
    import pandas as pd
    import numpy as np
    import swifter
    
    y_pred = model.predict(np.array(data[best_feats]))
    y_pred_proba = model.predict_proba(np.array(data[best_feats]))
    
    probabilities_df = pd.DataFrame(data = y_pred_proba, columns = uniques)
    probabilities_df_final = pd.concat([probabilities_df, identifiers.reset_index(drop = True)], axis = 1)
    
    final_pred_df = pd.concat([pd.Series(y_pred), probabilities_df_final], axis = 1).rename(columns={0:'class_number'})
    
    final_predictions_df = pd.concat([final_pred_df, data['inlierLabel']], axis = 1)
    
    uniques_list = list(uniques)
    def get_pred_names(row):
        return(uniques_list[row['class_number']])
    
    final_predictions_df['class_predictions'] = final_predictions_df.apply(get_pred_names, axis = 1)

    def outlierAssigner(row):
        if row['inlierLabel'] == -1:
            return('other')
        else:
            return(row['class_predictions'])
        
    final_predictions_df['class_predictions'] = final_predictions_df.swifter.apply(outlierAssigner, axis = 1)
    
    return(final_predictions_df)