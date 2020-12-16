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


def proccess_feature_importances(my_data, feature_importance_dictionary, best_feats):
    
    import pandas as pd
    import numpy as np
    
    
    feat_imp_df = pd.DataFrame(data = feature_importance_dictionary)
    feat_imp_df_final = pd.concat([feat_imp_df, pd.Series(my_data[best_feats].columns.values)], axis = 1)
    feat_imp_df_final.rename(columns = {0:'element'}, inplace = True )
    feat_imp_df_final.set_index('element', inplace=True)
    feat_imp_df_final_plot = feat_imp_df_final.T
    elements = feat_imp_df_final_plot.columns.values 
    mean_feature_importance = []
    for col in list(feat_imp_df_final_plot.columns.values):
        mean_feature_importance.append(feat_imp_df_final_plot[col].mean())
        
    mean_feature_importance_df = pd.concat([pd.Series(elements), pd.Series(mean_feature_importance)], axis = 1)
    mean_feature_importance_df.rename(columns={0:'elements', 1:'mean_importance'}, inplace=True)
    mean_feature_importance_df.sort_values(by='mean_importance', ascending=False, inplace=True)
    ordered_col_names = list(mean_feature_importance_df['elements'])
    
    return(feat_imp_df_final_plot[ordered_col_names])



def proccess_f1_vs_sample_size(site_frequencies_df, f1_df_final):
    
    import pandas as pd
    
    site_frequencies_df.rename(columns = {'Site':'class'}, inplace=True)
    f1_scores_for_plot = f1_df_final.reset_index(drop = False)
    f1_scores_for_plot['Mean F1 Score'] = f1_scores_for_plot.mean(axis = 1)
    combined_df = site_frequencies_df.set_index('class').join(other = f1_scores_for_plot.set_index('class'), how = 'inner')
    forPlot = combined_df.reset_index(drop = False).rename(columns = {'index':'class'})[['class', 'Number of Observations', 'Mean F1 Score']]
    
    return(forPlot)

    
    
    
    
    