'''
functions to assess data
'''



def learningCurve(model, X, y, cv, train_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    
    '''
    creates dataset for plotting learning curve
    
    Paramters:
    
        model : sklearn model, classifier
        
        X : pandas dataframe, known features
        
        y: pandas series, target
        
        train_sizes : list/numpy array, containimg decimals that 
            represent the fraction of dat to be used as training data
    Returns : 
        
        learning curve plot
        
    '''
    
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(model, X = X, y = y, cv=cv, n_jobs=-1, train_sizes= train_sizes,
                                                            shuffle = True, scoring = 'f1_macro', verbose = 2, random_state=42)
    
    train_scores_flat = []
    for i in train_scores:
        for i_2 in i:
            train_scores_flat.append(i_2)
            
    test_scores_flat = []
    for i in test_scores:
        for i_2 in i:
            test_scores_flat.append(i_2)
            
    train_sizes_duped = []
    for i in train_sizes:
        for i_2 in range(0, cv):
            train_sizes_duped.append(i)
            
    df = pd.DataFrame(data = {'train_sizes':train_sizes_duped, 'train_score':train_scores_flat, 'test_scores':test_scores_flat})
    
    df_melt = pd.melt(df, id_vars=['train_sizes'], value_vars=['train_score', 'test_scores'], var_name='train_or_test_data',
                      value_name='k_fold_stratified_scores')
    
    df_melt.sort_values(by = 'train_sizes', ascending=True, inplace=True)
    
    return(df_melt)