'''
list of functions for a generalised machine learning pipeline for binary or multiclass classification tasks
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
    
    train_sizes, train_scores, test_scores = learning_curve(model, X = X, y = y, cv=cv, n_jobs=-1, train_sizes= train_sizes, shuffle = True, scoring = 'f1_macro', verbose = 2, random_state=42)
    
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
    
    df_melt = pd.melt(df, id_vars=['train_sizes'], value_vars=['train_score', 'test_scores'], var_name='train_or_test_data', value_name='k_fold_stratified_scores')
    
    df_melt.sort_values(by = 'train_sizes', ascending=True, inplace=True)
    
    return(df_melt)



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



class myModel:
    
    '''
    class for fitting best possible random forest models using available trainig data by optimisation of hyperparamaters
    with cross-validation.
    '''

    def __init__(self, X, y):
        
        '''
        Paramaters:
        
            X : pandas dataframe, training data features
            
            y : pandas series, training data targets
            
        Attributes:
        
            self.X : pandas dataframe, training data features
            
            self.y : pandas series, training data targets
            
        '''
        
        self.X = X 
        self.y = y

        
    def evaluate_rfc(self):
        
        '''
        
        Evaluates best radom forest classifiers with available data by 10-fold stratified cross validation. Within each train fold
        this again undergoes 10-fold stratified cross-validation with grid search to identify best hyperparamaters for evaluating
        best model.
        
        Paramters : None
        
        Attributes :
            
            all attributes are evaluation metrics for 10-fold stratified cross validation
            self.class_f1_scores : numpy array, averaged class specific f1 scores
            self.accuracy_scores : list, accuracy scores
            self.macro_f1_scores : list, overall F1 scores
            self.f1_dict : dictionary, contains class specific f1 scores 
            self.feat_imp_dict : dictionary, contains feature important scores

        
        '''
        
        from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.ensemble import RandomForestClassifier
        from imblearn.over_sampling import SMOTE

        if True:
            skf = StratifiedKFold(n_splits=10, random_state=42)
            

            class_f1_scores = []
            macro_f1_scores = []
            accuracy_scores = []
            feat_imp =[]
            f1_dict = {}
            feat_imp_dict = {}
            count = 0



            for train_index, test_index in skf.split(self.X, self.y):

                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index] 

                #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y)
                count = count + 1
                print('making model:')
                key = 'round' + str(count)
                print(count)


                X_post_smote, y_post_smote = SMOTE(random_state=42).fit_sample(X_train, y_train)


                ###this section optimises model paramaters by gridsearch 

                esti = RandomForestClassifier(max_features = 'auto', random_state = 42, n_jobs = -1)

                n_estimators = [20, 25, 30]
                max_depth = [8, 10, 12]
                min_samples_split = [2, 3, 4]
                min_samples_leaf = [1, 2, 3]

                param_grid = {
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'n_estimators':n_estimators 
                              }

                clf = GridSearchCV(estimator = esti, param_grid= param_grid,
                                          n_jobs=-1, scoring='f1_macro', cv = 5, verbose=3)
                print('running grid search on this training data fold')
                clf.fit(X_post_smote, y_post_smote)
                optimumEstimator = clf.best_estimator_
                optimumEstimator.fit(X_post_smote, y_post_smote)
                print('''gridsearch identified optimum paramaters for the current training data fold, now model with optimumn
                      paramaters predicting using test_data for evaluation''')

                y_pred = optimumEstimator.predict(X_test)
                class_f1_scores = f1_score(y_test, y_pred, average = None)
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_scores.append(accuracy)
                macro_f1_scores.append(f1_score(y_test, y_pred, average = 'macro'))
                f1_dict[key] = class_f1_scores 
                feat_imp_dict[key] = optimumEstimator.feature_importances_
                
            self.class_f1_scores = class_f1_scores
            self.accuracy_scores = accuracy_scores
            self.macro_f1_scores = macro_f1_scores
            self.f1_dict = f1_dict
            self.feat_imp_dict = feat_imp_dict
            
            
    def evaluate_svm(self):
        
        '''
        
        Evaluates best support vector machines with available data by 10-fold stratified cross validation. Within each train fold
        this again undergoes 10-fold stratified cross-validation with grid search to identify best hyperparamaters for evaluating
        best model.
        
        Paramaters : None
        
        Attributes :
            
            all attributes are evaluation metrics for 10-fold stratified cross validation
            self.class_f1_scores : numpy array, averaged class specific f1 scores
            self.accuracy_scores : list, accuracy scores
            self.macro_f1_scores : list, overall F1 scores
            self.f1_dict : dictionary, contains class specific f1 scores 
            self.feat_imp_dict : dictionary, contains feature important scores

        
        '''
        
        from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.svm import SVC
        from imblearn.over_sampling import SMOTE

        if True:
            skf = StratifiedKFold(n_splits=10, random_state=42)
            

            class_f1_scores = []
            macro_f1_scores = []
            accuracy_scores = []
            feat_imp =[]
            f1_dict = {}
            feat_imp_dict = {}
            count = 0



            for train_index, test_index in skf.split(self.X, self.y):

                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index] 

                #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y)
                count = count + 1
                print('making model:')
                key = 'round' + str(count)
                print(count)


                X_post_smote, y_post_smote = SMOTE(random_state=42).fit_sample(X_train, y_train)


                ###this section optimises model paramaters by gridsearch 

                esti = SVC()
        
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100], 
                    'class_weight':['balanced', None], 
                    'gamma':[0.001, 0.01, 0.1, 1, 10]
                    }

                clf = GridSearchCV(estimator = esti, param_grid= param_grid,
                                  n_jobs=-1, scoring='f1_macro', cv = 5, verbose=3)

            
                print('running grid search on this training data fold')
                clf.fit(X_post_smote, y_post_smote)
                optimumEstimator = clf.best_estimator_
                optimumEstimator.fit(X_post_smote, y_post_smote)
                print('''gridsearch identified optimum paramaters for the current training data fold, now model with optimumn
                      paramaters predicting using test_data for evaluation''')

                y_pred = optimumEstimator.predict(X_test)
                class_f1_scores = f1_score(y_test, y_pred, average = None)
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_scores.append(accuracy)
                macro_f1_scores.append(f1_score(y_test, y_pred, average = 'macro'))
                f1_dict[key] = class_f1_scores 
                
            self.class_f1_scores = class_f1_scores
            self.accuracy_scores = accuracy_scores
            self.macro_f1_scores = macro_f1_scores
            self.f1_dict = f1_dict
            
            
            
    def evaluate_knn(self):
        
        '''
        
        Evaluates best k-nearest neighbours classifiers with available data by 10-fold stratified cross validation. Within each
        train fold
        this again undergoes 10-fold stratified cross-validation with grid search to identify best hyperparamaters for evaluating
        best model.
        
        Paramters : None
        
        Attributes :
            
            all attributes are evaluation metrics for 10-fold stratified cross validation
            self.class_f1_scores : numpy array, averaged class specific f1 scores
            self.accuracy_scores : list, accuracy scores
            self.macro_f1_scores : list, overall F1 scores
            self.f1_dict : dictionary, contains class specific f1 scores 
            self.feat_imp_dict : dictionary, contains feature important scores

        
        '''
        
        from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.neighbors import KNeighborsClassifier
        from imblearn.over_sampling import SMOTE

        if True:
            skf = StratifiedKFold(n_splits=10, random_state=42)
            

            class_f1_scores = []
            macro_f1_scores = []
            accuracy_scores = []
            feat_imp =[]
            f1_dict = {}
            feat_imp_dict = {}
            count = 0



            for train_index, test_index in skf.split(self.X, self.y):

                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index] 

                #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y)
                count = count + 1
                print('making model:')
                key = 'round' + str(count)
                print(count)


                X_post_smote, y_post_smote = SMOTE(random_state=42).fit_sample(X_train, y_train)


                ###this section optimises model paramaters by gridsearch 

                esti = KNeighborsClassifier()

                n_neighbors = [4, 6, 8, 10, 12, 14]
                weights = ['uniform', 'distance']


                param_grid = {
                           'n_neighbors': n_neighbors,
                           'weights': weights, 
                              }

                clf = GridSearchCV(estimator = esti, param_grid= param_grid,
                                          n_jobs=-1, scoring='f1_macro', cv = 5, verbose=3)
                print('running grid search on this training data fold')
                clf.fit(X_post_smote, y_post_smote)
                optimumEstimator = clf.best_estimator_
                optimumEstimator.fit(X_post_smote, y_post_smote)
                print('''gridsearch identified optimum paramaters for the current training data fold, now model with optimumn
                      paramaters predicting using test_data for evaluation''')

                y_pred = optimumEstimator.predict(X_test)
                class_f1_scores = f1_score(y_test, y_pred, average = None)
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_scores.append(accuracy)
                macro_f1_scores.append(f1_score(y_test, y_pred, average = 'macro'))
                f1_dict[key] = class_f1_scores 

                
            self.class_f1_scores = class_f1_scores
            self.accuracy_scores = accuracy_scores
            self.macro_f1_scores = macro_f1_scores
            self.f1_dict = f1_dict



            

            
            



class bestHyperparamaters:
    
    '''
    class for optimising hyperparamaters over entire training dataset to 
    get best model for fitting to unknown data
    '''
    
    def __init__(self, X, y):
        
        '''
        Paramaters:
        
            X : pandas dataframe, training data features
            
            y : pandas series, training data targets
            
        Attributes:
        
            self.X : pandas dataframe, training data features
            
            self.y : pandas series, training data targets
            
        '''
        
        self.X = X
        self.y = y
    


    
    
    def get_best_model(self):
        
        '''
        searches for best set of hyperparamaters for random forest classifier
        
        Paramaters:
        
            None
            
        attributes:
            
            self.best_model : sklearn random forest classifier model with best selection of hyperparamaters
            
            
        '''
        
        from imblearn.over_sampling import SMOTE
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV

        X_post_smote, y_post_smote = SMOTE(random_state=42).fit_sample(self.X, self.y)
        
        self.X_post_smote = X_post_smote
        self.y_post_smote = y_post_smote

        esti = RandomForestClassifier(n_estimators=5, random_state = 42)


        max_depth = [8, 10]
        min_samples_split = [3, 4]
        min_samples_leaf = [1, 2, 3]

        param_grid = {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                        }

        clf = GridSearchCV(estimator = esti, param_grid= param_grid,
                                          n_jobs=-1, scoring='f1_macro', cv = 10, verbose=3)
        clf.fit(X_post_smote, y_post_smote)
        esti_final = clf.best_estimator_
        
        self.best_model = esti_final
        
        
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
    

    