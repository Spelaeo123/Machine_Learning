'''
Functions for modeling data
'''

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
        
        

    

    