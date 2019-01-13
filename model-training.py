from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import pandas as pd
import os
import pickle
import numpy as np
from scipy.stats import t
import traceback

def main():
    save_path = r'C:\Users\bdardin\Documents\Political Bias Project'
    
    #unpickle or load the data
    with open(os.path.join(save_path,"x.txt"), "rb") as fp:
        X = pickle.load(fp)
        
    with open(os.path.join(save_path,"y.txt"), "rb") as fp:
        Y = pickle.load(fp)
    
    seed = 12345
    
    # create a dictionary to store the models and the parameters to test    
    model_dict = {"MLP": {"model":MLPClassifier(solver='adam', activation='relu', random_state=seed),
                          "params": {"hidden_layer_sizes": [(65,), (500,), (1000,), (1500,), (2000,)]},
                          "scores": []
                          },
                  "SVM": {"model": LinearSVC(random_state=seed),
                          "params": {'C': [2**-5, 2**-3, 1, 2**3, 2**5]},
                          "scores": []
                          }
                  }
    
    # create a set of trial arrays for each model for trial-level calculations
    num_trials = 10
    for k in model_dict.keys():
        model_dict[k]["trials"] = [[] for _ in range(num_trials)]
    
    # iterate through the trials
    for i in range(num_trials):
        # create the stratified cross-validation split for this trial
        skf = StratifiedKFold(n_splits=10, random_state=i, shuffle=True)
        
        # iterate over all the folds of the cross-validation
        for j, data in enumerate(skf.split(X,Y)):
            current = "Trial "+str(i)+" Fold "+str(j)
            print(current)
            
            train_index = data[0]
            test_index = data[1]
            
            # create another stratified cross-validation split for the grid search
            skf_grid = StratifiedKFold(n_splits=5, random_state=10+i*10+j, shuffle=True)
            
            for k in model_dict.keys():
                filename = os.path.join(save_path,k,current)
                
                # conduct the grid search using only the training split of the fold
                clf = GridSearchCV(estimator=model_dict[k]["model"], param_grid=model_dict[k]["params"], cv=skf_grid, verbose=20, n_jobs=-1)
                clf.fit(X[train_index], Y[train_index])
                
                # save the results of the grid search and the winning model
                pd.DataFrame.from_dict(data=clf.cv_results_, orient='columns').to_csv(filename+'.csv', header=True)
                pickle.dump(clf, open(filename+".sav", 'wb'))
                
                # use the winning model to produce predictions on the test set of the fold
                score = clf.score(X[test_index], Y[test_index])
                model_dict[k]["scores"].append(score)
                model_dict[k]["trials"][i].append(score)
                
                with open(os.path.join(save_path,k,"results.txt"), "a",encoding='utf-8',errors='ignore') as text_file:
                    text_file.write(current+': '+str(score)+'\n\n')
    
    # calculate the mean and standard deviation of each trial            
    for k in model_dict.keys():
        model_dict[k]["trial_mean"] = []
        model_dict[k]["trial_stdev"] = []
        
        for trial in model_dict[k]["trials"]:
            model_dict[k]["trial_mean"].append(np.mean(trial))
            model_dict[k]["trial_stdev"].append(np.std(trial))
    
    # save the model dictionary
    with open(os.path.join(save_path,"model_dict.txt"), "wb") as fp:
        pickle.dump(model_dict,fp)
    
    # optionally reload the model dictionary    
    #with open(os.path.join(save_path,"model_dict.txt"), "rb") as fp:
        #model_dict = pickle.load(fp)
    
    # calculate the differences in each model's performance estimates and the mean    
    mlp_svm = np.array(model_dict["MLP"]["scores"]) - np.array(model_dict["SVM"]["scores"])
    diff_mean = np.mean(mlp_svm)
    
    # calculate the variance using the sum of squares difference of each difference
    # from the mean then divide by 10 degress of freedom
    sum_square = np.sum(np.square(mlp_svm - diff_mean))
    deg_freedom = 10
    variance = sum_square / deg_freedom
    
    # calculate the t statistic and corresponding p-value
    t_stat = ( diff_mean * np.sqrt(deg_freedom+1) ) / np.sqrt(variance)
    pval = t.sf(np.abs(t_stat), deg_freedom)*2
    
    print("mean:",diff_mean,"variance:",variance,"t-stat:",t_stat,"p-value:",pval)
                    
if __name__ == "__main__":
    # wrap the program in a try/except block in case there are errors
    try:
        main()
    except Exception as e:
        print(e)
        print(traceback.format_exc())