from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
import sys


# Import train data
train = sys.argv[1]
train_data = pd.read_csv(train)

features = list(train_data.columns[0:4])
label = train_data.columns[4]

X_train = train_data[features]
y_train = train_data[label]
# Import test data
test = sys.argv[2]
test_data = pd.read_csv(test)
X_test = test_data[features]
y_test = test_data[label]


# Adaboost
def adaboost():
	adaboost = AdaBoostClassifier(n_estimators = 100, learning_rate=1, algorithm='SAMME.R',random_state=0)#base_estimator=None(defult) => it is DecisionTreeClassifier(max_depth=1)
	ADA_model = adaboost.fit(X_train, y_train)
	y_pred_ADA_test = ADA_model.predict(X_test)
	return(ADA_model)

# Random Forest 
def random_forest():
	forest = RandomForestClassifier(n_estimators=100, max_features=3, random_state=0)#max_depth = None (default) to grow each tree to the largest extent possible
	RF_model = forest.fit(X_train, y_train)
	y_pred_RF_test = RF_model.predict(X_test)
	return(RF_model)

# Neural Net
def neural_net():
	net = MLPClassifier(hidden_layer_sizes =(400,), activation = 'tanh', solver = 'adam', learning_rate = 'constant', max_iter = 250, momentum = 0.89, early_stopping = True, alpha=0.0001, learning_rate_init=0.001, power_t=0.5)
	NET_model = net.fit(X_train, y_train)
	y_pred_NET_test = NET_model.predict(X_test)
	return(NET_model)

# Logistic Regression
def logistic_regression():
	logreg = LogisticRegression(penalty = 'l2', C =1.5, fit_intercept=True, solver='liblinear', max_iter=400)
	LR_model = logreg.fit(X_train, y_train)
	y_pred_LR_test = LR_model.predict(X_test)
	return(LR_model)

# KNN
def knn():
	knn = KNeighborsClassifier(n_neighbors=3)
	KNN_model = knn.fit(X_train, y_train)
	y_pred_KNN_test = KNN_model.predict(X_test)
	return(KNN_model)


# Naive Bayes
def naive_bayes():
	nb = MultinomialNB(alpha=1.0, fit_prior=True)
	NB_model = nb.fit(X_train, y_train)
	y_pred_NB_test = NB_model.predict(X_test)
	return(NB_model)

# Decision Tree
def decision_tree():
	tree = DecisionTreeClassifier()
	TREE_model = tree.fit(X_train, y_train)
	y_pred_TREE_test = TREE_model.predict(X_test)
	return(TREE_model)


# Gridsearch with different hyperparameters
def tuning(model, hyperparameters):
	grid = GridSearchCV(estimator=model, param_grid=hyperparameters, n_jobs=-1, iid=True, cv=5, return_train_score=True, scoring='accuracy')
	grid.fit(X_train, y_train)

	train_means = grid.cv_results_['mean_train_score']
	train_sds = grid.cv_results_['std_train_score']
	test_means = grid.cv_results_['mean_test_score']
	test_sds = grid.cv_results_['std_test_score']
	mean_fit_times = grid.cv_results_['mean_fit_time']
	results = []
	for train_mean, train_sd, test_mean, test_sd, mean_fit_time, parameters in zip(train_means, train_sds, test_means, test_sds, mean_fit_times, grid.cv_results_['params']):
		results.append([train_mean, train_sd, test_mean, test_sd, mean_fit_time, parameters])
	results_df = pd.DataFrame(results, columns=('trainAvg', 'trainSD', 'testAvg', 'testSD', 'fitTime', 'parameters'))
	print(tabulate(results_df, headers='keys'))
	return(grid.best_params_)

# Detect the best model and output results
def best_model(model, best_hyperparameters):
	best_model = model
	best_hyperparameters = best_hyperparameters
	best_model.set_params(**best_hyperparameters)
	best_model.fit(X_train, y_train)
	y_pred_test = best_model.predict(X_test)

	print("\nAccuracy on training set of the best model", best_model.score(X_train, y_train))
	print("Accuracy on testing set of the best model", best_model.score(X_test, y_test))
	matrix_test = confusion_matrix(y_test, y_pred_test)
	tn, fp, fn, tp = matrix_test.ravel()
	print("Confusion matrix for test set of the best model:\n", matrix_test)
	print("\nTrue Negatives: ",tn)
	print("False Positives: ",fp)
	print("False Negatives: ",fn)
	print("True Positives: ",tp)
	print("Parameters of the best model: ", best_hyperparameters)
	return(best_model, best_model.score(X_test, y_test))

# Voting: pass in models and specify the weights
def voting(list_of_models, weights):
	ensemble = VotingClassifier(estimators=list_of_models, voting='hard', weights = weights, n_jobs=-1)
	ensemble = ensemble.fit(X_train, y_train)
	y_pred_test = ensemble.predict(X_test)
	print("\nAccuracy on training set of the ensemble model", ensemble.score(X_train, y_train))
	print("Accuracy on testing set of the ensemble model", ensemble.score(X_test, y_test))
	matrix_test = confusion_matrix(y_test, y_pred_test)
	tn, fp, fn, tp = matrix_test.ravel()
	print("Confusion matrix for test set:\n", matrix_test)
	print("\nTrue Negatives: ",tn)
	print("False Positives: ",fp)
	print("False Negatives: ",fn)
	print("True Positives: ",tp)
	return(ensemble)

print("Task 1:\n")

print("Random Forest Model: ")
forest_params = [{'n_estimators': [30, 35, 45, 50, 55, 100], 'criterion': ('gini', 'entropy'), 'oob_score': (True, False), 'class_weight': ('balanced', 'balanced_subsample', None)}]
RF_best_parameters = tuning(random_forest(), forest_params)
RF_best_model=best_model(model=RandomForestClassifier(), best_hyperparameters=RF_best_parameters)

print("\nAdaboost Model: ")
adaboost_params = [{'n_estimators': [35, 45, 50, 55, 100, 200], 'learning_rate': [1]}]
ADA_best_parameters = tuning(adaboost(), adaboost_params)
#ADA_best_parameters = {'n_estimators': 100, 'learning_rate': 1, 'algorithm': 'SAMME'}
ADA_best_model=best_model(model=AdaBoostClassifier(), best_hyperparameters=ADA_best_parameters)

print("\n###################################################################")
print("Task 2:")

print("\nNeural Net Model: ")
nn_params = [{'hidden_layer_sizes': [(100,), (200,)], 'activation': ['relu', 'tanh'], 'solver':['adam', 'lbfgs'],
'learning_rate': ['constant','adaptive'],'momentum': [0.9], 'early_stopping': [False, True]}]
NN_best_parameters = tuning(neural_net(), nn_params)
NN_best_model=best_model(model=MLPClassifier(), best_hyperparameters=NN_best_parameters)

print("\nLogistic Regression Model: ")
lr_params = [{'penalty': ['l2', 'l1'], 'C': [1, 1.5, 2], 'fit_intercept': [False, True], 'solver': ['liblinear'], 'max_iter': [200,400]}]
LR_best_parameters = tuning(logistic_regression(), lr_params)
LR_best_model=best_model(model=LogisticRegression(), best_hyperparameters=LR_best_parameters)

print("\nKNN Model: ")
knn_params = [{'n_neighbors': [1, 2, 3, 4, 5]}]
KNN_best_parameters = tuning(knn(), knn_params)
KNN_best_model=best_model(model=KNeighborsClassifier(), best_hyperparameters=KNN_best_parameters)

print("\nNaive Bayes Model: ")
nb_params = [{'alpha': [1.0, 1.5], 'fit_prior':[False, True]}]
NB_best_parameters = tuning(naive_bayes(), nb_params)
NB_best_model=best_model(model=MultinomialNB(), best_hyperparameters=NB_best_parameters)

print("\nDecision Tree Model: ")
dt_params = [{'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_depth': [None, 5,10], 'max_features':[None, 3]}]
DT_best_parameters = tuning(decision_tree(), dt_params)
DT_best_model=best_model(model=DecisionTreeClassifier(), best_hyperparameters=DT_best_parameters)

models = [('NN_best_model', NN_best_model[0]),('LR_best_model', LR_best_model[0]), ('KNN_best_model', KNN_best_model[0]), ('NB_best_model', NB_best_model[0]),  ('DT_best_model', DT_best_model[0]), ('RF_best_model', RF_best_model[0]), ('ADA_best_model', ADA_best_model[0])]
weights_accuracy = [NN_best_model[1], LR_best_model[1], KNN_best_model[1], NB_best_model[1], DT_best_model[1], RF_best_model[1], ADA_best_model[1]]

print("\nEnsemble model: unweigted majority voting over 5 models")
voting(list_of_models = models[0:4], weights = None)

print("\nEnsemble model: weigted majority voting over 5 models (weights proportional to accuracy)")
voting(list_of_models = models[0:4], weights = weights_accuracy[0:4])

print("\n###################################################################")
print("Task 3:")

print("\nEnsemble model: unweigted majority voting over 7 models")
voting(list_of_models = models, weights = None)

print("\nEnsemble model: weigted majority voting over 7 models (weights proportional to accuracy)")
voting(list_of_models = models, weights = weights_accuracy)