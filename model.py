import time
from datetime import tzinfo, datetime
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, BayesianRidge, ARDRegression, ElasticNet, LassoLars, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR

def read(csv_file):
	csvf = open(csv_file, 'r')
	fields = ("propertyid","transdate","transvalue","transdate_previous","transvalue_previous","bathroomcnt","bedroomcnt",
		"builtyear","finishedsquarefeet","lotsizesquarefeet","storycnt","latitude","longitude","usecode","censustract","viewtypeid")
	reader = csv.DictReader(csvf, fields)	
	reader.next()

	dataset = []
	for row in reader:
		for attr_name, val in row.iteritems():
			if val == None or val == '' or val == 'NULL':
				val = -1.0
				row[attr_name] = val
		dataset.append(row)
	return dataset

def get_attributes(row, attr, values, name, ptype):
	if row[attr] != -1.0:
		if ptype == 'float':
			values[name].append(float(row[attr]))
		if ptype == 'int':
			values[name].append(int(row[attr]))
		if ptype == 'date':
			values[name].append(float(time.mktime(datetime.strptime(row[attr],'%m/%d/%Y').timetuple())))

def get_values(dataset):
	values = {'transvalue_previous':[], 'bathroomcnt': [], 'bedroomcnt': [],
	'builtyear': [], 'finishedsquarefeet': [], 'lotsizesquarefeet': [], 'storycnt': [],
	'latitude': [], 'longitude': [], 'usecode': [], 'censustract': [], 'viewtypeid': []}

	for row in dataset:
		get_attributes(row, 'transvalue_previous', values, 'transvalue_previous', 'float')
		get_attributes(row, 'bathroomcnt', values, 'bathroomcnt', 'float')
		get_attributes(row, 'bedroomcnt', values, 'bedroomcnt', 'float')
		get_attributes(row, 'builtyear', values, 'builtyear', 'int')
		get_attributes(row, 'finishedsquarefeet', values, 'finishedsquarefeet', 'float')
		get_attributes(row, 'lotsizesquarefeet', values, 'lotsizesquarefeet', 'float')
		get_attributes(row, 'storycnt', values, 'storycnt', 'float')
		get_attributes(row, 'latitude', values, 'latitude', 'float')
		get_attributes(row, 'longitude', values, 'longitude', 'float')
		get_attributes(row, 'usecode', values, 'usecode', 'int')
		get_attributes(row, 'censustract', values, 'censustract', 'float')
		get_attributes(row, 'viewtypeid', values, 'viewtypeid', 'int')

	#for key, val in values.iteritems():
		#print key + " " + str(len(val))
			
	return values

def setup_with_avg_imputation(dataset):
	values = get_values(dataset)

	target = []
	feature_vector = []
	property_ids = []
	dates = []
	map_ = {}
	
	for row in dataset:
		features = []
		i = 0
		for attr_name, val in row.iteritems():
			if attr_name != 'transdate' and attr_name != 'propertyid' and attr_name != 'transdate_previous':
				if attr_name != 'builtyear' and attr_name != 'usecode' and attr_name != 'viewtypeid' and attr_name != 'transvalue':
					imputed_val = float(sum(values[attr_name]))/len(values[attr_name]) if len(values[attr_name]) != 0 else -1.0
					if row[attr_name] != -1.0:
						imputed_val = float(row[attr_name])
					features.append(imputed_val)
					map_[attr_name] = i
					i += 1
				elif attr_name != 'transvalue':
					imputed_val = np.median(values[attr_name]) if len(values[attr_name]) != 0 else 0
					if row[attr_name] != -1.0:
						imputed_val = float(row[attr_name])
					features.append(imputed_val)
					map_[attr_name] = i
					i += 1
				elif attr_name == 'transvalue':
					imputed_val = 0.0
					if row[attr_name] != -1.0:
						imputed_val = float(row[attr_name])
					target.append(imputed_val)

		feature_vector.append(features)
		property_ids.append(row['propertyid'])
		dates.append([row['transdate'], row['transdate_previous']])

	#print str(len(target)) + " " + str(len(feature_vector)) + " " + str(len(property_ids))

	return [feature_vector, target, property_ids, dates, map_]

def prune_dataset(dataset):
	pruned_dataset = []
	for row in dataset:
		r = {}
		for attr_name, val in row.iteritems():
			if attr_name != 'propertyid' and attr_name != 'transvalue' and attr_name != 'transdate':
				if attr_name == 'transdate_previous' and val != -1.0:
					r[attr_name] = float(time.mktime(datetime.strptime(val,'%m/%d/%Y').timetuple()))
				else:
					r[attr_name] = val

		pruned_dataset.append(r)
	return pruned_dataset

def get_avg_knn_neighbors(dataset, target, k, attr):
	pruned_dataset = []
	none_attr_list = []
	pruned_target = {}
	dset = []

	for attr_name, val in target.iteritems():
		if val != -1.0:
			pruned_target[attr_name] = val
		else:
			none_attr_list.append(attr_name);
	target_arr = [float(v) for v in pruned_target.values()]
	for row in dataset:
		if row[attr] != -1.0:
			dset.append(row)
			r = {}
			for attr_name, val in row.iteritems():
				if attr_name not in none_attr_list:
					r[attr_name] = val
			pruned_dataset.append([float(v) for v in r.values()])

	if len(pruned_dataset) < k:
		return 0.0	
	model = NearestNeighbors(n_neighbors=k).fit(pruned_dataset)
	distance, indices = model.kneighbors(target_arr, k)

	avg = 0.0
	for i in indices[0]:
		avg += float(dset[i][attr])
	return avg/k

def setup_with_knn_imputation(dataset, k):
	pruned_dataset = prune_dataset(dataset)

	target = []
	feature_vector = []
	property_ids = []
	dates = []
	map_ = {}	
	
	for row in dataset:
		features = []
		i = 0
		for attr_name, val in row.iteritems():
			if attr_name != 'transdate' and attr_name != 'propertyid' and attr_name != 'transdate_previous':
				if attr_name != 'builtyear' and attr_name != 'usecode' and attr_name != 'viewtypeid' and attr_name != 'transvalue':
					imputed_val = get_avg_knn_neighbors(pruned_dataset, prune_dataset([row])[0], k, attr_name) # avg of 7-NN
					if row[attr_name] != -1.0:
						imputed_val = float(row[attr_name])
					features.append(imputed_val)
					map_[attr_name] = i
					i += 1
				elif attr_name != 'transvalue':
					imputed_val = get_avg_knn_neighbors(pruned_dataset, prune_dataset([row])[0], k, attr_name) # avg of 7-NN
					if row[attr_name] != -1.0:
						imputed_val = float(row[attr_name])
					features.append(imputed_val)
					map_[attr_name] = i
					i += 1
				elif attr_name == 'transvalue':
					imputed_val = 0.0
					if row[attr_name] != -1.0:
						imputed_val = float(row[attr_name])
					target.append(imputed_val)

		feature_vector.append(features)
		property_ids.append(row['propertyid'])
		dates.append([row['transdate'], row['transdate_previous']])

	#print str(len(target)) + " " + str(len(feature_vector)) + " " + str(len(property_ids))

	return [feature_vector, target, property_ids, dates, map_]

def setup(imputed_dataset):
	target = []
	feature_vector = []
	property_ids = []
	dates = []
	map_ = {}
	
	for row in imputed_dataset:
		features = []
		i = 0
		for attr_name, val in row.iteritems():
			if attr_name != 'transdate' and attr_name != 'propertyid' and attr_name != 'transdate_previous':
				if attr_name != 'transvalue':
					features.append(float(val))
					map_[attr_name] = i
					i += 1
				elif attr_name == 'transvalue':
					target.append(float(val))

		feature_vector.append(features)
		property_ids.append(row['propertyid'])
		dates.append([row['transdate'], row['transdate_previous']])

	#print str(len(target)) + " " + str(len(feature_vector)) + " " + str(len(property_ids))

	return [feature_vector, target, property_ids, dates, map_]

def build_apply_linear_regression_model(features_training, target_training, features_test):
	model = LinearRegression()
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def build_apply_ridge_regression_model(features_training, target_training, features_test):
	model = RidgeCV(alphas = [0.1, 1.0, 10, 100])
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]	
	
def build_apply_lasso_regression_model(features_training, target_training, features_test):
	model = LassoCV(alphas = [0.1, 1.0, 10, 100])
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def build_apply_bayesian_regression_model(features_training, target_training, features_test):
	model = BayesianRidge()
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def build_apply_ard_regression_model(features_training, target_training, features_test):
	model = ARDRegression()
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def build_apply_elasticnet_regression_model(features_training, target_training, features_test):
	model = ElasticNet(alpha=0.5, l1_ratio=0.7)
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def build_apply_lassolars_regression_model(features_training, target_training, features_test):
	model = LassoLars(alpha=0.1)
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def build_apply_huber_regression_model(features_training, target_training, features_test):
	model = HuberRegressor()
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]
	
def build_apply_poly_regression_model(features_training, target_training, features_test):
	poly = PolynomialFeatures(degree=6)
	features_training = poly.fit_transform(features_training)
	features_test = poly.fit_transform(features_test)

	model = LinearRegression()
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model, features_training, features_test]

def build_apply_knn_regression_model(features_training, target_training, features_test, n):
	model = KNeighborsRegressor(n_neighbors=n)
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def build_apply_mlp_regression_model(features_training, target_training, features_test):
	model = MLPRegressor()
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def build_apply_rfr_regression_model(features_training, target_training, features_test):
	model = RandomForestRegressor()
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def build_apply_gbr_regression_model(features_training, target_training, features_test):
	model = GradientBoostingRegressor()
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def build_apply_dtr_regression_model(features_training, target_training, features_test):
	model = DecisionTreeRegressor()
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def build_apply_gauss_regression_model(features_training, target_training, features_test):
	model = GaussianProcessRegressor()
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def build_apply_svr_regression_model(features_training, target_training, features_test):
	model = SVR()
	model.fit(features_training, target_training)
	
	predict_training = model.predict(features_training)
	predict_test = model.predict(features_test)

	return [predict_training, predict_test, model]

def write_result(name, imputation, features_training, target_training, predict_training, property_ids_training, predict_test, property_ids_test, model, type_='linear'):
	csvf = open(name + '_model_training_' + imputation + '.csv','w')
	csvt = open(name + '_model_test_' + imputation + '.csv','w')
	fieldsf = ["propertyid", "transvalue", "predict"]
	fieldst = ["propertyid", "transvalue"]

	writerf = csv.DictWriter(csvf, fieldnames=fieldsf)
	writerf.writeheader()

	writert = csv.DictWriter(csvt, fieldnames=fieldst)
	writert.writeheader()
	
	for x in xrange(0,len(predict_training)):
		writerf.writerow({"propertyid": property_ids_training[x], "transvalue": '{:.0f}'.format(target_training[x]), "predict": '{:.0f}'.format(predict_training[x])})

	for x in xrange(0,len(predict_test)):
		writert.writerow({"propertyid": property_ids_test[x], "transvalue": '{:.0f}'.format(predict_test[x])})
	
	traning_result = open(name + '_model_result_' + imputation + '.txt','w')
	traning_result.write(str(model) + '\n\n')
	if type_ == 'linear':
		traning_result.write(str(model.coef_) + '\n\n')
	traning_result.write("Mean squared error: %.2f\n"
      % np.mean((predict_training - target_training) ** 2))
	APE = (abs(predict_training - target_training) / target_training) * 100
	traning_result.write("Mean absolute percentage error: %.2f\n"
      % (np.mean(APE)))
	traning_result.write("Median absolute percentage error: %.2f\n"
      % (np.median(APE)))
	traning_result.write("Percentage of prediction error within 1.0 percent: %.2f\n"
      % (float(sum(i < 1.0 for i in APE)) / len(APE) * 100))
	traning_result.write("Percentage of prediction error within 5.0 percent: %.2f\n"
      % (float(sum(i < 5.0 for i in APE)) / len(APE) * 100))
	traning_result.write("Percentage of prediction error within 10.0 percent: %.2f\n"
      % (float(sum(i < 10.0 for i in APE)) / len(APE) * 100))
	traning_result.write("Percentage of prediction error within 20.0 percent: %.2f\n"
      % (float(sum(i < 20.0 for i in APE)) / len(APE) * 100))
	traning_result.write('Variance score: %.2f\n' % model.score(features_training, target_training))

def write_csv(features, targets, property_ids, dates, maps, name):
	csvf = open('imputed_' + name, 'w')
	fields = ("propertyid","transdate","transvalue","transdate_previous","transvalue_previous","bathroomcnt","bedroomcnt",
		"builtyear","finishedsquarefeet","lotsizesquarefeet","storycnt","latitude","longitude","usecode","censustract","viewtypeid")
	writer = csv.DictWriter(csvf, fieldnames=fields)
	writer.writeheader()

	for x in xrange(0, len(property_ids)):
		writer.writerow({'propertyid': str(property_ids[x]), 'transdate': '' if dates[x][0] == -1.0 else dates[x][0], 'transvalue': '' if targets[x] == -1.0 else str(targets[x]), 
			'transdate_previous': '' if dates[x][1] == -1.0 else dates[x][1], 'transvalue_previous': '' if features[x][maps['transvalue_previous']] == -1.0 else str(features[x][maps['transvalue_previous']]), 
			'bathroomcnt': '' if features[x][maps['bathroomcnt']] == -1.0 else str(features[x][maps['bathroomcnt']]), 'bedroomcnt': '' if features[x][maps['bedroomcnt']] == -1.0 else str(features[x][maps['bedroomcnt']]), 
			'builtyear': '' if features[x][maps['builtyear']] == -1.0 else str(features[x][maps['builtyear']]), 'finishedsquarefeet': '' if features[x][maps['finishedsquarefeet']] == -1.0 else str(features[x][maps['finishedsquarefeet']]), 
			'lotsizesquarefeet': '' if features[x][maps['lotsizesquarefeet']] == -1.0 else str(features[x][maps['lotsizesquarefeet']]), 'storycnt': '' if features[x][maps['storycnt']] == -1.0 else str(features[x][maps['storycnt']]), 
			'latitude': '' if features[x][maps['latitude']] == -1.0 else str(features[x][maps['latitude']]), 'longitude': '' if features[x][maps['longitude']] == -1.0 else str(features[x][maps['longitude']]), 
			'usecode': '' if features[x][maps['usecode']] == -1.0 else str(round(features[x][maps['usecode']])), 'censustract': '' if features[x][maps['censustract']] == -1.0 else str(features[x][maps['censustract']]), 
			'viewtypeid': '' if features[x][maps['viewtypeid']] == -1.0 else str(round(features[x][maps['viewtypeid']]))})

def plot(property_ids_training, target_training, predict_training):
	plt.scatter(property_ids_training, target_training,  color='black')
	plt.plot(property_ids_training, predict_training, color='blue',
		linewidth=3)
	plt.xticks(())
	plt.yticks(())

	plt.show()

def plotter(property_ids_test, predict_test):
	plt.scatter(property_ids_test, predict_test, color='red')
	plt.xticks(())
	plt.yticks(())

	plt.show()

if __name__ == '__main__':
	
	# Imputation
	"""
	csv_file_test = 'test_ZILLOW_CONFIDENTIAL.CSV'
	csv_file_training = 'training_ZILLOW_CONFIDENTIAL.csv'

	test_dataset = read(csv_file_test)
	training_dataset = read(csv_file_training)
	#features_test, target_test, property_ids_test, dates_test, maps_test = setup_with_avg_imputation(test_dataset)
	#features_training, target_training, property_ids_training, dates_training, maps_training = setup_with_avg_imputation(training_dataset)
	features_test, target_test, property_ids_test, dates_test, maps_test = setup_with_knn_imputation(test_dataset, 2)	
	features_training, target_training, property_ids_training, dates_training, maps_training = setup_with_knn_imputation(training_dataset, 2)	

	#write_csv(features_test, target_test, property_ids_test, dates_test, maps_test, 'avg_' + csv_file_test)
	#write_csv(features_training, target_training, property_ids_training, dates_training, maps_training, 'avg_' + csv_file_training)
	write_csv(features_test, target_test, property_ids_test, dates_test, maps_test, 'knn2_' + csv_file_test)
	write_csv(features_training, target_training, property_ids_training, dates_training, maps_training, 'knn2_' + csv_file_training)
	"""

	# Building models
	# KNN3
	csv_file_test = 'imputed_knn3_test_ZILLOW_CONFIDENTIAL.CSV'
	csv_file_training = 'imputed_knn3_training_ZILLOW_CONFIDENTIAL.csv'

	test_dataset = read(csv_file_test)
	training_dataset = read(csv_file_training)

	features_test, target_test, property_ids_test, dates_test, maps_test = setup(test_dataset)	
	features_training, target_training, property_ids_training, dates_training, maps_training = setup(training_dataset)	

	"""
	linear_training_predict, linear_test_predict, linear_model = build_apply_linear_regression_model(features_training, target_training, features_test)
	write_result('linear_regression', 'knn3_imputation', features_training, target_training, linear_training_predict, property_ids_training, 
		linear_test_predict, property_ids_test, linear_model)
	plot(property_ids_training, target_training, linear_training_predict)
	
	ridge_training_predict, ridge_test_predict, ridge_model = build_apply_ridge_regression_model(features_training, target_training, features_test)
	write_result('ridge_regression', 'knn3_imputation', features_training, target_training, ridge_training_predict, property_ids_training, 
		ridge_test_predict, property_ids_test, ridge_model)
	plot(property_ids_training, target_training, ridge_training_predict)
	
	lasso_training_predict, lasso_test_predict, lasso_model = build_apply_lasso_regression_model(features_training, target_training, features_test)
	write_result('lasso_regression', 'knn3_imputation', features_training, target_training, lasso_training_predict, property_ids_training, 
		lasso_test_predict, property_ids_test, lasso_model)
	plot(property_ids_training, target_training, lasso_training_predict)

	bayes_training_predict, bayes_test_predict, bayes_model = build_apply_bayesian_regression_model(features_training, target_training, features_test)
	write_result('bayes_regression', 'knn3_imputation', features_training, target_training, bayes_training_predict, property_ids_training, 
		bayes_test_predict, property_ids_test, bayes_model)
	plot(property_ids_training, target_training, bayes_training_predict)

	ard_training_predict, ard_test_predict, ard_model = build_apply_ard_regression_model(features_training, target_training, features_test)
	write_result('ard_regression', 'knn3_imputation', features_training, target_training, ard_training_predict, property_ids_training, 
		ard_test_predict, property_ids_test, ard_model)
	plot(property_ids_training, target_training, ard_training_predict)

	elasticnet_training_predict, elasticnet_test_predict, elasticnet_model = build_apply_elasticnet_regression_model(features_training, target_training, features_test)
	write_result('elasticnet_regression', 'knn3_imputation', features_training, target_training, elasticnet_training_predict, property_ids_training, 
		elasticnet_test_predict, property_ids_test, elasticnet_model)
	plot(property_ids_training, target_training, elasticnet_training_predict)

	lassolars_training_predict, lassolars_test_predict, lassolars_model = build_apply_lassolars_regression_model(features_training, target_training, features_test)
	write_result('lassolars_regression', 'knn3_imputation', features_training, target_training, lassolars_training_predict, property_ids_training, 
		lassolars_test_predict, property_ids_test, lassolars_model)
	plot(property_ids_training, target_training, lassolars_training_predict)

	huber_training_predict, huber_test_predict, huber_model = build_apply_huber_regression_model(features_training, target_training, features_test)
	write_result('huber_regression', 'knn3_imputation', features_training, target_training, huber_training_predict, property_ids_training, 
		huber_test_predict, property_ids_test, huber_model)
	plot(property_ids_training, target_training, huber_training_predict)

	poly_training_predict, poly_test_predict, poly_model, features_training, features_test = build_apply_poly_regression_model(features_training, target_training, features_test)
	write_result('poly_regression', 'knn3_imputation', features_training, target_training, poly_training_predict, property_ids_training, 
		poly_test_predict, property_ids_test, poly_model)
	plot(property_ids_training, target_training, poly_training_predict)
	"""

	knn_training_predict, knn_test_predict, knn_model = build_apply_knn_regression_model(features_training, target_training, features_test,2)
	write_result('knn2_regression', 'knn3_imputation', features_training, target_training, knn_training_predict, property_ids_training, 
		knn_test_predict, property_ids_test, knn_model,'knn')
	plot(property_ids_training, target_training, knn_training_predict)
	plotter(property_ids_test, knn_test_predict)

	"""
	mlp_training_predict, mlp_test_predict, mlp_model = build_apply_mlp_regression_model(features_training, target_training, features_test)
	write_result('mlp_regression', 'knn3_imputation', features_training, target_training, mlp_training_predict, property_ids_training, 
		mlp_test_predict, property_ids_test, mlp_model,'mlp')
	plot(property_ids_training, target_training, mlp_training_predict)

	rfr_training_predict, rfr_test_predict, rfr_model = build_apply_rfr_regression_model(features_training, target_training, features_test)
	write_result('rfr_regression', 'knn3_imputation', features_training, target_training, rfr_training_predict, property_ids_training, 
		rfr_test_predict, property_ids_test, rfr_model,'rfr')
	plot(property_ids_training, target_training, rfr_training_predict)
	plotter(property_ids_test, rfr_test_predict)

	gbr_training_predict, gbr_test_predict, gbr_model = build_apply_gbr_regression_model(features_training, target_training, features_test)
	write_result('gbr_regression', 'knn3_imputation', features_training, target_training, gbr_training_predict, property_ids_training, 
		gbr_test_predict, property_ids_test, gbr_model,'gbr')
	plot(property_ids_training, target_training, gbr_training_predict)

	dtr_training_predict, dtr_test_predict, dtr_model = build_apply_dtr_regression_model(features_training, target_training, features_test)
	write_result('dtr_regression', 'knn3_imputation', features_training, target_training, dtr_training_predict, property_ids_training, 
		dtr_test_predict, property_ids_test, dtr_model,'dtr')
	plot(property_ids_training, target_training, dtr_training_predict)

	gauss_training_predict, gauss_test_predict, gauss_model = build_apply_gauss_regression_model(features_training, target_training, features_test)
	write_result('gauss_regression', 'knn3_imputation', features_training, target_training, gauss_training_predict, property_ids_training, 
		gauss_test_predict, property_ids_test, gauss_model,'gauss')
	plot(property_ids_training, target_training, gauss_training_predict)
	plotter(property_ids_test, gauss_test_predict)

	svr_training_predict, svr_test_predict, svr_model = build_apply_svr_regression_model(features_training, target_training, features_test)
	write_result('svr_regression', 'knn3_imputation', features_training, target_training, svr_training_predict, property_ids_training, 
		svr_test_predict, property_ids_test, svr_model,'svr')
	plot(property_ids_training, target_training, svr_training_predict)
	plotter(property_ids_test, svr_test_predict)
	"""

	# AVG
	csv_file_test = 'imputed_avg_test_ZILLOW_CONFIDENTIAL.CSV'
	csv_file_training = 'imputed_avg_training_ZILLOW_CONFIDENTIAL.csv'

	test_dataset = read(csv_file_test)
	training_dataset = read(csv_file_training)

	features_test, target_test, property_ids_test, dates_test, maps_test = setup(test_dataset)	
	features_training, target_training, property_ids_training, dates_training, maps_training = setup(training_dataset)	

	"""
	linear_training_predict, linear_test_predict, linear_model = build_apply_linear_regression_model(features_training, target_training, features_test)
	write_result('linear_regression', 'avg_imputation', features_training, target_training, linear_training_predict, property_ids_training, 
		linear_test_predict, property_ids_test, linear_model)
	plot(property_ids_training, target_training, linear_training_predict)
	
	ridge_training_predict, ridge_test_predict, ridge_model = build_apply_ridge_regression_model(features_training, target_training, features_test)
	write_result('ridge_regression', 'avg_imputation', features_training, target_training, ridge_training_predict, property_ids_training, 
		ridge_test_predict, property_ids_test, ridge_model)
	plot(property_ids_training, target_training, ridge_training_predict)
	
	lasso_training_predict, lasso_test_predict, lasso_model = build_apply_lasso_regression_model(features_training, target_training, features_test)
	write_result('lasso_regression', 'avg_imputation', features_training, target_training, lasso_training_predict, property_ids_training, 
		lasso_test_predict, property_ids_test, lasso_model)
	plot(property_ids_training, target_training, lasso_training_predict)	

	bayes_training_predict, bayes_test_predict, bayes_model = build_apply_bayesian_regression_model(features_training, target_training, features_test)
	write_result('bayes_regression', 'avg_imputation', features_training, target_training, bayes_training_predict, property_ids_training, 
		bayes_test_predict, property_ids_test, bayes_model)
	plot(property_ids_training, target_training, bayes_training_predict)

	ard_training_predict, ard_test_predict, ard_model = build_apply_ard_regression_model(features_training, target_training, features_test)
	write_result('ard_regression', 'avg_imputation', features_training, target_training, ard_training_predict, property_ids_training, 
		ard_test_predict, property_ids_test, ard_model)
	plot(property_ids_training, target_training, ard_training_predict)

	elasticnet_training_predict, elasticnet_test_predict, elasticnet_model = build_apply_elasticnet_regression_model(features_training, target_training, features_test)
	write_result('elasticnet_regression', 'avg_imputation', features_training, target_training, elasticnet_training_predict, property_ids_training, 
		elasticnet_test_predict, property_ids_test, elasticnet_model)
	plot(property_ids_training, target_training, elasticnet_training_predict)

	lassolars_training_predict, lassolars_test_predict, lassolars_model = build_apply_lassolars_regression_model(features_training, target_training, features_test)
	write_result('lassolars_regression', 'avg_imputation', features_training, target_training, lassolars_training_predict, property_ids_training, 
		lassolars_test_predict, property_ids_test, lassolars_model)
	plot(property_ids_training, target_training, lassolars_training_predict)

	huber_training_predict, huber_test_predict, huber_model = build_apply_huber_regression_model(features_training, target_training, features_test)
	write_result('huber_regression', 'avg_imputation', features_training, target_training, huber_training_predict, property_ids_training, 
		huber_test_predict, property_ids_test, huber_model)
	plot(property_ids_training, target_training, huber_training_predict)

	poly_training_predict, poly_test_predict, poly_model, features_training, features_test = build_apply_poly_regression_model(features_training, target_training, features_test)
	write_result('poly_regression', 'avg_imputation', features_training, target_training, poly_training_predict, property_ids_training, 
		poly_test_predict, property_ids_test, poly_model)
	plot(property_ids_training, target_training, poly_training_predict)
	"""

	knn_training_predict, knn_test_predict, knn_model = build_apply_knn_regression_model(features_training, target_training, features_test,2)
	write_result('knn2_regression', 'avg_imputation', features_training, target_training, knn_training_predict, property_ids_training, 
		knn_test_predict, property_ids_test, knn_model,'knn')
	plot(property_ids_training, target_training, knn_training_predict)
	plotter(property_ids_test, knn_test_predict)

	"""
	mlp_training_predict, mlp_test_predict, mlp_model = build_apply_mlp_regression_model(features_training, target_training, features_test)
	write_result('mlp_regression', 'avg_imputation', features_training, target_training, mlp_training_predict, property_ids_training, 
		mlp_test_predict, property_ids_test, mlp_model,'mlp')
	plot(property_ids_training, target_training, mlp_training_predict)

	rfr_training_predict, rfr_test_predict, rfr_model = build_apply_rfr_regression_model(features_training, target_training, features_test)
	write_result('rfr_regression', 'avg_imputation', features_training, target_training, rfr_training_predict, property_ids_training, 
		rfr_test_predict, property_ids_test, rfr_model,'rfr')
	plot(property_ids_training, target_training, rfr_training_predict)
	plotter(property_ids_test, rfr_test_predict)

	gbr_training_predict, gbr_test_predict, gbr_model = build_apply_gbr_regression_model(features_training, target_training, features_test)
	write_result('gbr_regression', 'avg_imputation', features_training, target_training, gbr_training_predict, property_ids_training, 
		gbr_test_predict, property_ids_test, gbr_model,'gbr')
	plot(property_ids_training, target_training, gbr_training_predict)

	dtr_training_predict, dtr_test_predict, dtr_model = build_apply_dtr_regression_model(features_training, target_training, features_test)
	write_result('dtr_regression', 'avg_imputation', features_training, target_training, dtr_training_predict, property_ids_training, 
		dtr_test_predict, property_ids_test, dtr_model,'dtr')
	plot(property_ids_training, target_training, dtr_training_predict)

	gauss_training_predict, gauss_test_predict, gauss_model = build_apply_gauss_regression_model(features_training, target_training, features_test)
	write_result('gauss_regression', 'avg_imputation', features_training, target_training, gauss_training_predict, property_ids_training, 
		gauss_test_predict, property_ids_test, gauss_model,'gauss')
	plot(property_ids_training, target_training, gauss_training_predict)
	plotter(property_ids_test, gauss_test_predict)

	svr_training_predict, svr_test_predict, svr_model = build_apply_svr_regression_model(features_training, target_training, features_test)
	write_result('svr_regression', 'avg_imputation', features_training, target_training, svr_training_predict, property_ids_training, 
		svr_test_predict, property_ids_test, svr_model,'svr')
	plot(property_ids_training, target_training, svr_training_predict)
	plotter(property_ids_test, svr_test_predict)
	"""

	# KNN7
	csv_file_test = 'imputed_knn7_test_ZILLOW_CONFIDENTIAL.CSV'
	csv_file_training = 'imputed_knn7_training_ZILLOW_CONFIDENTIAL.csv'

	test_dataset = read(csv_file_test)
	training_dataset = read(csv_file_training)

	features_test, target_test, property_ids_test, dates_test, maps_test = setup(test_dataset)	
	features_training, target_training, property_ids_training, dates_training, maps_training = setup(training_dataset)	

	"""
	linear_training_predict, linear_test_predict, linear_model = build_apply_linear_regression_model(features_training, target_training, features_test)
	write_result('linear_regression', 'knn7_imputation', features_training, target_training, linear_training_predict, property_ids_training, 
		linear_test_predict, property_ids_test, linear_model)
	plot(property_ids_training, target_training, linear_training_predict)
	
	ridge_training_predict, ridge_test_predict, ridge_model = build_apply_ridge_regression_model(features_training, target_training, features_test)
	write_result('ridge_regression', 'knn7_imputation', features_training, target_training, ridge_training_predict, property_ids_training, 
		ridge_test_predict, property_ids_test, ridge_model)
	plot(property_ids_training, target_training, ridge_training_predict)
	
	lasso_training_predict, lasso_test_predict, lasso_model = build_apply_lasso_regression_model(features_training, target_training, features_test)
	write_result('lasso_regression', 'knn7_imputation', features_training, target_training, lasso_training_predict, property_ids_training, 
		lasso_test_predict, property_ids_test, lasso_model)
	plot(property_ids_training, target_training, lasso_training_predict)

	bayes_training_predict, bayes_test_predict, bayes_model = build_apply_bayesian_regression_model(features_training, target_training, features_test)
	write_result('bayes_regression', 'knn7_imputation', features_training, target_training, bayes_training_predict, property_ids_training, 
		bayes_test_predict, property_ids_test, bayes_model)
	plot(property_ids_training, target_training, bayes_training_predict)

	ard_training_predict, ard_test_predict, ard_model = build_apply_ard_regression_model(features_training, target_training, features_test)
	write_result('ard_regression', 'knn7_imputation', features_training, target_training, ard_training_predict, property_ids_training, 
		ard_test_predict, property_ids_test, ard_model)
	plot(property_ids_training, target_training, ard_training_predict)

	elasticnet_training_predict, elasticnet_test_predict, elasticnet_model = build_apply_elasticnet_regression_model(features_training, target_training, features_test)
	write_result('elasticnet_regression', 'knn7_imputation', features_training, target_training, elasticnet_training_predict, property_ids_training, 
		elasticnet_test_predict, property_ids_test, elasticnet_model)
	plot(property_ids_training, target_training, elasticnet_training_predict)

	lassolars_training_predict, lassolars_test_predict, lassolars_model = build_apply_lassolars_regression_model(features_training, target_training, features_test)
	write_result('lassolars_regression', 'knn7_imputation', features_training, target_training, lassolars_training_predict, property_ids_training, 
		lassolars_test_predict, property_ids_test, lassolars_model)
	plot(property_ids_training, target_training, lassolars_training_predict)

	huber_training_predict, huber_test_predict, huber_model = build_apply_huber_regression_model(features_training, target_training, features_test)
	write_result('huber_regression', 'knn7_imputation', features_training, target_training, huber_training_predict, property_ids_training, 
		huber_test_predict, property_ids_test, huber_model)
	plot(property_ids_training, target_training, huber_training_predict)

	poly_training_predict, poly_test_predict, poly_model, features_training, features_test = build_apply_poly_regression_model(features_training, target_training, features_test)
	write_result('poly_regression', 'knn7_imputation', features_training, target_training, poly_training_predict, property_ids_training, 
		poly_test_predict, property_ids_test, poly_model)
	plot(property_ids_training, target_training, poly_training_predict)
	"""

	knn_training_predict, knn_test_predict, knn_model = build_apply_knn_regression_model(features_training, target_training, features_test,2)
	write_result('knn2_regression', 'knn7_imputation', features_training, target_training, knn_training_predict, property_ids_training, 
		knn_test_predict, property_ids_test, knn_model,'knn')
	plot(property_ids_training, target_training, knn_training_predict)
	plotter(property_ids_test, knn_test_predict)

	"""
	mlp_training_predict, mlp_test_predict, mlp_model = build_apply_mlp_regression_model(features_training, target_training, features_test)
	write_result('mlp_regression', 'knn7_imputation', features_training, target_training, mlp_training_predict, property_ids_training, 
		mlp_test_predict, property_ids_test, mlp_model,'mlp')
	plot(property_ids_training, target_training, mlp_training_predict)

	rfr_training_predict, rfr_test_predict, rfr_model = build_apply_rfr_regression_model(features_training, target_training, features_test)
	write_result('rfr_regression', 'knn7_imputation', features_training, target_training, rfr_training_predict, property_ids_training, 
		rfr_test_predict, property_ids_test, rfr_model,'rfr')
	plot(property_ids_training, target_training, rfr_training_predict)
	plotter(property_ids_test, rfr_test_predict)

	gbr_training_predict, gbr_test_predict, gbr_model = build_apply_gbr_regression_model(features_training, target_training, features_test)
	write_result('gbr_regression', 'knn7_imputation', features_training, target_training, gbr_training_predict, property_ids_training, 
		gbr_test_predict, property_ids_test, gbr_model,'gbr')
	plot(property_ids_training, target_training, gbr_training_predict)

	dtr_training_predict, dtr_test_predict, dtr_model = build_apply_dtr_regression_model(features_training, target_training, features_test)
	write_result('dtr_regression', 'knn7_imputation', features_training, target_training, dtr_training_predict, property_ids_training, 
		dtr_test_predict, property_ids_test, dtr_model,'dtr')
	plot(property_ids_training, target_training, dtr_training_predict)

	gauss_training_predict, gauss_test_predict, gauss_model = build_apply_gauss_regression_model(features_training, target_training, features_test)
	write_result('gauss_regression', 'knn7_imputation', features_training, target_training, gauss_training_predict, property_ids_training, 
		gauss_test_predict, property_ids_test, gauss_model,'gauss')
	plot(property_ids_training, target_training, gauss_training_predict)
	plotter(property_ids_test, gauss_test_predict)

	svr_training_predict, svr_test_predict, svr_model = build_apply_svr_regression_model(features_training, target_training, features_test)
	write_result('svr_regression', 'knn7_imputation', features_training, target_training, svr_training_predict, property_ids_training, 
		svr_test_predict, property_ids_test, svr_model,'svr')
	plot(property_ids_training, target_training, svr_training_predict)
	plotter(property_ids_test, svr_test_predict)
	"""

	# KNN3
	csv_file_test = 'imputed_knn2_test_ZILLOW_CONFIDENTIAL.CSV'
	csv_file_training = 'imputed_knn2_training_ZILLOW_CONFIDENTIAL.csv'

	test_dataset = read(csv_file_test)
	training_dataset = read(csv_file_training)

	features_test, target_test, property_ids_test, dates_test, maps_test = setup(test_dataset)	
	features_training, target_training, property_ids_training, dates_training, maps_training = setup(training_dataset)	

	"""
	linear_training_predict, linear_test_predict, linear_model = build_apply_linear_regression_model(features_training, target_training, features_test)
	write_result('linear_regression', 'knn2_imputation', features_training, target_training, linear_training_predict, property_ids_training, 
		linear_test_predict, property_ids_test, linear_model)
	plot(property_ids_training, target_training, linear_training_predict)
	
	ridge_training_predict, ridge_test_predict, ridge_model = build_apply_ridge_regression_model(features_training, target_training, features_test)
	write_result('ridge_regression', 'knn2_imputation', features_training, target_training, ridge_training_predict, property_ids_training, 
		ridge_test_predict, property_ids_test, ridge_model)
	plot(property_ids_training, target_training, ridge_training_predict)
	
	lasso_training_predict, lasso_test_predict, lasso_model = build_apply_lasso_regression_model(features_training, target_training, features_test)
	write_result('lasso_regression', 'knn2_imputation', features_training, target_training, lasso_training_predict, property_ids_training, 
		lasso_test_predict, property_ids_test, lasso_model)
	plot(property_ids_training, target_training, lasso_training_predict)

	bayes_training_predict, bayes_test_predict, bayes_model = build_apply_bayesian_regression_model(features_training, target_training, features_test)
	write_result('bayes_regression', 'knn2_imputation', features_training, target_training, bayes_training_predict, property_ids_training, 
		bayes_test_predict, property_ids_test, bayes_model)
	plot(property_ids_training, target_training, bayes_training_predict)

	ard_training_predict, ard_test_predict, ard_model = build_apply_ard_regression_model(features_training, target_training, features_test)
	write_result('ard_regression', 'knn2_imputation', features_training, target_training, ard_training_predict, property_ids_training, 
		ard_test_predict, property_ids_test, ard_model)
	plot(property_ids_training, target_training, ard_training_predict)

	elasticnet_training_predict, elasticnet_test_predict, elasticnet_model = build_apply_elasticnet_regression_model(features_training, target_training, features_test)
	write_result('elasticnet_regression', 'knn2_imputation', features_training, target_training, elasticnet_training_predict, property_ids_training, 
		elasticnet_test_predict, property_ids_test, elasticnet_model)
	plot(property_ids_training, target_training, elasticnet_training_predict)

	lassolars_training_predict, lassolars_test_predict, lassolars_model = build_apply_lassolars_regression_model(features_training, target_training, features_test)
	write_result('lassolars_regression', 'knn2_imputation', features_training, target_training, lassolars_training_predict, property_ids_training, 
		lassolars_test_predict, property_ids_test, lassolars_model)
	plot(property_ids_training, target_training, lassolars_training_predict)

	huber_training_predict, huber_test_predict, huber_model = build_apply_huber_regression_model(features_training, target_training, features_test)
	write_result('huber_regression', 'knn2_imputation', features_training, target_training, huber_training_predict, property_ids_training, 
		huber_test_predict, property_ids_test, huber_model)
	plot(property_ids_training, target_training, huber_training_predict)

	poly_training_predict, poly_test_predict, poly_model, features_training, features_test = build_apply_poly_regression_model(features_training, target_training, features_test)
	write_result('poly_regression', 'knn2_imputation', features_training, target_training, poly_training_predict, property_ids_training, 
		poly_test_predict, property_ids_test, poly_model)
	plot(property_ids_training, target_training, poly_training_predict)
	"""

	knn_training_predict, knn_test_predict, knn_model = build_apply_knn_regression_model(features_training, target_training, features_test,2)
	write_result('knn2_regression', 'knn2_imputation', features_training, target_training, knn_training_predict, property_ids_training, 
		knn_test_predict, property_ids_test, knn_model,'knn')
	plot(property_ids_training, target_training, knn_training_predict)
	plotter(property_ids_test, knn_test_predict)

	"""
	mlp_training_predict, mlp_test_predict, mlp_model = build_apply_mlp_regression_model(features_training, target_training, features_test)
	write_result('mlp_regression', 'knn2_imputation', features_training, target_training, mlp_training_predict, property_ids_training, 
		mlp_test_predict, property_ids_test, mlp_model,'mlp')
	plot(property_ids_training, target_training, mlp_training_predict)

	rfr_training_predict, rfr_test_predict, rfr_model = build_apply_rfr_regression_model(features_training, target_training, features_test)
	write_result('rfr_regression', 'knn2_imputation', features_training, target_training, rfr_training_predict, property_ids_training, 
		rfr_test_predict, property_ids_test, rfr_model,'rfr')
	plot(property_ids_training, target_training, rfr_training_predict)
	plotter(property_ids_test, rfr_test_predict)

	gbr_training_predict, gbr_test_predict, gbr_model = build_apply_gbr_regression_model(features_training, target_training, features_test)
	write_result('gbr_regression', 'knn2_imputation', features_training, target_training, gbr_training_predict, property_ids_training, 
		gbr_test_predict, property_ids_test, gbr_model,'gbr')
	plot(property_ids_training, target_training, gbr_training_predict)

	dtr_training_predict, dtr_test_predict, dtr_model = build_apply_dtr_regression_model(features_training, target_training, features_test)
	write_result('dtr_regression', 'knn2_imputation', features_training, target_training, dtr_training_predict, property_ids_training, 
		dtr_test_predict, property_ids_test, dtr_model,'dtr')
	plot(property_ids_training, target_training, dtr_training_predict)

	gauss_training_predict, gauss_test_predict, gauss_model = build_apply_gauss_regression_model(features_training, target_training, features_test)
	write_result('gauss_regression', 'knn2_imputation', features_training, target_training, gauss_training_predict, property_ids_training, 
		gauss_test_predict, property_ids_test, gauss_model,'gauss')
	plot(property_ids_training, target_training, gauss_training_predict)
	plotter(property_ids_test, gauss_test_predict)

	svr_training_predict, svr_test_predict, svr_model = build_apply_svr_regression_model(features_training, target_training, features_test)
	write_result('svr_regression', 'knn2_imputation', features_training, target_training, svr_training_predict, property_ids_training, 
		svr_test_predict, property_ids_test, svr_model,'svr')
	plot(property_ids_training, target_training, svr_training_predict)
	plotter(property_ids_test, svr_test_predict)
	"""
