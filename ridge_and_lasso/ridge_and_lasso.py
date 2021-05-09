import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso



def ridge_and_lasso():
    save_file('https://homes.cs.washington.edu/~hschafer/cse416/a2/home_data.csv', 'home_data.csv')
    sales = pd.read_csv('home_data.csv')
    sales = sales.sample(frac=0.01, random_state=0) 
    features = [
    'bedrooms', 
    'bathrooms',
    'sqft_living', 
    'sqft_lot', 
    'floors', 
    'waterfront', 
    'view', 
    'condition', 
    'grade',
    'sqft_above',
    'sqft_basement',
    'yr_built', 
    'yr_renovated'
    ]

    all_features = []
    for feature in features:
        square_feature = feature + '_square'
        sqrt_feature = feature + '_sqrt'
    
        sales[square_feature] = sales[feature] ** 2
        sales[sqrt_feature] = sales[feature].apply(sqrt)
    
        all_features.extend([feature, square_feature, sqrt_feature])

    # Standardize each of the features
    for feature in all_features:
        sales[feature] = standardize(sales[feature])

    # Make the price have mean 0 
    mean_price = sales['price'].mean() 
    sales['price'] -= mean_price

    # Split data into train, validation, and test (0.7, 0.1, 0.2)
    train_and_validation, test = train_test_split(sales, test_size=0.2, random_state=6)
    train, validation = train_test_split(train_and_validation, test_size=0.125, random_state=6)

    reg = LinearRegression().fit(train[all_features], train['price'])

    predict = reg.predict(test[all_features])

    rmse = math.sqrt(mean_squared_error(test['price'], predict))
    print(rmse) #384955.7529872539

    ridge_model(train, validation, test, all_features)

    lasso_model(train, validation, test, all_features)

    

def ridge_model(train, validation, test, all_features):
    # Start -5, End 5, Samples to generate 11
    l2 = np.logspace(-5, 5, 11)

    data_ridge = []
    for i in l2:
        model = Ridge(alpha=i, random_state=0).fit(train[all_features], train['price'])
        predict_train_price = model.predict(train[all_features])
        predict_valication_price = model.predict(validation[all_features])
        train_rmse = math.sqrt(mean_squared_error(train['price'], predict_train_price))
        validation_rmse = math.sqrt(mean_squared_error(validation['price'], predict_valication_price))
        data_ridge.append({
            'l2_penalty': i,
            'model': model,
            'train_rmse': train_rmse,
            'validation_rmse': validation_rmse
        })

    ridge_data = pd.DataFrame(data_ridge)
    print(ridge_data)

    f = plt.figure(1)
    # Plot the validation RMSE as a blue line with dots
    plt.plot(ridge_data['l2_penalty'], ridge_data['validation_rmse'], 'b-o', label='Validation')
    # Plot the train RMSE as a red line dots
    plt.plot(ridge_data['l2_penalty'], ridge_data['train_rmse'], 'r-o', label='Train')

    # Make the x-axis log scale for readability
    plt.xscale('log')

    # Label the axes and make a legend
    plt.xlabel('l2_penalty')
    plt.ylabel('RMSE')
    plt.legend()
    fname='ridge_l2_rmse.pdf'
    plt.savefig(fname)

    min_err = ridge_data['validation_rmse'].min()
    best_model_info = ridge_data[ridge_data['validation_rmse'] == min_err]
    best_l2 = best_model_info['l2_penalty']
    best_model = Ridge(alpha = best_l2, random_state = 0).fit(train[all_features], train['price'])

    predict_test_price = best_model.predict(test[all_features])
    test_rmse = math.sqrt(mean_squared_error(test['price'], predict_test_price)) 
    print(test_rmse) #350668.18324124085
    print_coefficients(best_model, all_features)


def lasso_model(train, validation, test, all_features):
    # Start 1, End 7, Samples to generate 7
    l1 = np.logspace(1, 7, num=7)

    data_lasso = []
    for j in l1:
        model = Lasso(alpha=j, random_state=0).fit(train[all_features], train['price'])
        predict_train_price = model.predict(train[all_features])
        predict_validation_price = model.predict(validation[all_features])
        train_rmse = math.sqrt(mean_squared_error(train['price'], predict_train_price))
        validation_rmse = math.sqrt(mean_squared_error(validation['price'], predict_validation_price))
        data_lasso.append({
            'l1_penalty': j,
            'model': model,
            'train_rmse': train_rmse,
            'validation_rmse': validation_rmse
        })

    lasso_data = pd.DataFrame(data_lasso)
    print(lasso_data)

    g = plt.figure(2)
    # Plot the validation RMSE as a blue line with dots
    plt.plot(lasso_data['l1_penalty'], lasso_data['validation_rmse'], 'b-o', label='Validation')

    # Plot the train RMSE as a red line dots
    plt.plot(lasso_data['l1_penalty'], lasso_data['train_rmse'], 'r-o', label='Train')

    # Make the x-axis log scale for readability
    plt.xscale('log')

    # Label the axes and make a legend
    plt.xlabel('l1_penalty')
    plt.ylabel('RMSE')
    plt.legend()

    fname = 'lasso_l1_rmse.pdf'
    plt.savefig(fname)

    index = lasso_data['validation_rmse'].idxmin()
    best_model_info = lasso_data.loc[index]
    best_l1 = best_model_info['l1_penalty']
    print(best_l1) #10000

    best_model = Lasso(alpha=best_l1, random_state=0).fit(train[all_features], train['price'])
    predict_test_price = best_model.predict(test[all_features])
    test_rmse = math.sqrt(mean_squared_error(test['price'], predict_test_price))
    print(test_rmse) #341484.0599829632

    print_coefficients(best_model, all_features)



def standardize(v):
    """
    Takes a single column of a DataFrame and returns a new column 
    with the data standardized - values are centered around the mean with a unit standard deviation (mean 0, std deviation 1)
    """
    std = v.std()
    if std == 0:
        return np.zeros(len(v))
    else:
        return (v - v.mean()) / std



def print_coefficients(model, features):
    """
    This function takes in a model column and a features column. 
    And prints the coefficient along with its feature name.
    """
    feats = list(zip(model.coef_, features))
    print(*feats, sep = "\n")



def save_file(url, file_name):
    r = requests.get(url)
    with open(file_name, 'wb') as f:
        f.write(r.content)


if __name__=='__main__':
    ridge_and_lasso()
