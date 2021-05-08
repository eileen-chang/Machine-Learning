import requests
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def house_prices():
    save_file('https://homes.cs.washington.edu/~hschafer/cse416/a1/home_data.csv', 'home_data.csv')
    sales = pd.read_csv('home_data.csv')

    # random_state makes everything deterministic
    train_data, test_data = train_test_split(sales, test_size=0.2, random_state=0)

    # assume sqrt_living has a relationship with pricing
    plt.scatter(train_data['sqft_living'], train_data['price'], marker='+', label='Train')
    plt.scatter(test_data['sqft_living'], test_data['price'], marker='.', label='Test') 
    plt.legend()
    plt.xlabel('Sqft Living')
    plt.ylabel('Price')
    fname='price_sqft_living_plot.pdf'
    plt.savefig(fname)

    basic_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
    advanced_features = basic_features + [
    'condition',      # condition of the house
    'grade',          # measure of qality of construction
    'waterfront',     # waterfront property 
    'view',           # type of view
    'sqft_above',     # square feet above ground
    'sqft_basement',  # square feet in basementab
    'yr_built',       # the year built
    'yr_renovated',   # the year renovated
    'lat',            # the longitude of the parcel
    'long',           # the latitide of the parcel
    'sqft_living15',  # average sq.ft. of 15 nearest neighbors 
    'sqft_lot15',     # average lot size of 15 nearest neighbors 
    ]

    reg = LinearRegression().fit(train_data[basic_features], train_data['price'])
    reg_adv = LinearRegression().fit(train_data[advanced_features], train_data['price'])

    predict_train = reg.predict(train_data[basic_features])
    predict_train_adv = reg_adv.predict(train_data[advanced_features])
    predict_test = reg.predict(test_data[basic_features])
    predict_test_adv = reg_adv.predict(test_data[advanced_features])

    rmse_train_basic = math.sqrt(mean_squared_error(train_data['price'], predict_train))
    rmse_train_adv = math.sqrt(mean_squared_error(train_data['price'], predict_train_adv))
    rmse_test_basic = math.sqrt(mean_squared_error(test_data['price'], predict_test))
    rmse_test_adv = math.sqrt(mean_squared_error(test_data['price'], predict_test_adv))

    print(rmse_train_basic) #258524.68484833508
    print(rmse_train_adv) #203805.41055523997
    print(rmse_test_basic) #244004.77443104176
    print(rmse_test_adv) #190473.37570967825




def save_file(url, file_name):
    r = requests.get(url)
    with open(file_name, 'wb') as f:
        f.write(r.content)


if __name__=='__main__':
    house_prices()




