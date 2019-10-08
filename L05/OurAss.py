# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:51:20 2019

@author: Jacob Hausted
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd
import math

# Load data - vægt data (kvinder/mænd)
data = pd.read_csv('C:/Users/Jacob Hausted/ML/Lektioner/Uge5_files/housing.csv', sep=',',header=0)

median_income = data['median_income']

mu = np.mean(median_income)
sigma = np.std(median_income)
sigma2 = np.var(median_income)
median = np.median(median_income)

myMax = np.max(median_income)
myMin = np.min(median_income)

xarr = np.linspace(myMax, myMin, 500)

fig, ax = plt.subplots(1, 1, figsize=[8,8])

ax.hist(median_income,bins=100, density=True) # Normalises the histogram
ax.plot(xarr, norm.pdf(xarr, mu, sigma))

print("Mean: ", mu, " and Median: ", median, " with the difference: ", math.fabs(mu-median))

ax.axvline(mu, color='b', label = "mean")
ax.axvline(median, color='r', label="median")
ax.legend()

#%%
# Check if there is a corelation between median income and median house value
medIncAndHVal = data[['median_income','median_house_value']]
corrcoef = np.corrcoef(medIncAndHVal.T) # obs: rækker=variable, kolonner=samples (modsat normalt..)
plt.scatter(medIncAndHVal['median_income'], medIncAndHVal['median_house_value'], s=1)
plt.title(corrcoef[1,0])
plt.xlabel('Med Income')
plt.ylabel('Med H Value')

print("Fifth Percentile of med H val: ", np.percentile(data['median_house_value'], 5))
print("Ninety-fith Percentile of H val: ", np.percentile(data['median_house_value'], 95))
print("With max: ", np.max(data['median_house_value']), " and Min: ", np.min(data['median_house_value']))

#%%
median_HouseVal = data['median_house_value']

mu = np.mean(median_HouseVal)
sigma = np.std(median_HouseVal)
sigma2 = np.var(median_HouseVal)
median = np.median(median_HouseVal)

myMax = np.max(median_HouseVal)
myMin = np.min(median_HouseVal)

xarr = np.linspace(myMax, myMin, 500)

fig, ax = plt.subplots(1, 1, figsize=[8,8])

ax.hist(median_HouseVal,bins=100, density=True) # Normalises the histogram
ax.plot(xarr, norm.pdf(xarr, mu, sigma))

print("Mean: ", mu, " and Median: ", median, " with the difference: ", math.fabs(mu-median))

ax.axvline(mu, color='b', label = "mean")
ax.axvline(median, color='r', label="median")
ax.legend()

