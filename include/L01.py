import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import os, sys

datapath = os.path.join("../datasets", "lifesat", "")

def set_data_path(dp):
    datapath = dp

def sayHello():
   print("Hello from module...")

def moreDummyFunc(myParam=True):
    if myParam == True: 
        print("Param true :)")
    else:
        print("Param untrue :(")

def reloadMe():
    print("I have been reloaded!")

def prepare_country_stats():
    # Load the data
    oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
    gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",thousands=',',delimiter='\t', encoding='latin1', na_values="n/a")

    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

def plot_country_stats():
    # Prepare the data
    country_stats = prepare_country_stats()

    # Visualize the data
    country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
    plt.show()


def get_full_country_stats():
    oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

    gdp_per_capita = pd.read_csv(datapath+"gdp_per_capita.csv", thousands=',', delimiter='\t',
                                encoding='latin1', na_values="n/a")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)

    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)

    return full_country_stats

def get_sample_data():
    full_country_stats = get_full_country_stats()

    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))

    sample_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
    #missing_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices]
    return sample_data

def plot_sample_data():
    sample_data = get_sample_data()
    sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))

    plt.axis([0, 60000, 0, 10])
    position_text = {
        "Hungary": (5000, 1),
        "Korea": (18000, 1.7),
        "France": (29000, 2.4),
        "Australia": (40000, 3.0),
        "United States": (52000, 3.8),
    }
    for country, pos_text in position_text.items():
        pos_data_x, pos_data_y = sample_data.loc[country]
        country = "U.S." if country == "United States" else country
        plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
                arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
        plt.plot(pos_data_x, pos_data_y, "ro")
    #save_fig('money_happy_scatterplot')
    plt.show()

def plot_sample_data_with_linreg():
    sample_data = get_sample_data()

    t0 = 4.8530528
    t1 = 4.91154459e-05

    sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
    plt.axis([0, 60000, 0, 10])
    M=np.linspace(0, 60000, 1000)
    plt.plot(M, t0 + t1*M, "b")
    plt.text(5000, 3.1, r"$\theta_0 = 4.85$", fontsize=14, color="b")
    plt.text(5000, 2.2, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="b")
    #save_fig('best_fit_model_plot')
    plt.show()