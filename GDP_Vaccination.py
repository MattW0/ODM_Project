import seaborn as sns
import plotly.express as px
import cvxpy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error as mae
from sklearn.datasets import load_diabetes

# Hyperparameter:
LAMBDA = np.logspace(-5, -1, 50, base=10)


def get_mae(theta, X, y):
    mae = np.average(np.abs(y - X @ theta.T), axis=0)
    return mae


def load_vaccination_data(plot=False):

    dfvac = pd.read_csv("Data/country_vaccinations.csv")
    dfvac = dfvac.drop_duplicates('country', keep='last')  # keep unique record by country

    dfpo = pd.read_csv("Data/population_by_country_2020.csv")

    df_vacc = pd.merge(dfvac, dfpo, how='right', left_on='country', right_on='Country (or dependency)')
    df_vacc = df_vacc.drop(
        ['total_vaccinations', 'daily_vaccinations_raw', 'daily_vaccinations', 'total_vaccinations_per_hundred',
         'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred', 'daily_vaccinations_per_million',
         'source_name', 'source_website', 'Yearly Change', 'Net Change', 'Density (P/Km²)', 'Land Area (Km²)',
         'Migrants (net)', 'Fert. Rate', 'Med. Age', 'Urban Pop %', 'World Share', 'vaccines',
         'Country (or dependency)'], axis=1)

    # deleting empty records
    df_vacc = df_vacc[df_vacc['people_fully_vaccinated'].notna()]

    # percentage of the vaccinated population
    df_vacc['% Vaccinated'] = df_vacc['people_fully_vaccinated'] * 100 / df_vacc['Population (2020)']

    # order alphabetically
    df_vacc = df_vacc.sort_values('country')

    if plot:
        df_vacc_temp = df_vacc.sort_values(by=['% vaccinated population'], ascending=True)

        plt.figure(figsize=(18, 40))
        ax = sns.barplot(x='% Vaccinated', y='country', data=df_vacc_temp)
        ax.set_xlabel("Percentage of Vaccinated Population by Country")
        plt.show()

    return df_vacc


def load_gdp_data(plot=False):
    df_gdp = pd.read_csv("Data/GDP.csv")

    years = [str(i) for i in range(1990, 2018)]
    years.append('2019')
    df_gdp = df_gdp.drop(years, axis=1)
    df_gdp = df_gdp[df_gdp['2018'].notna()]

    df_gdp.rename(columns={'Country ': 'Country', '2018': 'GDP per capita (2018)'}, inplace=True)

    if plot:
        df_gdp_temp = df_gdp.sort_values(by=['GDP per capita (2018)'], ascending=True)

        plt.figure(figsize=(18, 40))
        ax = sns.barplot(x='GDP per capita (2018)', y='Country', data=df_gdp_temp)
        ax.set_xlabel("Percentage of Vaccinated Population by Country")
        plt.show()

    return df_gdp


def merge_scatter_dfs(df_vacc, df_gdp, plot=False):
    df = pd.merge(df_vacc, df_gdp, how='outer', left_on='iso_code', right_on='Country Code')

    # Cleaning
    df = df[df['Country Code'].notna()]
    df = df[df['country'].notna()]
    df = df.drop(['country', 'iso_code', 'date'], axis=1)

    # Reordering
    countries = df[['Country Code', 'Country']]
    df = df.drop(columns=['Country Code', 'Country'])
    df.insert(loc=0, column='Country Code', value=countries['Country Code'])
    df.insert(loc=1, column='Country', value=countries['Country'])

    if plot:
        fig = px.scatter(df, x='GDP per capita (2018)', y='% Vaccinated', color='Country')
        fig.show()

    return df


def load_dataset():
    diabetes = load_diabetes()
    X, X_test, Y, Y_test = train_test_split(diabetes['data'],
                                            np.expand_dims(diabetes['target'], 1),
                                            test_size=0.25, random_state=0)
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)  # Bias column:
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)

    return X, X_test, Y, Y_test


# LP solver:
def lp_solver(X, Y, lambda_, k):
    d = 1 # X.shape[1]
    N = X.shape[0]
    beta = cp.Variable((N, 1))
    b = cp.Variable((d, 1))
    alpha = cp.Variable((1, 1))
    theta = cp.Variable((1, d))

    objective = cp.Minimize(k * alpha + cp.sum(beta) + lambda_ * cp.sum(b))
    constraints = [
        alpha + beta >= 1 / k * (Y - X @ theta.T),
        alpha + beta >= -1 / k * (Y - X @ theta.T),
        beta >= 0,
        -b <= theta,
        theta <= b
    ]

    LP = cp.Problem(objective, constraints)
    LP.solve()
    optimal_theta = theta.value
    optimal_value = LP.value
    dual_value = LP.constraints[0].dual_value

    return optimal_theta, optimal_value, dual_value


def main():
    df_vacc = load_vaccination_data()
    df_gdp = load_gdp_data()

    df = merge_scatter_dfs(df_vacc, df_gdp)

    TRAIN_VALIDATION_RATIO = 0.8
    train_size = math.floor(df.shape[0] * TRAIN_VALIDATION_RATIO)

    df_train = df.loc[0:train_size]
    df_validate = df.loc[train_size+1:]

    X = df_train['GDP per capita (2018)']
    Y = df_train['% Vaccinated']
    X_test = df_validate['GDP per capita (2018)']
    Y_test = df_validate['% Vaccinated']

    # Cross-validation:
    train, validation, thetas, optimal = [], [], [], []
    for l in LAMBDA:
        optimal_theta, optimal_value, dual_value = lp_solver(X, Y, l, math.floor(0.75 * X.shape[0]))
        thetas.append(optimal_theta)
        optimal.append(optimal_value)
        train.append(get_mae(optimal_theta, X, Y))
        validation.append(get_mae(optimal_theta, X_test, Y_test))

    best_lambda = LAMBDA[np.argmin(validation)]
    best_theta = thetas[np.argmin(validation)]
    best_value = optimal[np.argmin(validation)]

    print('---Optimal values--')
    print(f'Lambda: {best_lambda}')
    print(f'Optimal value: {best_value}')
    print(f'Theta:\n {best_theta}')


main()
