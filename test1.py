#%%
print("hello")
print("hoi")
print("branch")
print('BRANCH')
print("add lines main branch")
print("test")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# %%

# 1. Lees het CSV-bestand in als DataFrame
# Zorg dat 'datasets.csv' in dezelfde map staat als dit Python-bestand
df = pd.read_csv(r"C:\Users\annad\miniforge3\envs\TM10011\machine learning TM\tm10011-2025-26-example\datasets.csv")


# %%

# 2. Print het aantal datasets
# We tellen de unieke waarden in de kolom 'dataset'
aantal_datasets = df['dataset'].nunique()
print(f"Aantal datasets: {aantal_datasets}")


#%%
# 3. Print de namen van de datasets
# We vragen de lijst met unieke namen op
namen_datasets = df['dataset'].unique()
print("Namen van de datasets:")
print(namen_datasets)



# %%

# de statistieken printen
stats_df = df.groupby('dataset')[['x', 'y']].agg(['count', 'mean', 'var', 'std'])
print("Statistieken per dataset:")
print(stats_df)


# %%

# Violin plots voor X
plt.figure(figsize=(10, 5))
sns.violinplot(data=df, x='dataset', y='x')
plt.title('Verdeling van X-coördinaten per dataset')
plt.show()


#%%
# 4. Violin plots voor Y
plt.figure(figsize=(10, 5))
sns.violinplot(data=df, x='dataset', y='y')
plt.title('Verdeling van Y-coördinaten per dataset')
plt.show()

#%%
# Correlatie tussen x en y per dataset
print("\nCorrelatie tussen x en y:")
print(df.groupby('dataset')[['x', 'y']].corr().iloc[0::2, -1])


#%%
# Covariantie matrix per dataset
print("\nCovariantie matrix:")
print(df.groupby('dataset')[['x', 'y']].cov())


# %%

# Lineaire regressie per dataset
def get_linregress(data):
    # We halen x en y uit de groep
    res = stats.linregress(data['x'], data['y'])
    return pd.Series({
        'slope': res.slope,
        'intercept': res.intercept,
        'r_value': res.rvalue
    })

# We passen deze functie toe op elke groep
linreg_results = df.groupby('dataset').apply(get_linregress)
print("\nLineaire Regressie resultaten:")
print(linreg_results)

# %%


# Scatterplots (met FacetGrid)
g = sns.FacetGrid(df, col="dataset")
g.map_dataframe(sns.scatterplot, x="x", y="y")
g.set_axis_labels("X", "Y")
plt.show()

# Scatterplots mét regressielijn (met lmplot)
# Dit combineert scatterplot + de lijn die we bij stap 3 berekenden
sns.lmplot(data=df, x="x", y="y", col="dataset", ci=None)
plt.show()


# %%


