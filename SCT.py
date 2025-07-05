import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set(style='whitegrid', font_scale=1.2, rc={
    'figure.figsize': (8, 5),
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

df = sns.load_dataset('titanic')
print("\n📄 Basic Info:")
print(df.info())

print("\n📊 Summary Statistics:")
print(df.describe(include='all'))

print("\n❗ Missing Values:")
print(df.isnull().sum())

#Fill missing values and drop the column which is not useful
df = df.drop(columns=['alive', 'embark_town']) 
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['deck'] = df['deck'].fillna(df['deck'].mode()[0])

print("\n✅ Missing Values After Cleaning:")
print(df.isnull().sum())

#Performing EDA : 
#Let’s visualize the key variables to understand the data better.

#Survival Count
plt.figure(facecolor='#f5f5f5')
sns.countplot(data=df, x='survived', hue='survived', palette='Set2', dodge=False, edgecolor='black')
plt.title("Survival Distribution", fontsize=16)
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Survival by Gender
plt.figure(figsize=(8,5), facecolor='#f5f5f5')
sns.countplot(data=df, x='sex', hue='survived', palette='pastel', edgecolor='black')
plt.title("Survival by Gender", fontsize=16)
plt.xlabel("Sex")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='Survived', loc='upper right')
plt.show()

#Age Distribution
sns.histplot(data=df, x='age', kde=True, color='lightcoral', edgecolor='black', bins=30)
plt.title("Age Distribution of Passengers", fontsize=16)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.show()


#Survival  by class
sns.countplot(data=df, x='pclass', hue='survived', palette='Set1', edgecolor='black')
plt.title("Survival by Passenger Class", fontsize=16)
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Survived', loc='upper right')
plt.show()


#Correlation Heatmap
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5, square=True, cbar_kws={"shrink": .8})
plt.title("✔️ Feature Correlation Heatmap", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
