import pandas as pd

# Charger le dataset initial
file_path = 'Dataset_Thuy (1).csv'
df = pd.read_csv(file_path)

# Vérifier la distribution initiale des classes
target = df.iloc[:, -1]
class_distribution = target.value_counts()

print("Distribution initiale des classes :")
print(class_distribution)
