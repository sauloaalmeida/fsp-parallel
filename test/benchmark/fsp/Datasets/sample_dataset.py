import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dsPath = "/home/saulo/workspace/projetos-python/fsp-python-gpu/test/benchmark/fsp/Datasets/PowerSystemAttack_2Classes.csv"

X_y = pd.read_csv(dsPath , header=None).values
X = X_y[:, :-1]
y = X_y[:, -1]


# Holdout 20000 of the data and stratify based on y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5550, stratify=y, random_state=42)

print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)

print("X_test size:", len(X_test))
print("y_test size:", len(y_test))

unique_values_train, counts_train = np.unique(y_train, return_counts=True)
unique_values_test, counts_test = np.unique(y_test, return_counts=True)

# Calculate the proportions
proportions_train = counts_train / len(y_train)
proportions_test = counts_test / len(y_test)

# Display the unique values and their proportions
for value_train, proportion_train in zip(unique_values_train, proportions_train):
    print(f"Value train: {value_train}, Proportion train: {proportion_train:.2f}")

for value_test, proportion_test in zip(unique_values_test, proportions_test):
    print(f"Value_test: {value_test}, Proportion_test: {proportion_test:.2f}")

X_test_df = pd.DataFrame(X_test)  # Name columns as per your dataset

# Convert y_test to a DataFrame
y_test_df = pd.DataFrame(y_test)  # Use a column name that suits your target data

# Concatenate X_test and y_test into a single DataFrame
test_data = pd.concat([X_test_df, y_test_df], axis=1)

# Save the DataFrame to a CSV file
test_data.to_csv('/home/saulo/workspace/projetos-python/fsp-python-gpu/test/benchmark/fsp/Datasets/PowerSystemAttack_2Classes_40000Sample.csv', index=False, header=False)