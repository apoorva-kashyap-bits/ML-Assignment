import pandas as pd

# Load data
df = pd.read_csv('data/data.csv')
print(f'Total samples: {len(df)}')

# Get 20% for testing (stratified by diagnosis)
test_df = df.groupby('diagnosis', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42))

# Drop id and diagnosis columns
test_features = test_df.drop(['id', 'diagnosis'], axis=1)

# Save as test.csv
test_features.to_csv('test.csv', index=False)
print(f'✓ test.csv created with {len(test_features)} samples')
print(f'✓ Features: {len(test_features.columns)}')