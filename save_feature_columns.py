# save_feature_columns.py

import pandas as pd
import json

df = pd.read_csv('./processed_data/cleaned_train_data.csv')
feature_columns = df.drop(columns=['delay_minutes']).columns.tolist()

with open('./model/feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)

print("✅ Feature column names saved.")
