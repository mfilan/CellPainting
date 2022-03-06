import json

import pandas as pd

df = pd.read_csv("data.csv")[['compound_name', 'compound_label']].drop_duplicates()
with open("label_mapping.json", 'w') as fp:
    json.dump(df.to_dict('records'), fp)