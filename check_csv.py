import pandas as pd
df = pd.read_csv('claude/areaxdatetime.csv')
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
print(df.head(3).to_string())
