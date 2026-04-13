import pandas as pd
df = pd.read_csv('GR011_switch_labels.csv')
print(f'Current sessions: {len(df)}')
df = df[df['day_index'] <= 80]
df.to_csv('GR011_switch_labels.csv', index=False)
print(f'Reset to: {len(df)} sessions')
print(f'Last day: {df.day_index.iloc[-1]}, date: {df.date.iloc[-1]}')


