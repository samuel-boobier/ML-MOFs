import pandas as pd
from collections import Counter

data = pd.read_csv("MOF_data.csv")

name = data["MOF"].tolist()

name = [x[:6] for x in name]
print(len(name))
print(len(set(name)))

cnt = Counter()
for n in name:
    cnt[n] += 1

df = pd.DataFrame.from_dict(cnt, orient='index').reset_index()
df = df.rename(columns={'index': 'Name', 0: 'Count'})
df = df.sort_values(by="Count", ascending=False).reset_index()

print(len(df[df["Count"] > 1]))
print(df)
