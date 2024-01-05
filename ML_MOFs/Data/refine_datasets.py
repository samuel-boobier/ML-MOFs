import pandas as pd

train_ls = pd.read_csv("uniqueMOFstrain.txt", header=None, delim_whitespace=True)
train_ls.rename(columns={0: 'code', 1: 'hash'}, inplace=True)
train_ls['code'].replace('_qeq.cif', '.cif', inplace=True, regex=True)

print(train_ls)

train_df = pd.read_csv("MOF_data.csv")

new_train_df = train_df[train_df["MOF"].isin(train_ls["code"])]

print(new_train_df)


test_ls = pd.read_csv("uniqueMOFstest.txt", header=None, delim_whitespace=True)
test_ls.rename(columns={0: 'code', 1: 'hash'}, inplace=True)
test_ls['code'].replace('_qeq.cif', '', inplace=True, regex=True)

print(test_ls)

test_df = pd.read_csv("MOF_data_test.csv")
print(test_df)
new_test_df = test_df[test_df["MOF"].isin(test_ls["code"])]

print(new_test_df)
