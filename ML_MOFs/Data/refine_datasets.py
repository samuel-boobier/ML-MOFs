import pandas as pd

train_ls = pd.read_csv("uniqueMOFstrain.txt", header=None, delim_whitespace=True)
train_ls.rename(columns={0: 'code', 1: 'hash'}, inplace=True)
train_ls['code'].replace('_qeq.cif', '.cif', inplace=True, regex=True)


train_df = pd.read_csv("MOF_data_full.csv")
new_train_df = train_df[train_df["MOF"].isin(train_ls["code"])].sort_values(by="MOF").reset_index(drop=True)

new_train_df.to_csv("MOF_data.csv")

test_ls = pd.read_csv("uniqueMOFstest.txt", header=None, delim_whitespace=True)
test_ls.rename(columns={0: 'code', 1: 'hash'}, inplace=True)
test_ls['code'].replace('_qeq.cif', '', inplace=True, regex=True)


test_df = pd.read_csv("MOF_data_test_previous.csv")
new_test_df = test_df[test_df["MOF"].isin(test_ls["code"])].sort_values(by="MOF").reset_index(drop=True)

new_test_df.to_csv("MOF_data_test.csv")
