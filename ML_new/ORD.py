import pandas as pd

data = pd.read_csv("..\\ML_MOFs\\Data\\MOF_data.csv")
new_data = pd.read_csv("..\\ML_MOFs\\Data\\ALLabsoluteloading_postMOSAEC_singlecomp.csv")

new_data = new_data[new_data["MOF"].isin(data["MOF"])]
new_data.to_csv("..\\ML_MOFs\\Data\\temp.csv")
