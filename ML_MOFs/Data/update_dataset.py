import pandas as pd
import numpy as np

template_df = pd.read_csv("MOF_data_previous.csv")
targets_df = pd.read_excel("postMOSAEC_filtereddims.xlsx")
sc_targets_df = pd.read_excel("postMOSAEC_singlecomp.xlsx")
descriptors_df = pd.read_excel("dataframe_withdims.xlsx")

descriptors_df = descriptors_df.rename(columns={descriptors_df.columns[0]: "MOF"})

# inner merge by MOF
inner_merged = pd.merge(targets_df, descriptors_df, how="inner", on=["MOF"]).sort_values(by="MOF").reset_index()
inner_merged = pd.merge(inner_merged, sc_targets_df, how="inner", on=["MOF"]).sort_values(by="MOF").reset_index()

# remove dim
inner_merged = inner_merged.drop(['dim'], axis=1)

# add extra columns
inner_merged["LOG10 TSN"] = inner_merged["TSN"].apply(np.log10)
inner_merged["PLD log10"] = inner_merged["PLD"].apply(np.log10)
inner_merged["LCD log10"] = inner_merged["LCD"].apply(np.log10)
inner_merged["PV (cc/g) log10"] = inner_merged["PV (cc/g)"].apply(np.log10)
inner_merged["K0_CH4 log10"] = inner_merged["K0_CH4"].apply(np.log10)
inner_merged["K0_CO2 log10"] = inner_merged["K0_CO2"].apply(np.log10)
inner_merged["K0_H2S log10"] = inner_merged["K0_H2S"].apply(np.log10)
inner_merged["K0_H2O log10"] = inner_merged["K0_H2O"].apply(np.log10)
inner_merged["DC_CH4 log10"] = inner_merged["DC_CH4"].apply(np.log10)
inner_merged["DC_CO2 log10"] = inner_merged["DC_CO2"].apply(np.log10)
inner_merged["DC_H2S log10"] = inner_merged["DC_H2S"].apply(np.log10)
inner_merged["P_CH4 log10"] = inner_merged["P_CH4"].apply(np.log10)
inner_merged["P_CO2 log10"] = inner_merged["P_CO2"].apply(np.log10)
inner_merged["P_H2S log10"] = inner_merged["P_H2S"].apply(np.log10)

# add TSN class
TSN_class = []
for i in inner_merged["TSN"]:
    if i < 5:
        TSN_class.append("LOW")
    else:
        TSN_class.append("HIGH")

inner_merged["TSN Class"] = TSN_class

# reorder columns
new_dataset = inner_merged[list(template_df)]
new_dataset.to_csv("MOF_data_full.csv")
