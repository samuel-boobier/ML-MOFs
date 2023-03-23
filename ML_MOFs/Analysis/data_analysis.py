import pandas as pd
from numpy import log10, floor
from analysis_methods import data_analysis_table, range_subplot, corr_graph, plot

pd.set_option('display.precision', 2)
# Explore the ranges and correlation of the initial descriptor set

# load data
df_data = pd.read_csv("../Data/MOF_data.csv")
targets = ["CO2 loading (mol/kg)", "CH4 loading (mol/kg)", "SC CO2 loading (mol/kg)", "SC CH4 loading (mol/kg)", "TSN",
           "LOG10 TSN"]
initial_descriptors = ["PLD", "LCD", "Density (g/cc)", "VSA (m2/cc)", "GSA (m2/g)", "VF", "PV (cc/g)", "K0_CH4",
                       "K0_CO2", "K0_H2S", "K0_H2O", "DC_CH4", "DC_CO2", "DC_H2S", "P_CH4", "P_CO2", "P_H2S", "Qst_CH4",
                       "Qst_CO2", "Qst_H2S", "Qst_H2O"]

log10_descriptors = ["PLD log10", "LCD log10", "PV (cc/g) log10", "K0_CH4 log10", "K0_CO2 log10", "K0_H2S log10",
                     "K0_H2O log10", "DC_CH4 log10", "DC_CO2 log10", "DC_H2S log10", "P_CH4 log10", "P_CO2 log10",
                     "P_H2S log10"]
final_descriptors = ["PLD log10", "LCD log10", "Density (g/cc)", "VSA (m2/cc)", "VF", "DC_CH4 log10",
                     "DC_CO2 log10", "DC_H2S log10", "Qst_CH4", "Qst_CO2", "Qst_H2S", "Qst_H2O"]
target_descriptors = df_data[targets + initial_descriptors + log10_descriptors]

# Descriptor and target stats
descriptor_stats = data_analysis_table(target_descriptors)
descriptor_stats.to_csv("../Results/Analysis_results/dataset_analysis.csv")
print(descriptor_stats.to_latex(index=False))

# target ranges
df_targets = df_data[targets]
target_chunks = [targets[i:i+4] for i in range(0, len(targets), 4)]
for i in range(len(target_chunks)):
    range_subplot(df_targets[target_chunks[i]], "target_ranges_" + str(i))

# Initial descriptor ranges
df_descriptors = target_descriptors[initial_descriptors]
descriptors = list(df_descriptors)
desc_chunks = [descriptors[i:i+4] for i in range(0, len(descriptors), 4)]
for i in range(len(desc_chunks)):
    range_subplot(df_descriptors[desc_chunks[i]], "initial_descriptor_ranges_" + str(i))

# Initial descriptor ranges after log10 transformations
# PLD, LCD, PV, K0s, DCs, and Ps changed to log10
df_descriptors_log10 = df_data[log10_descriptors]
descriptors = list(log10_descriptors)
desc_chunks = [descriptors[i:i+4] for i in range(0, len(descriptors), 4)]
for i in range(len(desc_chunks)):
    range_subplot(df_descriptors_log10[desc_chunks[i]], "log10_descriptor_ranges_" + str(i))

# Initial descriptor correlation Analysis
descriptor_corr = corr_graph(df_data[initial_descriptors + log10_descriptors], "Correlation Initial Descriptors",
                             "corr_plot")
descriptor_corr.to_csv("../Results/Analysis_results/descriptor_correlation.csv")
pd.set_option('display.precision', 4)
print(descriptor_corr.to_latex(index=False))

# descriptor plots
plot("PV (cc/g) log10", "VF", df_data, "PV_VF")
plot("GSA (m2/g)", "VF", df_data, "GSA_VF")
plot("K0_CH4 log10", "Qst_CH4", df_data, "K0_Qst_CH4")
plot("K0_CO2 log10", "Qst_CO2", df_data, "K0_Qst_CO2")
plot("K0_H2O log10", "Qst_H2O", df_data, "K0_Qst_H2O")
plot("K0_H2S log10", "Qst_H2S", df_data, "K0_Qst_H2S")

# Final descriptors
# PV (cc/g) log10 removed due to correlation with VF
# GSA (m2/g) removed due to similarity with VF
# K0s removed due to similarity to Qsts
# Ps removed as deemed less relevant
