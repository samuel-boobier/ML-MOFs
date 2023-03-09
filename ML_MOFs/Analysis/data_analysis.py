import pandas as pd
from analysis_methods import data_analysis_table, range_subplot

# Explore the ranges and correlation of the initial descriptor set

# load data
df_data = pd.read_csv("../Data/MOF_data.csv")
target_descriptors = df_data.drop(columns=["MOF", "CO2 error (mol/kg)", "CH4 error (mol/kg)", "Selectivity (CO2)",
                                           "Selectivity error", "logS error", "TSN error", "logS", "TSN Class",
                                           "Maximum Dimensions"])

# Descriptor and target stats
descriptor_stats = data_analysis_table(target_descriptors)
descriptor_stats.to_csv("../Results/Analysis_results/dataset_analysis.csv")

# target ranges
targets = df_data[["CO2 loading (mol/kg)", "CH4 loading (mol/kg)", "TSN", "LOG10 TSN"]]
range_subplot(targets, "target_ranges")

# Initial descriptor ranges

# Initial descriptor ranges after log10 transformations

# Initial descriptor correlation Analysis

# Final descriptors
