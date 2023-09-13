<h1>ML for MOF Property Prediction</h1>
<h2>Getting Started</h2>
The code is written in Python 3.8. We recommend using Anaconda to build a virtual environment.
Details of how to download Anaconda can be found here:
<br><br>
<a href="https://www.anaconda.com/">https://www.anaconda.com/</a>
<br><br>
After cloning the repository, create a virtual environment by running the following commands in terminal or Anaconda 
prompt.
<br>

`conda create --name ML_MOFs --file requirements.txt`
<br>
`conda activate ML_MOFs`
<br>
<br>
The kaleido package is required to save graphs. Install this using pip.
<br>

`pip install -U kaleido`
<h2>Datasets</h2>
All the data used in this study can be found in ML_MOFs/Data/
<br><br>
MOF_data.csv - Target and descriptor values for the dataset used for initial 10-fold cross validation
<br>
MOF_data_test.csv - Target and descriptor values for the unseen test set

<h2>Dataset Analysis</h2>
To perform an analysis of the dataset run ML_MOFs/Analysis/data_analysis.py.
<br><br>
Basic statistics and pairwise descriptor correction are saved to ML_MOFs/Results/Analysis_results/.
<br><br>
Histograms of target and descriptor ranges are saved to ML_MOFs/Graphs/Analysis_graphs/.

<h2>Machine Learning</h2>
For 10-fold cross validation on the full dataset, run ML_MOFs/ML/ML_main.py. Predictions are saved to 
ML_MOFs/Results/ML_results/Classification and ML_MOFs/Results/ML_results/Regression. 
<br><br>
Further analysis of the models produced can be generated by running ML_MOFs\ML\classification_analysis.py and
ML_MOFs\ML\regression_analysis.py.


<h2>Additional Figures</h2>
Additional figures are generated by running ML_MOFs/figures.py and are saved in ML_MOFs/Graphs/Figures.

<h2>Running the model for your own training/test sets</h2>
Run `ML_MOFs/ML/test_ML.py` hcanginging lines 84 and 85 to the location of your training and test sets respectively.
<br><br>
Note: you will need to calculate the descriptors as detailed in our publication prior to machine learning, using our datasets as a template for column names.
