# Machine Learning Nanodegree
# Identifying Customer Segment with Arvato
## Unsupervised learning

## Installations
1. **Python 3.6**  and libraries
 - [Numpy](http://www.numpy.org/)
 - [Pandas](http://pandas.pydata.org/)
 - [Matplotlib](http://matplotlib.org/)
 - [Seaborn](http://matplotlib.org/)
 - [scikit-learn](http://scikit-learn.org/stable/)
 
2. **Anaconda** 
 - Jupyther Notebook

## Introduction

In this project, Bertelsmann partners AZ Direct and Arvato Financial Solutions have provided two datasets 
one with demographic information about the people of Germany, and one with that same information for customers 
of a mail-order sales company. 
 Performed Unsupervised Learning to form segments/clusters (after dimensionality reduction) to find clusters in the demographics dataset.
  Predicted clusters of the customer dataset using the same clustering technique.
  A comparison of the two cluster distributions was done to see where the strongest customer base for the company.
  
 ## File Discription
 `Identify_Customer_Segments.ipynb`
 This is the main file where the data is loaded,preprocessed and Clustering is carried out
 
 `myfunctions.py`
 Functions used in the notebook.
 
- return_inv_DECADE() - returns intervarls based on PRAEGENDE_JUGENDJAHRE column
- replace_to_nans() - Function takes in DataFrame for General Population Data of Germany and returns dataframe by replacing missing or unknown   to NaNs with help of  feature summary data.
- drop_rows_columns() - Function drops outlier columns and rows with NaNs more than threshold
    which is provided as an input
    
- re_encode() 
  - Fucntion Re-encodes
         - Categorical columns 
         - mixed and ordinal columns by setting dummies, converting to intervals
   - Dropping 
        - multi-level categorical columns
        - Ordinal columns which can not be converted to intervals
- show_var_for_comp()- Plots variance percentage per component and shows Total variance
- show_ith_comp_wt() - Plots weights for one principle component

## Data
- Udacity_AZDIAS_Subset.csv: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
- Udacity_CUSTOMERS_Subset.csv: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
- Data_Dictionary.md: Detailed information file about the features in the provided datasets.
- AZDIAS_Feature_Summary.csv: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
     

