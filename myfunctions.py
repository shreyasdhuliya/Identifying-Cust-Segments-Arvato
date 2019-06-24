import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def row_numbers(data):
    ''' 
    Function takes in DataFrame and prints number of rows
    
    Args:
    -----
    DataFrame
    '''
    print("total number of Rows: ",data.shape[0])
    
def col_numbers(data):
    ''' 
    Function takes in DataFrame and prints number of columns
    
    Args:
    -----
    DataFrame
    '''
    print("Total number of Columns: ",data.shape[1])
    
def column_nan_ratios(data):
    ''' 
    Function takes in DataFrame and displays bar graph of percentage of NaNs per column
    
    Args:
    -----
    DataFrame 
    '''
    nan_per_col_percentage=[]
    nan_per_col_percentage = data.isnull().sum().values/data.shape[0]
    col_names = data.columns
    df = pd.DataFrame({'Percentage missing': nan_per_col_percentage}, index=col_names)
    ax = df.plot.bar(rot=0,figsize=(20,4))
    plt.xticks(rotation=90)
    plt.title('Percentage missing in each column')
    plt.xlabel('Column Name')
    plt.ylabel('Percentage')
    plt.show()

#***********************************************************************************************************************
#-----------------------------INTERVAL FUNCTIONS FOR PRAEGENDE_JUGENDJAHRE---------------------------------------------
#************************************************************************************************************************

def return_inv_DECADE(x):
    '''
    Function returns intervals for column cell values as described in Datadictionary for the column
    PRAEGENDE_JUGENDJAHRE to form a new column DECADE
    
    Args:
    float: value in the cell
    
    Returns:
    Interval variable: for the value in the cell
    
    '''
    #Defining pandas intervals and storing it in 6 variables inv_X0s.
    #inv_40s = pd.Interval(left=40, right=49,closed = 'both')
    #inv_50s = pd.Interval(left=50, right=59,closed = 'both')
    #inv_60s = pd.Interval(left=60, right=69,closed = 'both')
    #inv_70s = pd.Interval(left=70, right=79,closed = 'both')
    #inv_80s = pd.Interval(left=80, right=89,closed = 'both')
    #inv_90s = pd.Interval(left=90, right=99,closed = 'both')

    #storing list of integers which refers to 40s,50s.. 90s, in list_val as defined in Data_dictionary.md
    list_val = [[1.0,2.0],[3.0,4.0],[5.0,6.0,7.0],[8.0,9.0],[10.0,11.0,12.0,13.0],[14.0,15.0]]
    #Storing the pandas intervals in list_inv
    list_inv = [1.0,2.0,3.0,4.0,5.0,6.0]
    
    for i,list_v in enumerate(list_val):
        if x in list_v:
            return(list_inv[i])
    



# ************************************************************************************************************************
# -------------------------------------------------- FUNCTIONS for clean_data(df,df_2) ------------------------------------
# *************************************************************************************************************************

def replace_to_nans(azdias,feat_info):
    ''' 
    Function takes in DataFrame for General Population Data of Germany and
    returns dataframe by replacing missing or unknown to NaNs with help of
    feature summary data.
    
    Args:
    -----
    DataFrame - Genereal population data
    DataFrame - Feature Summary
    '''
    print('Log\nIn Replace to NaNs function\n--------------')
    #converting missing values and space, X and XX as defined as missing or unknown in feature summary for CAMEO_DEUG_2015
    #to np.nan
    azdias.replace(['X','XX','',' '],np.nan,inplace=True)
    #converting columns to float as all missing values are in float
    #azdias = azdias.astype(float, errors = 'ignore')
    
    print('-Replaced X,XX,space to np.nan')
    
    #Parsing Unknown and missing from features DataFrame and storing in a list "miss_unknown_list_f" as float values
    miss_unknown_list = [x.replace("[","").replace("]","").split(',')  for x in feat_info['missing_or_unknown']] 
    
    miss_unknown_list_f = []
    for x in miss_unknown_list:
    #list inside list
        list_in = []
        for mis in x:
            try:
                mis = float(mis)
                list_in.append(mis)
            except:
                mis = np.nan
                list_in.append(mis)
        list_in
        miss_unknown_list_f.append(list_in)
    print('-Completed parsing missing nad unknown')
    
    #Replacing all the missing and unknowns to np.nan
    for col,mis_unkwn in zip(azdias.columns,miss_unknown_list_f):
    
        for miss in mis_unkwn:
            azdias[col].replace(mis_unkwn,np.nan,inplace = True) 
            
    print("-Completed replacing missing and unknown to NaNs")
    return(azdias)
                
def drop_rows_columns(azdias,thresh):
    ''' 
    Function drops outlier columns and rows with NaNs more than threshold
    which is provided as an input
    
    Args:
    -----
    DataFrame - Genereal population data
    int - Threshold, Number of NaNs to be allowed in a row 
    '''
    
    print('\nIn drop_rows_columns function\n------------')
    #threshold given by the user for NaNs in a row
    threshold = thresh
    nan_per_col = azdias.isnull().sum().values
    #percentage calculated by missing data in each column divided by total number of rows in DataFrame
    nan_per_col_percentage = nan_per_col/azdias.shape[0]
    #column_missing_inv(azdias)
    
    #storing outlier columns of missing data of more than 20% in a column
    col_outliers = [col for col,per in zip(azdias.columns,nan_per_col_percentage) if per > .30]
    #print(col_outliers)
    #Dropping outlier columns
    azdias.drop(columns = col_outliers, axis=1,inplace= True, errors = 'ignore')
    print('-Dropped Oulier Columns')
    
    #store the rows counts
    count = azdias.shape[0]
    
    #Dropping rows with NaNs mare than threshold
    azdias.dropna(thresh = len(azdias.columns)-threshold,inplace = True)
    print('-Drop Rows with more than ' + str(threshold) +' NaNs') 
    
    #total number of rows droped
    count = count - azdias.shape[0]
    
    #returning DataFrame
    return(count,azdias)
    
def re_encode(azdias,feat_info,count):
    ''' 
    Fucntion Re-encodes
         - Categorical columns 
         - mixed and ordinal columns by setting dummies, converting to intervals
    Dropping 
        - multi-level categorical columns
        - Ordinal columns which can not be converted to intervals
        - 
    
    Args:
    -----
    DataFrame - Genereal population data
    int - Threshold, Number of NaNs to be allowed in a row 
    '''
    
    print('\nIn re_encode function\n------------')
    #List of Column names and types using feature summary DataFrame
    cols_and_type = []
    for i,col in enumerate(list(feat_info.attribute)):
        cols_and_type.append([col,feat_info.loc[i,'type']])
    print('Categorical Columns -WIP')
    
    #                    ----------------------RE-ENCODE 'OST_WEST_KZ' ----------------------------
    
    #Re-encoding binary 'OST_WEST_KZ' to 'OST_WEST_KZ_W' and 'OST_WEST_KZ_E'
    if 'OST_WEST_KZ' in azdias.columns:
        azdias = pd.get_dummies(azdias, columns=['OST_WEST_KZ'])
    print('   Created Dummies for OST_WEST_KZ')
    
    #                    -------------------Dropping CAtegorical Multilevel Type Columns after  ----------------------------
    #                    -----------------------------Re-encoding OST_WEST_KZ5----------------------------
    
    #Storing Multilevel categorical columns in a list
    drop_col = ['OST_WEST_KZ','CAMEO_DEUG_2015','CAMEO_DEU_2015','AGER_TYP','CJT_GESAMTTYP','FINANZTYP','GFK_URLAUBERTYP','LP_FAMILIE_FEIN','LP_FAMILIE_GROB','LP_STATUS_FEIN','LP_STATUS_GROB','NATIONALITAET_KZ','SHOPPER_TYP','TITEL_KZ','ZABEOTYP','KK_KUNDENTYP','GEBAEUDETYP']
    
    #dropping multilevel columns and 'OST_WEST_KZ' because it is re-encoded as dummy variables
    azdias.drop(columns = drop_col, axis=1,inplace= True, errors = 'ignore')
    print('   Dropped Multilevel Categorical columns and Re-encoded OST_WEST_KZ')
    
    print('Mixed data columns -WIP')
    
    #                    ----------------------RE-ENCODE 'PRAEGENDE_JUGENDJAHRE' ----------------------------
    
    #Column movement storing 1 for Mainstream and 0 for Avantgarde
    azdias['MOVEMENT'] = azdias.apply(lambda x: x.PRAEGENDE_JUGENDJAHRE in [1.0,3.0,5.0,8.0,10.0,12.0,14.0], axis=1).astype(float)
    
    #Decade with intervals using function ------->return_inv_DECADE()
    azdias['DECADE'] = azdias.apply(lambda row: return_inv_DECADE(row.PRAEGENDE_JUGENDJAHRE), axis=1)
    
    print('   Created DECADE AND MOVEMENT for PRAEGENDE_JUGENDJAHRE')
    
     #                    ----------------------RE-ENCODE 'CAMEO_INTL_2015' ----------------------------
    
    #10s column
    azdias['CAMEO_INTL_2015_10s'] = azdias.apply(lambda x: 0 if x.CAMEO_INTL_2015 in [np.nan] else int(int(x.CAMEO_INTL_2015)/10) , axis=1).replace(0,np.nan)
   
    #1s column
    azdias['CAMEO_INTL_2015_1s'] = azdias.apply(lambda x: 0 if x.CAMEO_INTL_2015 in [np.nan] else int(x.CAMEO_INTL_2015)%10 , axis=1).replace(0,np.nan)
    
    print('   Created CAMEO_INTL_2015_10s AND CAMEO_INTL_2015_1s for CAMEO_INTL_2015')
    
    #                    ----------------------Dropping Mixed Type Columns after  ----------------------------
    #                    -------------Re-encoding PRAEGENDE_JUGENDJAHRE and  CAMEO_INTL_2015----------------------------
    
    mixed_col = []
    for col, col_type in cols_and_type:
        if col_type == 'mixed':
            mixed_col.append(col)
            
    azdias.drop(columns =mixed_col,inplace = True,errors = 'ignore')
    print('   Dropped mixed column')
    
    #-------------------------------------------Ordinal Column type------------------------------------------------------
    
    print('Ordinal Data columns -WIP')
    #Finding Columns of type Oridinal
    ordinal_col = []
    for col, col_type in cols_and_type:
        if col_type == 'ordinal':
            ordinal_col.append(col)
            
     #                    ------------Dropping Ordinal Type Columns that can not be Intervals ----------------------------
    #                    ----------------------And are multilevel columns ----------------------------
    list_drop_ordinal = []
    for col in ordinal_col:
        if col not in ['ALTERSKATEGORIE_GROB','WOHNDAUER_2008', 'KONSUMNAEHE','BALLRAUM','EWDICHTE','INNENSTADT','ORTSGR_KLS9']:
            list_drop_ordinal.append(col)
    
    azdias.drop(columns =list_drop_ordinal,inplace = True,errors = 'ignore')
    
    print('  Dropping Non Interval columns')
    

    print('  Dropping remaining ordinal columns') 
    print('Success - Return DataFrame')
    return(count,azdias)

#--------------------------------------------------------------------------------------------------------------------------------------
#*********************************************************PCA PLOT FUCNTIONS**********************************************************
#------------------------------------------------------------------------------------------------------------------------------------
from sklearn.decomposition import PCA

def show_var_for_comp(df,n_com):
    '''
    Plots variance percentage per component and shows Total variance
    
    Args:
    DAtaFrame: Scaled DataFrame 
    Int: Number of components
    
    '''
    
    pca = PCA(n_components = n_com)
    pca.fit_transform(df)
    pca_components = np.arange(1,(len(pca.explained_variance_ratio_)+1))
    plt.bar(pca_components,pca.explained_variance_ratio_*100)
    plt.title('Explained Variance Per Principal Component\nTotal Variance Explained:' + str(pca.explained_variance_ratio_.sum()*100)+'%')
    plt.xlabel("Principal Component")
    plt.ylabel('Variance Explained(%)')
    plt.show()
        
def show_ith_comp_wt(df,pca,i):
    '''
    Plots weights for one component
    
    Args:
    DataFrame: Which was PCA transformed
    PCA: Transformed PCA 
    i: ith component of the PCA
    
    '''
    component = pd.DataFrame(np.round(pca.components_[i-1].reshape(1,len(df.keys())), 4), columns = df.keys())
    fig, ax = plt.subplots(figsize = (14,8))
    component.plot(ax = ax, kind = 'bar')
    ax.set_title('Explained Variance: ' + str(pca.explained_variance_ratio_[i-1]))
    ax.set_ylabel("Feature Weights")
    ax.set_xlabel('Dimension{}'.format(i))
    
    
    
    
    