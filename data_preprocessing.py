# Data Preprocessing
# Step 1. Import pre-libraries ------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
  
# Step 2. Import dataset ------------------------------------------------------
dataset = pd.read_csv('Data\data_pre1.csv')
# -----------------------------------------------------------------------------    

# Step 3. Matrix of features --------------------------------------------------
X = dataset.loc[:, ['Country', 'Age', 'Salary']]
Y = dataset.iloc[:,-1:]
# -----------------------------------------------------------------------------

# Step 4. Missing data management ---------------------------------------------
# Check for missing value
# Check columnwise
X.isnull().sum()
# Find NaN percentange columnwise
NAN = [(clm_name, dataset[clm_name].isna().mean()*100) for clm_name in dataset]
NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])
# If percentage is greater than 50 we are going to drop those columns
# Check threshold crossing column names
NAN[NAN['percentage'] > 50]
# Drop using drop methods from original dataset

# Using SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
X.loc[:, ['Age', 'Salary']] = imputer.fit_transform(X.loc[:, ['Age', 'Salary']])
# -----------------------------------------------------------------------------   
 
# Step 5. Categorical data management -----------------------------------------
# Using LabelEncoder and OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# One for independent variable
clm_x = ColumnTransformer([("Combine",
                            OneHotEncoder(),[0])], 
                            remainder="passthrough")
X = clm_x.fit_transform(X)

# Using get_dummies
'''
We can also use get_dummies()
pd.get_dummies(data = dataset)

We can store this result.
If we want to keep the old categorical column and add new dummies columns we can do that
using .concat()

s1 = pd.get_dummies(data = dataset)
pd.concat([dataset, s1, axis = 1])
'''
# One for dependent variable here we need only Label Encoder
from sklearn.preprocessing import LabelEncoder
labelencoderY = LabelEncoder()
Y = labelencoderY.fit_transform(Y.values.ravel())
# -----------------------------------------------------------------------------        

# Step 6. Splitting of dataset -----
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
# -----------------------------------------------------------------------------

# 7. Feature scaling ----------------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
# -----------------------------------------------------------------------------

# Algorithm section -----------------------------------------------------------

# -----------------------------------------------------------------------------

# Accuracy finder section -----------------------------------------------------

# -----------------------------------------------------------------------------

# Visualization section -------------------------------------------------------

# -----------------------------------------------------------------------------

# Model saving section --------------------------------------------------------

# -----------------------------------------------------------------------------


