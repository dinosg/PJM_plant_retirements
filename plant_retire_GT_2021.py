#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:47:48 2024
process the retired plants and fit a random forest model, then see how it applies to existing plants.

This version uses 2021 EIA form 923 data, which appears to include all the powerplants - the 2023 version is missing some
@author: dpg
"""
# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
# a quick hack to trim whitespace but without messing up numbers by changing them to nan's
def trim_whitespace(value):
    if isinstance(value, str):
        return value.strip()
    else:
        return value

# Step 2: Load training data into a Pandas DataFrame
# Assuming your DataFrame is named 'train_data'
# You can read data from a file or use any other method to create your DataFrame
# Example: train_data = pd.read_csv('your_training_data.csv')

# Step 3: Preprocess the data if needed
# (e.g., handle missing values, encode categorical variables)

# Step 4: Split the data into features (X) and the target variable (y)

#plants = pd.read_clipboard()  #read plants from excel clipboard
#plants.to_csv('retired_ST_exCA.csv')
#plants.to_csv('retired_GT.csv')
#READ IN THE TRAINING DATA FIRST. TRAIN THE DATA. THEN APPLY IT TO TEST DATA
plants = pd.read_csv('retired_GT.csv')  #for gas turbine plants
#plants = pd.read_csv('retired_ST_exCA.csv')   #for steam turbine plants

target_column = 'plant life'
X = plants[['Heatrate', 'Nameplate Capacity (MW)']]
y = plants[target_column]
y = y.astype('float64')

# Step 5: Split the data into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create and train the Random Forest model
# Create and train the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = rf_model.predict(X_valid)

# Evaluate the model's performance
mse = mean_squared_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize predicted vs actual values
plt.scatter(y_valid, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()


#NOW READ IN TERST DATA - the full EIA dataset for PJM plants. Will need to calculate heatrates for them the same way
# as for the validation & training set. you MIGHT train the model on all the all the data in the training + validation set.
#of course, since none of those plants have retired yet and we're only making predictions, it's anyone's guess what will happen

EIA860fn ='EIA860_2022.xlsx'
EIA923fn = 'EIA923_2021.xlsx'


EIA860_active = pd.read_excel(EIA860fn, sheet_name ='Operable')

isca = EIA860_active["Prime Mover"] == "CA"
isct = EIA860_active["Prime Mover"] == "CT"
iscs = EIA860_active["Prime Mover"] == "CS"
isst = (EIA860_active["Prime Mover"] == "ST" ) & (EIA860_active['Energy Source 1'] != "NUC")   #No Nukes!! cause their PM is steam too...
isgt = EIA860_active["Prime Mover"] == "GT"
#NOW READ IN TERST DATA - the full EIA dataset for PJM plants. Will need to calculate heatrates for them the same way
# as for the validation & training set. you MIGHT train the model on all the all the data in the training + validation set.
#of course, since none of those plants have retired yet and we're only making predictions, it's anyone's guess what will happen

#pickout the steam turbines

EIA860_active = EIA860_active[isgt]

EIA923_active = pd.read_excel(EIA923fn, sheet_name = 'Page 1 Generation and Fuel Data')

#cleanup line breaks
EIA860_active.columns = EIA860_active.columns.str.replace('\n', ' ')
EIA923_active.columns = EIA923_active.columns.str.replace('\n', ' ')

#EIA923_active.columns[columns_to_sum] = EIA923_active.columns[columns_to_sum].str.replace(',','')     # EIA923_active.columns.astype(float)

#extract only those plants in PJM
isPJM = EIA923_active['Balancing Authority Code'] =='PJM'
EIA923_active_pjm = EIA923_active[isPJM]



#now join the tables. BC only the EIA923 table has the plants with PJM, needs to be an inner join and the MWh of generation will get duplicated
#for each unit bc EIA923 isn't by unit, only by fuel. Should sum the generation in EIA923 first by plant to lump gas with oil.
#
#break out by prime mover though to make sure ST units at the same plant don't get lumped together with ccs or gt's
#
#the cc units (CCS, CA, CT) require special care as above for the training 

isca = EIA923_active_pjm['Reported Prime Mover'] == "CA"  #steam turbine component of cc
isct = EIA923_active_pjm['Reported Prime Mover'] == "CT"  #gas turbine component of cc
iscs = EIA923_active_pjm['Reported Prime Mover'] == "CS"  #cc's integrated as a single unit (only a few of these-)
isst = EIA923_active_pjm['Reported Prime Mover'] == "ST" 
isgt = EIA923_active_pjm['Reported Prime Mover'] == "GT" 

EIA923_pjm_gt = EIA923_active_pjm[isgt]

#lets get the MWh generation by fuel
EIA923_pjm_byfuel = EIA923_active_pjm.groupby('Reported Fuel Type Code')['Net Generation (Megawatthours)'].sum().reset_index()
EIA923_pjm_byfuel.to_csv('EIA923PJM_gt_by_fuel.csv')



#compute heatrates
other_cols = [ 'Combined Heat And Power Plant', 
       'Plant Name', 'Operator Name', 'Operator Id', 'Plant State']
cols_to_sum = ['Elec Fuel Consumption MMBtu', 'Net Generation (Megawatthours)']
heatrates = EIA923_pjm_gt.groupby('Plant Id')[ cols_to_sum].sum().reset_index()
heatrates["Heatrate"] = 1000*heatrates['Elec Fuel Consumption MMBtu']/heatrates['Net Generation (Megawatthours)']
isnz = heatrates['Net Generation (Megawatthours)'] > 0
heatrates = heatrates[isnz]   #get rid of the crap resulting in NAs

#produces heatrates by PLANT not by UNIT
#now merge onto EIA860 data
EIA923_860_gt = EIA860_active.merge(heatrates, left_on='Plant Code',right_on = 'Plant Id', how='inner')

#now retrain with all the training data
X = plants[['Heatrate', 'Nameplate Capacity (MW)']]
y = plants[target_column]
y = y.astype('float64')

# Create and train the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

# Make predictions on the test set

X_test = EIA923_860_gt[['Heatrate', 'Nameplate Capacity (MW)' ]]
X_test['Heatrate'] = X_test['Heatrate'].fillna(25000)  #the turbines with NaNs weren't run so their heatrates are crap- fill appropriately
y_pred = rf_model.predict(X_test)
EIA923_860_gt["predicted_life"] = y_pred

#look at uprates. 1st, clean up
EIA923_860_gt["Planned Uprate Year"] = EIA923_860_gt["Planned Uprate Year"].apply(trim_whitespace)
isuprate = EIA923_860_gt["Planned Uprate Year"] != ""
#archive original operating date because we're going to over-write them with uprate year
EIA923_860_gt["original op yr"] = ''
EIA923_860_gt["original op mo"] = ''

EIA923_860_gt.loc[isuprate, "original op yr"] = EIA923_860_gt.loc[isuprate, 'Operating Year']
EIA923_860_gt.loc[isuprate, "original op mo"] = EIA923_860_gt.loc[isuprate, 'Operating Month']

#now copy the uprate date into the operating year & mo date
EIA923_860_gt.loc[isuprate, 'Operating Year'] = EIA923_860_gt.loc[isuprate, 'Planned Uprate Year'] 
EIA923_860_gt.loc[isuprate, 'Operating Month'] = EIA923_860_gt.loc[isuprate, 'Planned Uprate Month'] 

fl = np.floor(EIA923_860_gt["predicted_life"])
EIA923_860_gt["proj retire yr"]  = EIA923_860_gt['Operating Year'] + fl
EIA923_860_gt["proj retire mo"] = EIA923_860_gt['Operating Month'] + np.rint(12*(EIA923_860_gt["predicted_life"] - fl))

toobig = EIA923_860_gt["proj retire mo"] > 12  #more than 12 months no need to adjust
EIA923_860_gt.loc[toobig, "proj retire mo"] = EIA923_860_gt.loc[toobig, "proj retire mo"] - 12
EIA923_860_gt.loc[toobig, "proj retire yr"] = EIA923_860_gt.loc[toobig, "proj retire yr"] + 1




EIA923_860_gt.to_csv("EIA_proj_retirements_gt_2021.csv")
