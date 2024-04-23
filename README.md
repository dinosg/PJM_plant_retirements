# PJM_plant_retirements
scikit-learn random forest regression to model PJM fossil plant retirements
This archive contains code for modeling fossil plant retirements in PJM (and can be adapted to model retirements elsewhere since it draws on historical records from EIA forms 860 and 923 covering the entire US)

EIA form 860 has information on plant sizes and for retired plants, retirement dates. 

EIA form 923 contains annual (and monthly) power generation & fuel consumption, from which heatrates are inferred.

Historical EIA form 923 (from 2007 or so) has been processed to infer heatrates for plants no longer in operation (and thus not included in current EIA form 923, although are included in current EIA form 860)

There are three separate python scripts:
- plant_retire_cc_ca2021.py     for estimating plant lives for combined cycles (CC, and CA = single shaft combined cycles)

- plant_retire_GT_2021.py       " gas turbines (simple cycles)

- plant_retire_ST_2021.py       " steam turbines (which include both coal and a few gas units)


  The scripts all start out by reading in data on plants that have been retired, then use scikit-learn RandomForestRegressor to model.

  Validation data is segregated and not used for fitting so the model is fit tested then tested on the validation data.

  The 2 model variables are plant size and heatrate. For the combined cycles, a third variable is whether or not a unit is a gas turbine or steam turbine subunit (usually the steam turbines last longer).

  Then in the test data stage, all the known retirement data is fit for the model, then applied to operating plants to predict plant lives.
   
The EIA Form 923 model doesn't break each plant down by exact unit, only by unit type so it's assumed the calculcated heatrates apply evenly to all the units, something that may not necessarily be true.

Then if only a CC gas turbine unit is predicted to retire for a combined cycle plant, instead of getting into the complexity of whether the unit will be repowered and total plant life extended, we just say some of the plant capacity is closed and prorate everything accordingly. An approximation. No plant economics are considered in this simple model, or environmental compliance requirements. To some extent these are implict in the historical retirement data.
