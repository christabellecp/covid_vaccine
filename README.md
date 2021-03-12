# Binary Classification of Covid Positive Given Covid Vaccination

YouTube: https://youtu.be/Rkpv2V6BVWI 
<br>
DeepNote: https://deepnote.com/project/d6ae7e72-8a51-41d7-9e3d-b1345c1fc514#%2Ffinal.ipynb


## 1. Ask: Research Question / Hypothesis
#### Can we successfully build a model that can predict whether or not someone will get Covid-19 given that they have received a vaccination and if so, what model performs the best? 
  

## 2. Acquire Data
----
#### **Data**: CDC's Vaccine Adverse Event Reporting System
1. Data Variables 
    - **id**: Unique Patient ID
    - **manu**: makers of vaccine products
    - **dose**: the number of vaccine doses given for a vaccine
    - **route**: route of administration/ the path the vaccine was taken into the body
    - **site**: the site for a vaccine product
    - **recovd_date:** indicates when the patient recovered from adverse symptoms
    - **state**: Patient's Home State
    - **age**: Patient's Age
    - **sex**: Patient's Sex
    - **hospdays**: Days spent in the hospital recovering from adverse symptoms
    - **disable**: Whether or not patient is disabled
    - **recvd**: Whether or not patient recovered from adverse symptoms
    - **numdays**: Number of days patient had adverse symptoms
    - **adminby**: public, private, other, military, work, pharmacy, senior living, school, unknown agency
    - **hosp_visit**: Whether or not patient had to visit the hospital from adverse symptoms
    - **er_visit**: Whether or not patient had to visit the ER from adverse symptoms
    - **history**: Patient's istory of diseases
    - **allergies**: Patient's allergies
    - **other_meds**: Patient's current medications
    - **symptom1** 1st adverse symptom from vaccine
    - **symptom2**: 2nd adverse symptom from vaccine
    - **symptom3**: 3rd adverse symptom from vaccine
    - **symptom4**: 4th adverse symptom from vaccine
    - **symptom5**: 5th adverse symptom from vaccine
    
## 3. Process Data
1. **Fill NaN Columns with Appropriate Values** 

2. **Merge the Symptoms into Singular Column** 
    - Create a new column that holds a flattened list of all symptoms 
    
    
3. **Create a new boolean column for every top allergy, other diagnosis, and medication**

   
4. **Create 'Covid' Target Column** 
    - There are two symptoms that encompass whether or not someone got covid 'SARS-CoV-2 test positive' and'COVID-19'. 
    - Need to merge those two columns together and return True if either of them are True
    
    
5. **Drop 'SARS-CoV-2 test positive', 'COVID-19' and 'All Symptoms' Columns**
6. **Standardize Numerical Variables**
7. **Encode Categorical Variables**
8. **Ensure the Same Ratio of Classes for Train, Val and Test**have 
9. **Oversample Minority Class**

## 4. Model 
----
#### **Metric:** Balanced Accuracy since we have imbalanced data, balanced accuracy score is best to account for both covid positive and not covid positive outcome classes.

## Summary
1. With ~ 81.8% balanced accuracy, we can predict whether or not someone who receives a vaccine will still get covid
2. The best model turned out to be random forest, possibly due to the fact that it is robust to outliers and feature’s that aren’t as important will simply not get chosen during feature selection
3. Despite class imbalance, we were able to get the model to perform fairly well due to oversampling techniques.
4. This is a useful model for those who may feel uncertain about getting the vaccine and wants to know their chances of getting covid even after receiving the vaccine

## Limitations 
1. Model only used the top allergies, diseases, and medication. The data pipeline dropped all others that may have had higher predictive power.
2. A lot of other variables that would be predictive of whether or not someone still gets covid is missing (e.g, whether or not someone lives in a city).
3. Next steps: Continue working on feature engineering. Carry out methods to measure feature importance and encode more allergies/diseases/medication. 


