import numpy as np
import pandas as pd
from   category_encoders             import *
from   sklearn.compose               import *
from   sklearn.ensemble              import *
from   sklearn.linear_model          import *
from   sklearn.impute                import *
from   sklearn.metrics               import *
from   sklearn.pipeline              import *
from   sklearn.preprocessing         import *
from   sklearn.model_selection       import *
from   sklearn.feature_selection     import *
from   imblearn.over_sampling        import *
from   sklearn.neural_network        import MLPClassifier
from   sklearn.gaussian_process      import GaussianProcessClassifier
from   sklearn.neighbors             import *
from   sklearn.naive_bayes           import *

class FillNaNs():
    def __init__(self, columns, values):
        self.columns = columns
        self.values = values
    
    def transform(self, X, **transform_params):        
        fillnas = {self.columns[i]: self.values[i] for i in range(len(self.columns))} 
        cpy_df = X.fillna(value=fillnas)
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self
    
class MergeSymptomDFs():
    """Specialized class that inputs a list of symptom columns 
       and returns a new column with a flattened list of all the symptoms"""
    def __init__(self, columns, col, final_cols, final_cols2):
        self.columns = columns
        self.col = col
        self.final_cols = final_cols
        self.final_cols2 = final_cols2

    def transform(self, X, **transform_params): 
        cpy_df = X.copy()
        cpy_df2 = X.copy()
        
        # Groups df based on ID and joins symptoms together
        for symp in self.columns:
            cpy_df = pd.merge(cpy_df, cpy_df.groupby(self.col)[symp].apply(np.array).reset_index(name=symp), on=self.col)
        
        # Iterate through symptom columns and adds it all into one singular column
        for idx in range(len(self.columns)-1):
            cpy_df[self.columns[idx+1]+'_x'] = cpy_df[self.columns[idx]+'_x'] + ', ' + cpy_df[self.columns[idx+1]+'_x']
            cpy_df[self.final_cols[0]] = cpy_df[self.columns[idx]+'_x']
            cpy_df = cpy_df.drop(columns=[self.columns[idx]+'_x',self.columns[idx]+'_y'])
        
        # Joins cleaned symptom df to original df
        cpy_df = pd.merge(cpy_df[self.final_cols],cpy_df2[self.final_cols2], on = 'id')          
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self  

    
class ConvertColumnLists():
    """Converts dataframe to have a new boolean column for every
       string in the list of strings 'list_strings' that returns
       whether or not the string is in 'column' """
    def __init__(self, list_strings, column):
        self.list_strings = list_strings
        self.column = column

    
    def transform(self, X, **transform_params): 
        cpy_df = X.copy()
        
        #creates a new column for every top adverse symptom
        for c in self.list_strings:
            cpy_df[c] = cpy_df[self.column].str.contains(c, case=False)
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self

class BooleanInAnyDF():
    """ Inputs columns in a df 'cols', and returns a new column 
        'new_col' that indicates whether or not 'value' is in any 
        of the input columns, then drops the input columns"""
    def __init__(self, cols, value, new_col):
        self.cols = cols
        self.value = value
        self.new_col = new_col
    
    def transform(self, X, **transform_params): 
        cpy_df = X.copy()
        df = (cpy_df[self.cols].isin([self.value]))
        cpy_df[self.new_col] = (df.any(axis=1))
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self

class DropColumns():
    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X, **transform_params):
        cpy_df = X.copy().drop(columns=self.columns)
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self
    
    
fillNa = {'hospdays':0.0, 'disable':'N', 'hosp_visit':'N', 'er_visit':'N', 'history': 'None',
          'allergies': 'None', 'other_meds': 'None'}

symptoms = [f'symptom{i}' for i in range(1,6)]

final_cols = ['all_symptoms', 'id', 'manu', 'dose' ,'route', 'state', 'age', 'sex', 'disable', 
              'history', 'other_meds', 'allergies']

covid_cols = ['SARS-CoV-2 test positive','COVID-19', 'all_symptoms', 'COVID']

text = ['history', 'allergies', 'other_meds', 'id']

history_list = ['asthma', 'arthritis', 'diabetes', 'migraines', 'hypothyroidism','gerd', 'high blood pressure', 'depression']

meds = ['levothyroxine', 'multivitamin', 'tylenol', 'adderall', 'ibuprofen',  'levothyroxine', 'naltrexone']

allergies = ['penicillin', 'codeine', 'amoxicillin', 'shellfish', 'aspirin', 'lactose']



d_manu  =          st.sidebar.selectbox('Manufacturer', ['PFIZER\BIONTECH', 'MODERNA', 'UNKNOWN MANUFACTURER'])
d_dose  =          st.sidebar.selectbox('# of Doses', ['1', '2', '3', '4', '5', '6', '7+'])
d_route =          st.sidebar.selectbox('Route', ['IM', 'SYR', 'OT', 'SC', 'UN', 'ID', 'JET'])
d_state  =         st.sidebar.selectbox('State', ['CA', 'TX', 'NY', 'FL', 'OH', 'IL', 'PA', 'WA', 'MA', 'WI', 'MI', 'CO',
                                                  'NJ', 'AZ', 'NC', 'MO', 'MN', 'IN', 'VA', 'GA', 'MD', 'TN', 'KY', 'NM',
                                                  'LA', 'IA', 'OR', 'CT', 'OK', 'ME', 'MS', 'AL', 'SC', 'NE', 'KS', 'NV',
                                                  'AR', 'NH', 'UT', 'PR', 'MT', 'ID', 'WV', 'AK', 'HI', 'ND', 'SD', 'RI',
                                                  'WY', 'VT', 'DE', 'DC', 'GU', 'MP', 'VI', 'MH', 'XB'])
d_age  =           st.sidebar.slider('Age', 0, 150, key=7)
d_sex=             st.sidebar.selectbox('Sex', ['F', 'M', 'U'])

d_disable  =       st.sidebar.selectbox('Disability', ['N', 'Y'])
d_asthma  =        st.sidebar.selectbox('Asthma', [True, False])
d_arthritis =      st.sidebar.selectbox('Arthritis', [True, False])
d_diabetes  =      st.sidebar.selectbox('Diabete', [True, False])
d_migraines  =     st.sidebar.selectbox('Migraines', [True, False])
d_hypothyroidism = st.sidebar.selectbox('Hypothyroidism', [True, False])

d_gerd  =          st.sidebar.selectbox('GERD', [True, False])
d_hbp  =           st.sidebar.selectbox('High Blood Pressure', [True, False])
d_depression =     st.sidebar.selectbox('Depression', [True, False])
d_levothyroxine=   st.sidebar.selectbox('Levothyroxine', [True, False])
d_multivitamin=    st.sidebar.selectbox('Multivitamin', [True, False])
d_tylenol =        st.sidebar.selectbox('Tylenol', [True, False])

d_adderall  =      st.sidebar.selectbox('Adderall', [True, False])
d_ibuprofen  =     st.sidebar.selectbox('Ibuprofen', [True, False])
d_naltrexone  =    st.sidebar.selectbox('Naltrexone', [True, False])
