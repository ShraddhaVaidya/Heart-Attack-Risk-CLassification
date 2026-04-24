import pickle
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

model = pickle.load(open('rf_model.pkl', 'rb'))

st.title('Heart Attack Risk CLassification')

#create input features
age = st.number_input('Age', min_value=20, max_value=100, value=25)
restingbp = st.number_input('Resting Blood Pressure', min_value=0,max_value=300,value=100)
cholestrol = st.number_input('Cholestrol', min_value=0,max_value=700,value=100)
fastingbs = st.selectbox('Fasting BS',(0,1))
maxhr = st.number_input('Max HR', min_value=60,max_value=250,value=140)
oldpeak= st.number_input('Oldpeak', min_value=-3.0,max_value=6.5,value=1.0)
gender= st.selectbox('Gender',('M','F'))
chestpaintype= st.selectbox('Chest Pain',('ATA','NAP','ASY','TA'))
restingecg= st.selectbox('Resting ECG',('Normal','ST','LVH'))
exerciseangina= st.selectbox('Exercise Angina',('N','Y'))
st_slope= st.selectbox('ST Slope',('Up','Flat','Down'))

Exercise_Agina=1 if exerciseangina=='Y' else 0

Sex_F= 1 if gender=='F' else 0
Sex_M= 1 if gender=='M' else 0

Chest_PainType_dict= {'ASY':3,'NAP':2,'ATA':1,'TA':0}
Chest_PainType=Chest_PainType_dict[chestpaintype]

Resting_ECG_dict = {'Normal':0,'LVH':1,'ST':2}
Resting_ECG=Resting_ECG_dict[restingecg]

st_Slope_dict= {'Down':0,'Up':1,'Flat':2}
st_Slope=st_Slope_dict[st_slope]

input_features=pd.DataFrame({
    'Age':[age],
    'RestingBP':[restingbp],
    'Cholesterol':[cholestrol],
    'FastingBS':[fastingbs],
    'MaxHR':[maxhr],
    'Oldpeak':[oldpeak],
    'Exercise_Angina':[Exercise_Agina],
    'Sex_F':[Sex_F],
    'Sex_M':[Sex_M],
    'Chest_PainType':[Chest_PainType],
    'Resting_ECG':[Resting_ECG],
    'st_Slope':[st_Slope],
})

scaler=StandardScaler()
input_features[['Age','RestingBP','Cholesterol','MaxHR']]=scaler.fit_transform(input_features[['Age','RestingBP','Cholesterol','MaxHR']])

if st.button('Predict'):
  predictions=model.predict(input_features)[0]
  if predictions==1:
    st.error('⚠️High Risk of Heart Attack')
  else:
    st.success('Low Risk of Heart Attack😎👍')
