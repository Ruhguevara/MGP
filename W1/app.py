import streamlit as st
import pandas as pd
import numpy as np
import joblib

# streamlit run app.py

from predictions import ordinal_encoder, get_prediction

model = joblib.load(r'W1/random_forest_final.joblib')

st.set_page_config(page_title='Accident Severity Prediction', 
                   page_icon=':car:', layout='wide', initial_sidebar_state='auto')

# Creating option list for drpdown menu
options_day = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ["18-30", "31-50", "Over 51", "Unknown", "Under 18"]

options_acc_area = ['Residential areas', 'Office areas', '  Recreational areas',
 ' Industrial areas', 'Other', ' Church areas', '  Market areas',
 'Unknown', 'Rural village areas', ' Outside rural areas', ' Hospital areas',
 'School areas', 'Rural village areasOffice areas', 'Recreational areas']

options_cause = ['Moving Backward', 'Overtaking', 'Changing lane to the left',
 'Changing lane to the right', 'Overloading', 'Other',
 'No priority to vehicle', 'No priority to pedestrian', 'No distancing',
 'Getting off the vehicle improperly', 'Improper parking', 'Overspeed',
 'Driving carelessly', 'Driving at high speed', 'Driving to the left',
 'Unknown', 'Overturning', 'Turnover', 'Driving under the influence of drugs',
 'Drunk driving']

options_vehicle_type = ['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)',
 'Public (13?45 seats)', 'Lorry (11?40Q)', 'Long lorry', 'Public (12 seats)',
 'Taxi', 'Pick up upto 10Q', 'Stationwagen', 'Ridden horse', 'Other', 'Bajaj',
 'Turbo', 'Motorcycle', 'Special vehicle', 'Bicycle']

options_driver_exp = ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No Licence', 'Below 1yr',
 'unknown']

options_lanes = ['Undivided Two way', 'other', 'Double carriageway (median)', 'One way',
 'Two-way (divided with solid lines road marking)',
 'Two-way (divided with broken lines road marking)', 'Unknown']

features = ["hour", "day_of_week", "casualties", "accident_cause", "vehicles_involved", 
            "vehicle_type", "driver_age", "accident_area", "driver_experience", "lanes"]

st.markdown("<h1 style='text-align: center; color: black;'> Accident Severity Prediction </h1>", 
            unsafe_allow_html=True)

def main():
    with st.form("prediction_form"):

        st.subheader("Enter the input for following features:")

        hour = st.slider("Hour", 0, 23, value =0, format="%d")
        day_of_week = st.selectbox("Day of Week", options = options_day)
        casualties = st.slider("Accident hour", 1, 8, value =0, format="%d")
        accident_cause = st.selectbox("Accident Cause", options = options_cause)
        vehicles_involved = st.slider("Vehicles Involved", 1, 7, value =0, format="%d")
        vehicle_type = st.selectbox("Vehicle Type", options = options_vehicle_type)
        driver_age = st.selectbox("Driver Age", options = options_age)
        accident_area = st.selectbox("Accident Area", options = options_acc_area)
        driver_experience = st.selectbox("Driver Experience", options = options_driver_exp)
        lanes = st.selectbox("Lanes", options = options_lanes)

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        day_of_week = ordinal_encoder(day_of_week, options_day)
        accident_cause = ordinal_encoder(accident_cause, options_cause)
        vehicle_type = ordinal_encoder(vehicle_type, options_vehicle_type)
        driver_age = ordinal_encoder(driver_age, options_age)
        accident_area = ordinal_encoder(accident_area, options_acc_area)
        driver_experience = ordinal_encoder(driver_experience, options_driver_exp)
        lanes = ordinal_encoder(lanes, options_lanes)

        data = np.array([hour, day_of_week, casualties, accident_cause, vehicles_involved, 
                         vehicle_type, driver_age, accident_area, driver_experience, lanes]).reshape(1, -1)

        pred = get_prediction(data = data, model = model)

        st.write("The predicted severity of the accident is: ", pred[0])

if __name__ == '__main__':
    main()

