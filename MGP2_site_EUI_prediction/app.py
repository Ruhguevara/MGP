import streamlit as st
import pandas as pd
import numpy as np
import joblib

# streamlit run app.py

from predictions import ordinal_encoder, get_prediction

model = joblib.load(r'data/catb_best_final.joblib')

# WORK IN PROGRESS, TOO MANY VARIABLES

if __name__ == '__main__':
    main()

