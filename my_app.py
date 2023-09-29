import streamlit as st
from pycaret.datasets import get_data
from pycaret.regression import *

st.markdown("# Pycaret AutoML in Streamlit!")

st.markdown("#### read data from pycaret repo")
data = get_data('insurance')
st.write(data)

st.markdown("#### initialize setup")
s = setup(data, target='charges')
st.write(s)

st.markdown("#### check all the available models")
st.write(models())

st.markdown("#### train decision tree")
dt = create_model('dt')
st.write(dt)

st.markdown("#### compare all models")
best_model = compare_models()
st.write(best_model)

st.markdown("#### predict on hold-out")
pred_holdout = predict_model(best_model)
st.write(pred_holdout)
