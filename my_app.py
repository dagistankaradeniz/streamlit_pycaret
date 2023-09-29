import streamlit as st
from gplearn.genetic import SymbolicRegressor
from pycaret.datasets import get_data
from pycaret.regression import *

st.markdown("# `Pycaret AutoML` in Streamlit!")

st.markdown("#### read data from `pycaret` repo")
data = get_data('insurance')
st.write(data)

st.markdown("#### initialize setup")
with st.spinner('Preparing...'):
    s = setup(data, target='charges')
with st.expander("expand to see data"):
    st.write(s)

st.markdown("#### check all the available models")
with st.spinner('Preparing...'):
    st.write(models())

st.markdown("#### `train` decision tree")
with st.spinner('Preparing...'):
    dt = create_model('dt')
st.write(dt)

st.markdown("#### compare all models")
with st.spinner('Preparing...'):
    best_model = compare_models()
st.write(best_model)

st.markdown("#### `predict` on hold-out")
with st.spinner('Preparing...'):
    pred_holdout = predict_model(best_model)
st.write(pred_holdout)

st.markdown('### create copy of data drop target column')
data2 = data.copy()
data2.drop('charges', axis=1, inplace=True)
st.write(data2)

st.markdown('### generate predictions')
with st.spinner('Preparing...'):
    predictions = predict_model(best_model, data=data2)
st.write(predictions)

st.markdown('### import untrained estimator')
sc = SymbolicRegressor()
st.markdown('### `train` using create_model')
with st.spinner('Preparing...'):
    sc_trained = create_model(sc)
st.write(sc_trained)

st.markdown('### check hold-out score')
with st.spinner('Preparing...'):
    pred_holdout_sc = predict_model(sc_trained)
st.write(pred_holdout_sc)
