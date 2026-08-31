import streamlit as st
from src.utils import *

st.title('PolyFit Testing Branch')
st.divider()


uploaded_file = st.file_uploader(
    'Upload a clean CSV file',
    type = 'csv'
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.dataframe(df)

# Valid dataset
is_valid = validate_dataset(df)

if is_valid is True:
    st.success('Valid dataset!')
else:
    st.warning('Invalid dataset!')

st.write(df.dtypes)