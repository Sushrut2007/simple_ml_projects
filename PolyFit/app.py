import streamlit as st
from src.utils import *

st.title('PolyFit')
st.caption('Know the best degree complexity for your Polynomial Regression!')
st.divider()

# --------------------
# User inputs
# --------------------
st.header('Step 1')

# File uploader
uploaded_file = st.file_uploader(
    'Upload a clean CSV file',
    type = 'csv'
)

# Validate the csv file to ensure proper columns
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate dataset
    is_valid = validate_dataset(df)

    if is_valid is True:
        st.success('Valid dataset!')
    else:
        st.warning('Invalid dataset!')

else:
    st.info('Please upload a CSV file before you can move further.')
    st.stop()


st.divider()
# Selection of target variable
st.header('Step 2')

target = st.selectbox('Target variable', options=[column for column in df.columns],
                      help='The target value to predict by the model')


st.divider()

# Maximum degree model, n = max_degree i.e n models will be trained (degree 1 to n)
st.header('Step 3')
 
max_degree = st.slider('Select the maximum degree of polynomial',
                       min_value=1, max_value=20, step=1,
                       help='n models will be trained, degree 1 to n.')

                       
st.divider()

# Tolerence in %, which can allow even a high error model that is less complex.
st.header('Step 4')

tolerance = st.slider('Select a tolerance in %', 
                      min_value=0, max_value=100, help='Maximum percentage increase in RMSE allowed compared with the best-performing model. A higher tolerance favors simpler polynomial degrees.')


