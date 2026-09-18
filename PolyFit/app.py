import streamlit as st
import hashlib
from src.utils import *

# Initialize session states
if 'model_settings' not in st.session_state:
    st.session_state.model_settings = {
        'current_file_hash' : None,

        'current_settings' : {
            'target': None,
            'max_degree': None,
            'tolerance': None
        },

        'saved_settings_by_file': {}
    }


st.title('PolyFit')
st.caption('Know the best degree complexity for your Polynomial Regression!')
st.divider()


def save_model_settings(file_hash, settings, persist=False):
    """
    Save the model settings for the current dataset in session state.
    If the user wants to save the model settings for future use, also save that seperatly.
    """
    st.session_state['model_settings']['current_file_hash'] = file_hash
    st.session_state['model_settings']['current_settings'] = settings.copy()

    if persist is True:
        st.session_state['model_settings']['saved_settings_by_file'][file_hash] = settings.copy()


def has_saved_settings(uploaded_file):
    """
    Check if the new csv file uploaded has saved settings or not.

    Returns: bool
        True is file is new, False if the user has previously uploaded the same file.
    """

    current_hash = get_file_hash(uploaded_file)

    # Set as the current file hash
    st.session_state['model_settings']['current_hash'] = current_hash

    # Check if current file settings were saved previously
    is_saved = current_hash in st.session_state['model_settings']['saved_settings_by_file']

    return current_hash, is_saved

def set_model_settings():
    """
    Generate a form for the model settings upon selection of valid dataset.
    """

    with st.form('polyfit_model_settings'):
        # Selection of target variable
        st.header('Step 1')

        target = st.selectbox('Target variable', options=[column for column in df.columns],
                            help='The target value to predict by the model')


        st.divider()
        # Maximum degree model, n = max_degree i.e n models will be trained (degree 1 to n)
        st.header('Step 2')
        
        max_degree = st.slider('Select the maximum degree of polynomial',
                            min_value=1, max_value=20, step=1,
                            help='n models will be trained, degree 1 to n.')

                            
        st.divider()
        # Tolerence in %, which can allow even a high error model that is less complex.
        st.header('Step 3')

        tolerance = st.slider('Select a tolerance in %', 
                            min_value=0, max_value=100, help='Maximum percentage increase in RMSE allowed compared with the best-performing model. A higher tolerance favors simpler polynomial degrees.')

        save_settings = st.checkbox('Save model settings for this csv?')

        # Submit button
        submitted = st.form_submit_button('Finalize dataset and model settings')

        # Return settings as a dictionary upon submission
        if submitted:
            st.success('Model settings set successfully!')
            settings = {
                'target': target,
                'max_degree': max_degree,
                'tolerance': tolerance
            }

            return settings, save_settings
        return None


# Dataset uploader
uploaded_file = st.file_uploader(
    'Upload a clean file',
    type = 'csv'
)

# Validate the csv file
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check if the current uploaded file model settings was saved previously
    file_hash, is_saved = has_saved_settings(uploaded_file)

    # User has saved settings for the uploaded csv
    if is_saved is True:
        # Ask the user to load the previous settings
        to_load_settings = st.radio('How would you like to configure this file?', ['Load saved settings', 'Enter new settings'])
        # Load and save as the current settings 
        if to_load_settings == 'Load saved settings':
        
            saved_settings = (st.session_state['model_settings']
                              ['saved_settings_by_file'][file_hash]
                              .copy())

            save_model_settings(file_hash, saved_settings, persist=False)

        # Display the form
        else:
            form_result = set_model_settings()
            if form_result is not None:
                settings, save_settings = form_result
                # Save the NEW model settings if the user wants, else don't
                if save_settings is True:
                    save_model_settings(file_hash, settings, persist=True)
                else:
                    save_model_settings(file_hash, settings, persist=False)

    # The csv file does not have any saved settings
    else:
        # Validate the csv to ensure proper format
        is_valid = validate_dataset(df)
        # Display model settings form
        if is_valid is True:
            st.success('Valid dataset!')
            st.divider()

            form_result = set_model_settings()
            if form_result is not None:
                settings, save_settings = form_result
                # Save the model settings if the user wants, else don't
                if save_settings is True:
                    save_model_settings(file_hash, settings, persist=True)
                else:
                    save_model_settings(file_hash, settings, persist=False)

        # Invalid dataset
        else:
            st.warning('Invalid dataset!')
            st.stop()

else:
    st.info('Please upload a CSV file before you can move further.')
    st.stop()


st.write(st.session_state)
