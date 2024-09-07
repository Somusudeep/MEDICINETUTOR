import pickle
import streamlit as st

# Load the saved models
diabetes_model = pickle.load(open("./models/diabetes_model_new.sav", 'rb'))
heart_model = pickle.load(open("./models/heart_disease_model.sav", 'rb'))
parkinsons_model = pickle.load(open("./models/parkinsons_model.sav", 'rb'))

# Main page content
st.title("Disease Prediction System")

# Centralized buttons
st.markdown("""
<style>
    .centered-buttons {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 80vh;
    }
    .centered-buttons .stButton {
        margin: 0 10px;
        font-size: 18px;
        padding: 15px 25px;
        color: white;
        background-color: #007BFF;
        border: none;
        border-radius: 5px;
    }
    .centered-buttons .stButton:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button('Heart Disease Prediction'):
        st.session_state.page = 'Heart Disease Prediction'

with col2:
    if st.button('Diabetes Prediction'):
        st.session_state.page = 'Diabetes Prediction'

with col3:
    if st.button('Parkinson\'s Prediction'):
        st.session_state.page = 'Parkinson\'s Prediction'

# Display prediction form based on button click
if 'page' in st.session_state:
    selected = st.session_state.page
else:
    selected = None

# Heart Disease Prediction Page   
if selected == 'Heart Disease Prediction': 
    st.title('Heart Disease Prediction using ML')
    
    st.write("### Please enter the following details:")

    st.write("#### Chest Pain types:")
    st.write("1. **Type 1**: Typical Angina")
    st.write("2. **Type 2**: Atypical Angina")
    st.write("3. **Type 3**: Non-Anginal Pain")
    st.write("4. **Type 4**: Asymptomatic")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', min_value=0)
    
    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])
    
    with col3:
        cp = st.selectbox('Chest Pain types', ['Type 1', 'Type 2', 'Type 3', 'Type 4'])

    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0)

    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0)

    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])

    with col1:
        restecg = st.selectbox('Resting Electrocardiographic results', ['Normal', 'Abnormal'])

    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=0)

    with col3:
        exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])

    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0)

    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment', ['Up', 'Flat', 'Down'])
        st.write("1. **Up**: Up-sloping")
        st.write("2. **Flat**: Flat")
        st.write("3. **Down**: Down-sloping")

    with col3:
        ca = st.number_input('Major vessels colored by flourosopy', min_value=0)

    with col1:
        thal = st.selectbox('thal: 1 = normal; 2 = fixed defect; 3 = reversible defect', [1, 2, 3])
        st.write("1. **Normal**: Normal")
        st.write("2. **Fixed defect**: Fixed defect")
        st.write("3. **Reversible defect**: Reversible defect")
        
    # Code for Prediction
    heart_diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        if not all([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
            st.warning("Please fill in all the fields.")
        else:
            # Convert categorical values to numeric
            sex_numeric = 1 if sex == 'Male' else 0
            fbs_numeric = 1 if fbs == 'Yes' else 0
            exang_numeric = 1 if exang == 'Yes' else 0
            restecg_numeric = 1 if restecg == 'Abnormal' else 0
            cp_numeric = ['Type 1', 'Type 2', 'Type 3', 'Type 4'].index(cp) + 1
            slope_numeric = ['Up', 'Flat', 'Down'].index(slope)
            
            heart_prediction = heart_model.predict([[age, sex_numeric, cp_numeric, trestbps, chol, fbs_numeric, restecg_numeric, thalach, exang_numeric, oldpeak, slope_numeric, ca, thal]])                          
            
            if (heart_prediction[0] == 1):
              heart_diagnosis = 'The person has a heart disease'
            else:
              heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)  

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    
    st.title('Diabetes Prediction using ML')
    
    st.write("### Please enter the following details:")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0)
        
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0)
    
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0)
    
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0)
    
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0)
    
    with col3:
        BMI = st.number_input('BMI value', min_value=0.0)
    
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0)
    
    with col2:
        Age = st.number_input('Age of the Person', min_value=0)
        
    # Code for prediction
    diab_diagnosis=''
    
    if st.button('Diabetes Test Result'):
        if not all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            st.warning("Please fill in all the fields.")
        else:
            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            
            if (diab_prediction[0] == 1):
              diab_diagnosis = 'The person is diabetic'
            else:
              diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)  

# Parkinson's Prediction Page  
if selected == 'Parkinson\'s Prediction':    
    
    st.title("Parkinson's Disease Prediction using ML")
    
    st.write("### Please enter the following details:")

    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.number_input('MDVP: Fo(Hz)', min_value=0.0)
        
    with col2:
        fhi = st.number_input('MDVP: Fhi(Hz)', min_value=0.0)
        
    with col3:
        flo = st.number_input('MDVP: Flo(Hz)', min_value=0.0)
        
    with col4:
        Jitter_percent = st.number_input('MDVP: Jitter(%)', min_value=0.0)
        
    with col5:
        Jitter_Abs = st.number_input('MDVP: Jitter(Abs)', min_value=0.0)
        
    with col1:
        RAP = st.number_input('MDVP: RAP', min_value=0.0)
        
    with col2:
        PPQ = st.number_input('MDVP: PPQ', min_value=0.0)
        
    with col3:
        DDP = st.number_input('Jitter: DDP', min_value=0.0)
        
    with col4:
        Shimmer = st.number_input('MDVP: Shimmer', min_value=0.0)
        
    with col5:
        Shimmer_dB = st.number_input('MDVP: Shimmer(dB)', min_value=0.0)
        
    with col1:
        APQ3 = st.number_input('Shimmer: APQ3', min_value=0.0)
        
    with col2:
        APQ5 = st.number_input('Shimmer: APQ5', min_value=0.0)
        
    with col3:
        APQ = st.number_input('Shimmer: APQ', min_value=0.0)
        
    with col4:
        DDA = st.number_input('Shimmer: DDA', min_value=0.0)
        
    with col5:
        NHR = st.number_input('NHR', min_value=0.0)
        
    with col1:
        HNR = st.number_input('HNR', min_value=0.0)
        
    with col2:
        RPDE = st.number_input('RPDE', min_value=0.0)
        
    with col3:
        DFA = st.number_input('DFA', min_value=0.0)
        
    with col4:
        spread1 = st.number_input('spread1', min_value=0.0)
        
    with col5:
        spread2 = st.number_input('spread2', min_value=0.0)
        
    with col1:
        D2 = st.number_input('D2', min_value=0.0)
        
    with col2:
        PPE = st.number_input('PPE', min_value=0.0)
        
    # Code for Prediction
    parkinsons_diagnosis = ''
    
    if st.button("Parkinson's Test Result"):
        if not all([fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]):
            st.warning("Please fill in all the fields.")
        else:
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])                          
            
            if (parkinsons_prediction[0] == 1):
              parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
              parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)
