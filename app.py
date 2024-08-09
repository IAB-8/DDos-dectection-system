import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
# Load the trained model and scaler
model = joblib.load('grid_search_hist.pkl')

# Feature names used in the model
feature_names = ['Total Fwd Packets', 'Total Length of Fwd Packets',
       'Fwd Packet Length Max', 'Fwd Packet Length Mean',
       'Fwd Packet Length Std', 'Bwd Packet Length Min',
       'Bwd Packet Length Mean', 'Fwd IAT Total', 'Fwd IAT Std', 'Fwd IAT Max',
       'Fwd Header Length', 'Packet Length Mean', 'Average Packet Size',
       'Avg Fwd Segment Size', 'Fwd Header Length.1', 'Subflow Fwd Packets',
       'Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       'Init_Win_bytes_backward', 'act_data_pkt_fwd']

sample_data = pd.DataFrame({
    'Total Fwd Packets': [5, 10, 15],
    'Total Backward Packets': [3, 8, 12],
    'Total Length of Fwd Packets': [1000, 2000, 3000],
    'Fwd Packet Length Max': [500, 1000, 1500],
    'Fwd Packet Length Mean': [150, 300, 450],
    'Fwd Packet Length Std': [200, 400, 600],
    'Bwd Packet Length Min': [5, 10, 15],
    'Fwd IAT Total': [1000000, 2000000, 3000000],
    'Fwd IAT Mean': [100000, 200000, 300000],
    'Fwd IAT Std': [200000, 400000, 600000],
    'Fwd IAT Max': [5000000, 10000000, 15000000],
    'Fwd Header Length': [60, 70, 80],
    'Average Packet Size': [600, 1200, 1800],
    'Avg Fwd Segment Size': [150, 300, 450],
    'Fwd Header Length.1': [60, 70, 80],
    'Subflow Fwd Packets': [5, 10, 15],
    'Subflow Fwd Bytes': [1000, 2000, 3000],
    'Init_Win_bytes_forward': [5000, 10000, 15000],
    'act_data_pkt_fwd': [2, 4, 6]
})
# Class labels mapping
class_mapping = {
    0: 'BENIGN',
    1: 'DDos'}
# Dataset statistics
stats = {
    'Total Fwd Packets': {'mean': 4.874972, 'std': 15.423004, 'min': 1, 'max': 1932},
    'Total Length of Fwd Packets': {'mean': 939.479993, 'std': 3249.429866, 'min': 0, 'max': 183012},
    'Fwd Packet Length Max': {'mean': 538.545235, 'std': 1864.144128, 'min': 0, 'max': 11680},
    'Fwd Packet Length Mean': {'mean': 164.829636, 'std': 504.896961, 'min': 0, 'max': 3867},
    'Fwd Packet Length Std': {'mean': 214.911050, 'std': 797.417625, 'min': 0, 'max': 6692.644993},
    'Bwd Packet Length Min': {'mean': 16.719072, 'std': 50.480966, 'min': 0, 'max': 1460},
    'Bwd Packet Length Mean': {'mean': 890.552629, 'std': 1120.328575, 'min': 0, 'max': 5800.500000},
    'Fwd IAT Total': {'mean': 1.539680e+07, 'std': 3.160847e+07, 'min': 0, 'max': 1.200000e+08},
    'Fwd IAT Std': {'mean': 5.195299e+06, 'std': 1.078643e+07, 'min': 0, 'max': 7.670000e+07},
    'Fwd IAT Max': {'mean': 1.299457e+07, 'std': 2.748888e+07, 'min': 0, 'max': 1.200000e+08},
    'Fwd Header Length': {'mean': 111.523649, 'std': 375.793989, 'min': 0, 'max': 39396},
    'Packet Length Mean': {'mean': 515.011263, 'std': 559.065245, 'min': 0, 'max': 1936.833333},
    'Average Packet Size': {'mean': 574.579024, 'std': 626.097078, 'min': 0, 'max': 2528},
    'Avg Fwd Segment Size': {'mean': 164.829636, 'std': 504.896961, 'min': 0, 'max': 3867},
    'Fwd Header Length.1': {'mean': 111.523649, 'std': 375.793989, 'min': 0, 'max': 39396},
    'Subflow Fwd Packets': {'mean': 4.874972, 'std': 15.423004, 'min': 1, 'max': 1932},
    'Subflow Fwd Bytes': {'mean': 939.479993, 'std': 3249.429866, 'min': 0, 'max': 183012},
    'Init_Win_bytes_forward': {'mean': 4247.474336, 'std': 8037.836210, 'min': -1, 'max': 65535},
    'Init_Win_bytes_backward': {'mean': 600.768987, 'std': 4317.595552, 'min': -1, 'max': 65535},
    'act_data_pkt_fwd': {'mean': 3.311556, 'std': 12.270118, 'min': 0, 'max': 1931}
}
def randomize_input():
    randomized_input = {}
    for feature in feature_names:
        if feature in stats:
            mean = stats[feature]['mean']
            std = stats[feature]['std']
            min_val = stats[feature]['min']
            max_val = stats[feature]['max']
            # Generate a random value from a normal distribution
            value = np.random.normal(mean, std)
            # Clip the value to be within the min and max bounds
            value = np.clip(value, min_val, max_val)
            randomized_input[feature] = value
    return randomized_input

# Streamlit UI
st.title("DDos Detection System")

st.markdown("""
### About
Welcome to the Network Traffic Classifier! This app predicts whether network traffic is 'BENIGN' or 'DDos' based on user inputs. 

**Key Features:**
- **Interactive Analysis**: Enter your network traffic data and get immediate predictions.
- **Visual Insights**: View visualizations of feature distributions and model predictions.
- **Educational Content**: Learn about network traffic characteristics and security.

### How to Use:
1. **Input Data**: Enter the values for each feature in the sidebar.
2. **Visualize**: Check out feature distributions and predictions in charts.
3. **Predict**: Click the "Predict" button to get a classification of your network traffic.

Stay informed and secure with our easy-to-use tool!
""")

# Sidebar for user input
st.sidebar.header("Enter Network Traffic Data")

# Create a session state for user input
if 'user_input' not in st.session_state:
    st.session_state.user_input = {feature: 0.0 for feature in feature_names}

user_input = st.session_state.user_input

for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(f"{feature}:", min_value=-2.0, max_value=1000000000.0, value=user_input[feature])

# Buttons for actions
col1, col2 = st.sidebar.columns([2, 1])
with col1:
    if st.sidebar.button("Randomize Input"):
        st.session_state.user_input = randomize_input()
        st.rerun()  # Refresh the app to show randomized values

with col2:
    if st.sidebar.button("Reset"):
        st.session_state.user_input = {feature: 0.0 for feature in feature_names}
        st.rerun()  # Refresh the app to reset values
if st.sidebar.button("Predict"):
    # Create a DataFrame from user input
    input_data = pd.DataFrame([user_input])
    
    
    # Make a prediction
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)
    predicted_class = class_mapping.get(prediction[0], 'Unknown')
    
    # Display input data and prediction probability
    st.write("Input Data:")
    st.write(input_data)
    st.write("Prediction Probability:")
    st.write(proba)
    
    # Display prediction with color coding
    if predicted_class == 'BENIGN':
        st.subheader("Prediction:")
        st.success(f"Predicted Class: {predicted_class}", icon="✅")
        st.markdown('<style>div.row-widget.stButton {color: green;}</style>', unsafe_allow_html=True)
    elif predicted_class == 'DDos':
        st.subheader("Prediction:")
        st.error(f"Predicted Class: {predicted_class}", icon="❌")
        st.markdown('<style>div.row-widget.stButton {color: red;}</style>', unsafe_allow_html=True)
    else:
        st.subheader("Prediction:")
        st.warning(f"Predicted Class: {predicted_class}")

    rf_model= joblib.load('best_Random Forest_model.pkl')
    if hasattr(rf_model, 'feature_importances_'):
        feature_importances = rf_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        
        st.subheader("Feature Importances:")
        st.write(importance_df)
        
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        st.pyplot(fig)

# Display sample data for testing
st.sidebar.header("Sample Data")
if st.sidebar.checkbox("Show Sample Data"):
    st.subheader("Sample Network Traffic Data")
    st.write(sample_data)  # Ensure sample_data is defined elsewhere in your script

# Educational Content
st.markdown("""
### Educational Content
**Network Traffic Analysis:**
Network traffic can reveal a lot about the nature of activities occurring on a network. Identifying abnormal patterns, such as 'DDos' attacks, helps in securing networks and preventing disruptions.

**Key Terms:**
- **BENIGN**: Normal network traffic that does not indicate any threats.
- **DDos**: Distributed Denial of Service attack, which aims to overwhelm a network with excessive traffic.

Stay proactive in understanding and managing your network traffic to maintain a secure and efficient network environment.
""")