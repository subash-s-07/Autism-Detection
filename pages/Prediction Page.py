import streamlit as st
import pandas as pd
import pickle
from lime.lime_tabular import LimeTabularExplainer
dbfile = open('Data_Models.pkl', 'rb')
models = pickle.load(dbfile)
dbfile.close()

x_train = pd.read_csv('x_train.csv')
x_train = x_train.drop('Unnamed: 0', axis=1)
y_train = pd.read_csv('y_train.csv')
y_train = y_train.drop('Unnamed: 0', axis=1)
train = pd.read_csv("Dav_data.csv")
train = train.drop("Unnamed: 0",axis = 1)
page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://wallpapersmug.com/download/3840x2400/4d7d03/light-colors-geometric-pattern-abstract.jpg");
        background-size: 120%;
        background-position: top left;
        background-attachment: local;
    }}
    [data-testid="stSidebar"] {{
        background-image: url("https://wallpapersmug.com/download/3840x2400/4d7d03/light-colors-geometric-pattern-abstract.jpg");
        background-size: 470%;
        background-position: top left;
        background-attachment: local;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);
    }}
    [data-testid="stToolbar"] {{
        right: 2rem;
    }}
    </style>
"""
st.set_page_config(
        page_title="Autism Diagnosis App",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded",
    )
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Autism Diagnosis App")
st.header("Input Features")

# Create columns for input fields
col1, col2, col3 = st.columns(3)

# Define input fields and labels
input_fields = [
    ("A1_Score", "A2_Score", "A3_Score"),
    ("A4_Score", "A5_Score", "A6_Score"),
    ("A7_Score", "A8_Score", "A9_Score"),
    ("A10_Score", "age", "ethnicity"),
    ("jaundice", "austim", "contry_of_res"),
    ("result", None, None)
]

# Initialize input features dictionary
input_features = {}

# Define ethnicity options and their corresponding integer values
ethnicity_options = ['White-European', 'South Asian', 'Middle Eastern ', 'Latino',
                    'Asian', 'Black', 'Turkish', 'Others', 'Hispanic', 'Pasifika', 'others']
ethnicity_values = [9, 7, 4, 3, 0, 1, 8, 5, 2, 6, 10]
country_options = ['United States', 'New Zealand', 'Canada', 'United Arab Emirates',
                'United Kingdom', 'Australia', 'Sri Lanka', 'India', 'Nicaragua',
                'Netherlands', 'Armenia', 'Sierra Leone', 'Argentina', 'Iceland',
                'Egypt', 'Afghanistan', 'Costa Rica', 'Jordan', 'Angola',
                'Pakistan', 'Brazil', 'Ireland', 'Viet Nam', 'Ethiopia', 'Finland',
                'Italy', 'France', 'Malaysia', 'Austria', 'Japan', 'Philippines',
                'Iran', 'Czech Republic', 'Russia', 'Romania', 'Mexico', 'Belgium',
                'Uruguay', 'Kazakhstan', 'Germany', 'South Africa', 'Aruba',
                'Saudi Arabia', 'Hong Kong', 'Serbia', 'Ecuador', 'Cyprus',
                'China', 'Bahamas', 'Bangladesh', 'Oman', 'Bolivia', 'Sweden',
                'American Samoa', 'Ukraine', 'Niger', 'Spain']
country_values = [54, 36, 13, 52, 53, 6, 49, 26, 37, 35, 4, 46, 3, 25, 19, 0, 15, 31, 2, 40, 12, 28, 56, 20, 21, 29, 22, 33, 7,
                30, 41, 27, 17, 43, 42, 34, 10, 55, 32, 23, 47, 5, 44, 24, 45, 18, 16, 14, 8, 9, 39, 11, 50, 1, 51, 38, 48]
# Iterate over rows and columns to create input fields
for row_idx, cols in enumerate(input_fields):
    for col_idx, label in enumerate(cols):
        if label:
            with col1 if col_idx == 0 else col2 if col_idx == 1 else col3:
                if label in ["jaundice", "austim"]:
                    input_features[label] = st.selectbox(f"{label}:", [0, 1])
                elif label == "ethnicity":
                    # Use the ethnicity_options list for the select box options
                    selected_ethnicity = st.selectbox(f"{label}:", ethnicity_options)
                    # Map the selected ethnicity to its corresponding integer value
                    input_features[label] = ethnicity_values[ethnicity_options.index(selected_ethnicity)]
                elif label == "contry_of_res":
                    selected_country = st.selectbox(f"{label}:", country_options)
                    input_features[label] = country_values[country_options.index(selected_country)]
                else:
                    input_features[label] = st.number_input(f"{label}:")

# Predict button
input_data = pd.DataFrame([input_features])

# Predict using the selected model
selected_model = st.selectbox("Select a Model", list(models.keys()))
prediction = models[selected_model].predict(input_data)[0]

# Define emoji URLs for different predictions
emoji_no = "https://c.tenor.com/tW_SaqozWz4AAAAM/thats-negative-danny-roberts.gif"
emoji_yes = "https://wordsjustforyou.com/wp-content/uploads/2019/08/Get-Well-Soon-Gif_01240819_wordsjustforyou.gif"

# Display the prediction using emojis
st.subheader("Model Prediction:")
if prediction == 0:
    st.image(emoji_no, width=200)
    st.info("No Autism Detected")
    st.info("This is the information message.")
else:
    st.image(emoji_yes, width=200)
    st.success("Autism Detected")
    st.success("This is a success message.")

# Lime explanation
explainer = LimeTabularExplainer(x_train.values, mode="classification", training_labels=y_train.index,
                                    feature_names=x_train.columns)
instance_to_explain = input_data.iloc[0]

def predict_fn(x):
    return models[selected_model].predict_proba(x)

explanation = explainer.explain_instance(instance_to_explain.values, predict_fn, num_features=len(input_data.columns))
explanation_html = explanation.as_html()

# Display Lime explanation as an image and the input features in two columns
col1, col2 = st.columns(2)

# Display Lime explanation as an image
col1.subheader("Lime Explanation (Image)")
explanation_image = explanation.as_pyplot_figure()
col1.pyplot(explanation_image)

# Display the input features
col2.subheader("Input Features:")
col2.write(input_features)
# Display the Lime explanation
st.subheader("Lime Explanation:")

# Create a list to store feature names and weights
feature_weights = [(feature, weight) for feature, weight in explanation.as_list()]

# Create a DataFrame from the list for better formatting
explanation_df = pd.DataFrame(feature_weights, columns=["Feature", "Weight"])

# Create a bar chart to visualize feature weights
st.bar_chart(explanation_df.set_index("Feature"))

# Create a data table to display feature names and weights
st.write("Feature Weights:")
st.dataframe(explanation_df)
st.components.v1.html(explanation_html, height=500)