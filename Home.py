import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report as cr
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
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
data = pd.read_csv("Final_Data(1).csv")
data = data.drop("id",axis=1)
print(data.shape)

x=data.drop('Class/ASD',axis=1)
y=data['Class/ASD']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def skmetrics(get,name):
    st.title(name)
    lr_predictions = get.predict(x_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")

    classification_rep = classification_report(y_test, lr_predictions)

    print("Classification Report:")
    print(classification_rep)
    # Calculate the confusion matrix
    confusion = confusion_matrix(y_test, lr_predictions)

    # Plot the confusion matrix as a heatmap with a different color map ('Oranges')
    st.write("Confusion Matrix:")
    plt.figure(figsize=(4, 3))  # Reduced figsize
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Oranges', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    st.pyplot()

    # Create a line graph to compare actual vs. predicted labels
    x_values = np.arange(len(y_test))
    plt.figure(figsize=(6, 3))  # Reduced figsize

    # Set a lighter background color
    plt.gca().set_facecolor('#f0f0f0')

    plt.plot(x_values, y_test, label='Actual', marker='o', linestyle='-', color='blue')
    plt.plot(x_values, lr_predictions, label='Predicted', marker='x', linestyle='--', color='orange')
    plt.xlabel('Sample Index')
    plt.ylabel('Class/ASD')
    plt.title('Actual vs. Predicted Labels')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    st.write("Line Graph (Actual vs. Predicted Labels):")
    st.pyplot()

    # Calculate the ROC curve
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    fpr, tpr, thresholds = roc_curve(y_test_bin, lr_predictions)

    # Calculate the AUC (Area Under the Curve) score
    roc_auc = roc_auc_score(y_test_bin, lr_predictions)

    # Plot the ROC curve
    plt.figure(figsize=(4, 3))  # Reduced figsize
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.write("ROC Curve:")
    st.pyplot()
def main(data,db_accuracy,db):
    st.title("Autism Diagnosis App")
    st.header("What is Autism?")
    st.markdown("<p style='font-size:30px;color: #FF5733;'>Autism, or autism spectrum disorder (ASD), refers to a broad range of conditions characterized by challenges with social skills, repetitive behaviors, speech and nonverbal communication.</p>", unsafe_allow_html=True)


    st.header("Causes and Challenges")
    st.markdown("<p style='font-size: 30px; color: #FF5733;'>It is mostly influenced by a combination of genetic and environmental factors. Because autism is a spectrum disorder, each person with autism has a distinct set of strengths and challenges. The ways in which people with autism learn, think, and problem-solve can range from highly skilled to severely challenged.</p>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:30px;color: #FF5733;'>Research has made clear that high quality early intervention can improve learning, communication and social skills, as well as underlying brain development. Yet the diagnostic process can take several years.</p>", unsafe_allow_html=True)

    st.header("The Role of Machine Learning")
    st.markdown("<p style='font-size:30px;color: #FF5733;'>This dataset is composed of survey results for more than 700 people who filled an app form. There are labels portraying whether the person received a diagnosis of autism, allowing machine learning models to predict the likelihood of having autism, therefore allowing healthcare professionals to prioritize their resources.</p>", unsafe_allow_html=True)
    st.title("DataSet")
    st.header("Header of the Dataset:")
    st.dataframe(data.head())


    dark_color = "#00008B"

    # Define the color for the text inside <strong> tags
    strong_text_color = "#FF5733"  # Change this to your desired color

    # Define the font size for the text inside <strong> tags
    strong_text_size = "24px"  # Increase the font size by 20px (original: 4px)

    # Set the header with increased text size and dark color
    st.markdown("<h1>Dataset Info</h1>", unsafe_allow_html=True)

    # Set the text for each item with increased text size and different color for <strong> tags
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>ID:</strong> ID of the patient</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>A1_Score to A10_Score:</strong> Scores based on Autism Spectrum Quotient (AQ) 10 item screening tool</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>age:</strong> Age of the patient in years</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>gender:</strong> Gender of the patient</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>ethnicity:</strong> Ethnicity of the patient</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>jaundice:</strong> Whether the patient had jaundice at the time of birth</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>autism:</strong> Whether an immediate family member has been diagnosed with autism</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>country_of_res:</strong> Country of residence of the patient</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>used_app_before:</strong> Whether the patient has undergone a screening test before</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>result:</strong> Score for AQ1-10 screening test</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>age_desc:</strong> Age description of the patient</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>relation:</strong> Relation of the patient who completed the test</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {dark_color}; font-size: 18px;'><strong style='color: {strong_text_color}; font-size: {strong_text_size};'>Class/ASD:</strong> Classified result as 0 or 1. Here, 0 represents No and 1 represents Yes. This is the target column.</p>", unsafe_allow_html=True)


    data1 = {
        'A1_Score': [0.0439],
        'A2_Score': [0.0726],
        'A3_Score': [0.1533],
        'A4_Score': [0.1486],
        'A5_Score': [0.1175],
        'A6_Score': [0.1372],
        'A7_Score': [0.0697],
        'A8_Score': [0.027],
        'A9_Score': [0.1474],
        'A10_Score': [0.1172],
        'age': [0.0277],
        'gender': [0.0073],
        'ethnicity': [0.0484],
        'jaundice': [0.018],
        'austim': [0.0663],
        'contry_of_res': [0.1041],
        'used_app_before': [0.0],
        'result': [0.1512],
        'age_desc': [0.0],
        'relation': [0.0092]
    }

    df = pd.DataFrame(data1).T.reset_index()
    df.columns = ['Feature', 'Value']
    df = df.rename(columns={'index': 'Feature'})

    # Define a function for conditional formatting with lighter colors
    def color_light(val):
        color = 'lightcoral' if val < 0.05 else 'lightgreen'
        return f'background-color: {color}'

    # Apply conditional formatting to the DataFrame
    styled_df = df.style.applymap(color_light, subset=['Value'])

    # Create a Streamlit app with columns layout
    col1, col2 = st.columns([1, 1])  # Create two columns with equal width

    # Display the DataFrame in the first column
    with col1:
        st.title('Information Gain')
        st.dataframe(styled_df, width=500, height=400)

    # Create a bar chart with colors in the second column
    with col2:
        st.title('Bar Chart of Values')
        fig, ax = plt.subplots()
        bars = ax.bar(df['Feature'], df['Value'], color=[ 'lightgreen' if val >= 0.05 else 'lightcoral' for val in df['Value']])
        ax.set_xlabel('Feature')
        ax.set_ylabel('Value')
        
        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        st.pyplot(fig)
    model_names = ['Logistic Regression','Decision tree','Random forest','SVM','KNN','Naive bayers']

    # Title
    st.title("Models and its metrics")
    selected_model = st.selectbox('Select a Model', model_names)
    data.to_csv('Final_Data_1.csv')

    x=data.drop('Class/ASD',axis=1)
    y=data['Class/ASD']

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train.to_csv('x_train_1.csv')

    y_train.to_csv('y_train_1.csv')

    st.header(selected_model)
    x_1=pd.read_csv('1.csv')
    x_1=x_1.drop('Unnamed: 0',axis=1)
    y_1=pd.read_csv('2.csv')
    y_1=y_1.drop('Unnamed: 0',axis=1)
    try:
        def evaluate_model(model, X_test, y_test):
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = cr(y_test, y_pred,output_dict=True)
            return accuracy, report
        accuracy, report = evaluate_model(db[selected_model], x_1, y_1)

        average_report_df = pd.DataFrame(report).transpose()
        metrics = ['precision', 'recall', 'f1-score']
        st.write(average_report_df)
    except:
        pass
    skmetrics(db[selected_model],selected_model)

if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    train = pd.read_csv("train.csv")
    dbfile1 = open('Models_Acc.pkl', 'rb')
    dbfile2 = open('Data_Models.pkl', 'rb')
    db1 = pickle.load(dbfile1)
    db2 = pickle.load(dbfile2)
    main(train,db1,db2)
