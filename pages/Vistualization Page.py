import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
import plotly.figure_factory as ff
from scipy import stats
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import datacleaner
from datacleaner import autoclean
import warnings
warnings.filterwarnings("ignore")

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
st.markdown(page_bg_img, unsafe_allow_html=True)

def mean_autism():
    # Calculate mean scores
    score_features = train.filter(regex='A[0-9]_', axis=1).columns.tolist()
    mean_scores = train.groupby('Class/ASD')[score_features].mean().T

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colormap = 'viridis'

    mean_scores.plot(kind='bar', ax=ax, colormap=colormap, width=0.8)
    plt.title('Mean Score - Autism Spectrum Quotient (AQ) 10 item screening tool')
    plt.xlabel('AQ Question')
    plt.ylabel('Mean Score')

    x_ticks = np.arange(len(score_features))
    x_tick_labels = [x.split('_')[0] for x in score_features]
    plt.xticks(x_ticks, x_tick_labels, rotation=0)

    for i, col in enumerate(mean_scores.columns):
        for x, y in zip(x_ticks, mean_scores[col]):
            plt.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom')

    plt.legend(title='Class/ASD', loc='upper right')
    plt.gca().set_facecolor('#F8004F')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    
#mean_autism()
def Treemap():
    color_scale = px.colors.qualitative.Set3

    fig = px.treemap(train,
                    path=['contry_of_res', 'Class/ASD'],
                    color='Class/ASD',
                    color_continuous_scale=color_scale,
                    labels={'contry_of_res': 'Country of Residence', 'Class/ASD': 'Class/ASD'},
    )
    fig.update_layout(
        title="<b>COUNTRY OF RESIDENCE OF THE PATIENT - TREEMAP</b>",
        title_font=dict(size=20, family="Sans Serif"),
        height=600,
        width=1000,
        template='plotly_dark', 
        autosize=False,
        margin=dict(l=50, r=50, b=50, t=100),
        treemapcolorway=color_scale, 
    )
    fig.update_layout(margin=dict(t=50, l=50, r=50, b=100))

    #fig.show()
    st.plotly_chart(fig)


#Treemap()
def ageDistribution():
    col = 'age'
    title = "AGE DISTRIBUTION"
    
    # Create a histogram using Seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(train[col], kde=True, color='skyblue', bins=10, ax=ax)
    
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel("Age", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.gca().set_facecolor('lightgray') 
    
    # Display the plot using Streamlit
    st.pyplot(fig)
    
def ethinicity():
    ethnicity_colors = ['chartreuse', 'coral', 'magenta', 'dodgerblue', 'darkgreen', 'darkorchid', 'brown', 'BlueViolet', 'cyan', 'coral', 'Turquoise', 'Gold']

    # Create a countplot with custom colors
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set(style="whitegrid")

    sns.countplot(data=train, x='ethnicity', palette=ethnicity_colors, ax=ax)

    plt.title("Autism by Ethnicity", fontsize=16, weight='bold')
    plt.xlabel("Ethnicity", fontsize=14)
    plt.ylabel("Count", fontsize=14)

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Display the plot using Streamlit
    st.pyplot(fig)

def screeningTest():
    
    ASD_pos_result = train[train['Class/ASD'] == 1]['result']
    ASD_neg_result = train[train['Class/ASD'] == 0]['result']

    # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Violin(x=ASD_pos_result, line_color='lightseagreen', name='ASD_positive', y0=0))
    fig.add_trace(go.Violin(x=ASD_neg_result, line_color='red', name='ASD_negative', y0=0))

    fig.update_traces(orientation='h', side='positive', meanline_visible=True)
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

    fig.update_layout(
        title='<b> Screening Test Result Distribution (QA1-10) <b>',
        font_family="San Serif",
        xaxis_title='Result',
        titlefont={'size': 20},
        width=800,
        height=500,
        template="plotly_white",
        showlegend=True,
    )
    fig.update_yaxes(showgrid=False, showline=False, showticklabels=False)

    # Display the plot using Streamlit
    st.plotly_chart(fig)

def heatmap():
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    # Function to plot Cramer's V correlation heatmap
    def plot_cramersV_corr(df):
        rows = []
        for x in df:
            col = []
            for y in df:
                cramers = cramers_v(df[x], df[y])
                col.append(round(cramers, 2))
            rows.append(col)

        cramers_results = np.array(rows)
        df_corr = pd.DataFrame(cramers_results, columns=df.columns, index=df.columns)

        mask = np.triu(np.ones_like(df_corr, dtype=bool))
        df_corr = df_corr.mask(mask)

        fig = go.Figure(data=go.Heatmap(
            z=df_corr.values,
            x=df_corr.columns.values,
            y=df_corr.index.values,
            colorscale='purples',
        ))
        fig.update_layout(
            title='<b>Correlation Heatmap (Categorical features) <b>',
            font_family="San Serif",
            title_x=0.5,
            titlefont={'size': 20},
            width=750,
            height=700,
            xaxis_showgrid=False,
            xaxis={'side': 'bottom'},
            yaxis_showgrid=False,
            yaxis_autorange='reversed',
            autosize=False,
            margin=dict(l=150, r=50, b=150, t=70, pad=0),
        )

        # Display the plot using Streamlit
        st.plotly_chart(fig)

    # Call the function to plot the heatmap
    plot_cramersV_corr(train.drop(['age', 'result'], axis=1))


def relationCompleted():
    age_plot = train.groupby(['age', 'Class/ASD']).size().reset_index(name="size")

    # Create a figure and set its size
    fig, ax = plt.subplots(figsize=(13, 6))

    # Create a histogram plot with KDE and color-coded by Class/ASD
    sns.histplot(data=age_plot, x='age', hue="Class/ASD", kde=True, palette='Dark2', ax=ax)

    # Set plot title and labels
    ax.set_title("Relation of patients who completed the test")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")

    # Display the plot using Streamlit
    st.pyplot(fig)

def screening():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a KDE plot with color-coded by Class/ASD
    sns.kdeplot(data=train, x='result', hue="Class/ASD", palette="CMRmap_r", fill=True, common_norm=False, ax=ax)

    # Set plot title and labels
    ax.set_title("Screening Test Result Distribution")
    ax.set_xlabel("Result")
    ax.set_ylabel("Density")

    # Display the plot using Streamlit
    st.pyplot(fig)

def Autismcountry():
    # Get the top 10 countries by residence count
    trainn = train.copy()
    trainn['index'] = range(len(trainn))
    country_plt = trainn["contry_of_res"].value_counts().reset_index().sort_values("contry_of_res", ascending=False).iloc[:10, :]

    # Create and display the barplot
    plt.figure(figsize=(20, 6))
    sns.barplot(x="index", y="contry_of_res", data=country_plt, palette="BuPu_r")
    plt.xlabel("Country of Residence")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot()


    

def ExtraTree(train):

    train , test = train_test_split(train)

    X = train.drop(columns=["Class/ASD",'used_app_before','result']).copy() #drop some usless features
    y = train["Class/ASD"].copy()
    X_test = test.drop(columns=['used_app_before','result',"Class/ASD"]).copy()

    cat_col = [col for col in X.columns if X[col].dtype == 'object']
    num_col = [col for col in X.columns if X[col].dtype == 'int']

    label_encoder = LabelEncoder()
    train_x = X.copy()
    test_x = X_test.copy()
    for col in cat_col:
            train_x[col] = label_encoder.fit_transform(train_x[col])
            test_x[col] = label_encoder.fit_transform(test_x[col])

    X_train, X_valid, y_train, y_valid = train_test_split(train_x,y ,test_size=0.15, random_state=42)




    n_estimators = 3647
    max_depth = 75
    min_samples_split = 14
    min_samples_leaf = 15
    criterion = 'gini'
    ##
    splits = 2
    predictions = []
    scores = []
    feat_imp = pd.DataFrame()
    train_x = train_x.values
    kf =  StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    for fold, (idx_train, idx_valid) in enumerate(kf.split(train_x,y)):
        X_tr, y_tr = train_x[idx_train], y.iloc[idx_train]
        X_val, y_val = train_x[idx_valid], y.iloc[idx_valid]
        model =  ExtraTreesClassifier(n_estimators = n_estimators,
                                    max_depth = max_depth,
                                    min_samples_split = min_samples_split,
                                    min_samples_leaf = min_samples_leaf,
                                    criterion = criterion,
                                    random_state = 42)
        model.fit(X_tr,y_tr)
        val_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, val_pred)
        scores.append(score)
        print(f"Fold: {fold + 1} roc_auc Score is : {score}")
        fold_imp= pd.DataFrame()
        fold_imp["Feature"] = test_x.columns
        fold_imp["importance"] = model.feature_importances_
        fold_imp["fold"] = fold+ 1
        feat_imp = pd.concat([feat_imp, fold_imp], axis=0)
        print('*'*40)
        test_preds = model.predict_proba(test_x)[:, 1]
        predictions.append(test_preds)
    print(f" mean Validation roc_aucis : {np.mean(scores)}")

    # Group and calculate the mean importance for each feature
    plot = feat_imp.groupby("Feature").mean().reset_index()

    # Create and display the barplot
    plt.figure(figsize=(18, 10))
    sns.barplot(x="importance", y="Feature", data=plot.sort_values(by="importance", ascending=False), palette='YlGnBu_r')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title('Feature Importance - Extra Tree')
    plt.tight_layout()
    st.pyplot()



#ExtraTree(train=train)
def corr():
    # Assuming 'column_name' is the problematic column
    trainn = autoclean(train)

    correlation_matrix = trainn.corr()
    st.title('Correlation Heatmap')
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    st.pyplot()

def Distogram():
    nume = ["result"]

    # Create a Streamlit app
    st.title("Distogram Plot")

    # Select the numeric column to plot using a dropdown
    selected_column = st.selectbox("Select a numeric column:", nume)

    # Plot the distogram using Seaborn
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(data=train, x=selected_column, kde=True, color='skyblue', bins=40)
    plt.title(f"Distogram Plot of {selected_column}")
    plt.xlabel(selected_column, fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.gca().set_facecolor('lightgray')
    st.pyplot(plt)
st.title("Autism Analysis")

# Add a section for each visualization
st.header("Mean Autism")
mean_autism()

st.header("Treemap")
Treemap()

st.header("Age Distribution")
ageDistribution()


st.header("Ethnicity")
ethinicity()

st.header("Screening Test")
screeningTest()

st.header("Heatmap")
heatmap()

st.header("Relation Completed")
relationCompleted()

st.header("Screening")
screening()



st.header("Extra Trees Classifier")
ExtraTree(train)

st.header("Correlation Heatmap")
corr()

st.header("Distogram")
Distogram()

st.header("Autism by Country")
Autismcountry()