import streamlit as st
import nbformat
from nbconvert import HTMLExporter
from nbconvert.writers import FilesWriter
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
st.title("IPython Notebook Code in Streamlit")
ipynb_file = "LabTest_2ML (1).ipynb"
notebook = nbformat.read(ipynb_file, as_version=4)
exporter = HTMLExporter()
body, resources = exporter.from_notebook_node(notebook)
if ipynb_file is not None:
    st.markdown("### IPython Notebook Code:")
    code_html = body
    st.components.v1.html(code_html, height=5000)

