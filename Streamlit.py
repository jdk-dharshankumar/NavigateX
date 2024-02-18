import pandas as pd
import streamlit as st
import plotly.express as px 


st.set_page_config(page_title ='DigiTronix')
st.title('Attendence Survey By NavigateX')

csv_file_path = (r"Desktop\NavigateX\Attendance_python.csv")


try:
    df = pd.read_csv(csv_file_path)
    st.write(df)
except FileNotFoundError:
    st.error("File not found. Please check the file path.")
except Exception as e:
    st.error(f"An error occurred: {e}")
