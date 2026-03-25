# streamlit_app.py
import streamlit as st
from pathlib import Path

# local
from auditory_cortex.optimal_stimulus.frontend import Frontend
import auditory_cortex
st.set_page_config(layout="wide", page_title="My App")

home_dir = Path(auditory_cortex.__file__).resolve()
working_dir = home_dir.parents[2] / 'streamlit'
# working_dir = Path('/home/ahmedb/projects/workbook/streamlit').resolve()
ui = Frontend(working_dir)
ui.render()