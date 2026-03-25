import os
import sys
import glob

os.environ.pop("LD_LIBRARY_PATH", None)

# Put your conda env lib directory first
conda_prefix = sys.prefix
conda_lib = os.path.join(conda_prefix, "lib")
if os.path.isdir(conda_lib):
    os.environ["LD_LIBRARY_PATH"] = conda_lib

# Preload conda's CUDA libs first
for pattern in [
    "libcublas.so*",
    "libcusolver.so*",
    "libcudart.so*",
]:
    for lib in sorted(glob.glob(os.path.join(conda_lib, pattern))):
        try:
            import ctypes
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

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