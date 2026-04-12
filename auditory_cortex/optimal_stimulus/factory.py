import os
import io
import zipfile
import streamlit as st
from pathlib import Path

# local
from .generator import StimGenerator

@st.cache_resource(show_spinner=False)
def get_generator(dataset_name, model_name, layer_id, mVocs, bin_width, lag, shuffled):
    return StimGenerator(
        dataset_name, model_name, layer_id, 
        mVocs, bin_width, lag, shuffled
        )


def _make_zip(file_infos):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for info in file_infos:
            zf.write(info["path"], arcname=info["filename"])
    zip_buffer.seek(0)
    return zip_buffer

# def zip_directory(dir_path):
#     zip_buffer = io.BytesIO()
#     dir_path = Path(dir_path)

#     with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
#         for file_path in dir_path.glob("*"):
#             if file_path.is_file():
#                 zf.write(file_path, arcname=file_path.name)

#     zip_buffer.seek(0)
#     return zip_buffer

def zip_directory(directory_path):
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Preserve folder structure inside the zip
                arcname = os.path.relpath(file_path, start=directory_path)
                zip_file.write(file_path, arcname)
    
    zip_buffer.seek(0)
    return zip_buffer


# # dummy definition...remove for actual
# import time
# from pathlib import Path

# class StimGenerator:
#     def __init__(self, dataset_name, model_name, layer_id):
#         self.dataset_name = dataset_name
#         self.model_name = model_name
#         self.layer_id = layer_id
#         # heavy loading here
#         for seconds in range(1):
#             time.sleep(1)
        

#     def fit_encoding_model(self, session_id):
#         # returns dict: channel -> corr
#         for seconds in range(1):
#             time.sleep(1)
#         return {
#             "ch_07": 0.82,
#             "ch_03": 0.74,
#             "ch_11": 0.69,
#             "ch_01": 0.55,
#             "ch_02": 0.62,
#             "ch_04": 0.44,
#             "ch_12": 0.39,
#             "ch_06": 0.75,
#             "ch_08": 0.52,
#             "ch_09": 0.14,
#             "ch_14": 0.59,
#             "ch_05": 0.35,
#         }

#     def generate_stimuli(self, out_dir, unit_id, task, duration, n_stimuli, target_audio):
#         # heavy compute here
#         out_dir.mkdir(exist_ok=True)
#         files = []
#         for seconds in range(1):
#             time.sleep(1)
#         for i in range(n_stimuli):
#             f = out_dir / f"{unit_id}_{task}_{i}.wav"
#             f.write_bytes(b"FAKE_WAV_DATA")
#             files.append(f)
#         return files