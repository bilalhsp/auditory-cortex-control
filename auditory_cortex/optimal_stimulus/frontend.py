from typing import Any
import streamlit as st


# from auditory_cortex.optimal_stimulus.factory import get_generator

import auditory_cortex.optimal_stimulus.factory as factory


class Frontend:
    def __init__(self, working_dir):

        self.working_dir = working_dir

        self._init_session_state()
        self.max_table_height = 250
        self.max_table_rows = 5

    def _get_generator(self):
        cfg = st.session_state.generator_config
        return factory.get_generator(
            cfg["dataset_name"], cfg["model_name"], cfg["layer_id"],
            cfg["mVocs"], cfg["bin_width"], cfg["lag"], cfg["shuffle"],
        )
    
    def render(self):
        self._render_header()
        self._render_sidebar()
        self._render_main()

    def _render_main(self):
        
        self._experiment_summary()
        # self._upload_and_fit()
        self._upload_session()
        self._fit_encoding_model()
        self._display_correlations()
        self._stimulus_generation()
        self._download_generated_clips()


    def _render_header(self):
        st.title("Optimal Stimulus Generator")
        st.caption("Upload → Fit → Generate")

        with st.expander("See full workflow"):
            st.write(
                "Initialize experiment → Upload session → Fit model → "
                "Select channel → Generate stimuli"
            )


    def _init_session_state(self):
        defaults = {
            "generator_ready": False,
            "corr_ready": False,
            "generator": None,
            "corr_dict": None,
            "session_file_path": None,
            "experiment_config": None,
            "rec_session_name": None,
        }

        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def _render_sidebar(self):
        datasets_supported = ["ucdavis", "ucdavisAct", "ucdavisBAK"]
        encoding_models_supported = ["deepspeech2"]
        dnn_layer_ids = [0, 1, 2]
        with st.sidebar:
            st.header("Experiment setup")
            with st.form("init_form"):
                stimulus_set = st.selectbox(
                    "Stimulus type",
                    ["mVocs", "TIMIT"],
                    index=1
                )
                mVocs = (stimulus_set == "mVocs")

                # dataset_name = st.radio("Dataset", datasets_supported, index=2)
                dataset_name = st.selectbox("Dataset", datasets_supported, index=2)
                model_name = st.selectbox("Encoding model", encoding_models_supported, index=0)
                layer_id = st.selectbox("Layer ID", dnn_layer_ids, index=2)

                bw_options = {
                    "20 ms": 20,
                    "50 ms (default)": 50,
                    "100 ms": 100,
                }
                bw_label = st.selectbox("Bin width", list(bw_options.keys()), index=1, disabled=True)
                bin_width = bw_options[bw_label]

                lag_options = {
                    "100 ms": 100,
                    "200 ms (default)": 200,
                    "300 ms": 300,
                }

                lag_label = st.selectbox("Receptive field", list(lag_options.keys()), index=1, disabled=True)
                lag = lag_options[lag_label]
                shuffle = st.toggle("Use untrained model", value=False, disabled=True)
                init_submitted = st.form_submit_button("Start")


        if init_submitted:
            # experiment config: used for rendering
            st.session_state.experiment_config = {
                "Stimulus type": stimulus_set,
                "Dataset": dataset_name,
                "Encoding model": model_name,
                "Layer": layer_id,
                "Bin width": f"{bin_width} ms",
                "Receptive field": f"{lag} ms",
            }

            # generator config: used for creating/generator object
            st.session_state.generator_config = {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "layer_id": layer_id,
                "mVocs": mVocs,
                "bin_width": bin_width,
                "lag": lag,
                "shuffle": shuffle,
            }
            with st.spinner("Loading models and features..."):
                generator = self._get_generator()

            st.session_state.generator_ready = True



    

    def _experiment_summary(self):
        if not st.session_state.generator_ready:
            st.info("Configure experiment in the sidebar and click Start.")
            return

        st.success("Experiment ready. You may start recording neural data.")
        if st.session_state.generator_ready and st.session_state.experiment_config is not None:
            st.subheader("Experiment summary")
            cfg = st.session_state.experiment_config
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Stimulus type**  \n{cfg['Stimulus type']}")
                    st.markdown(f"**Dataset**  \n{cfg['Dataset']}")
                with col2:
                    st.markdown(f"**Encoding model**  \n{cfg['Encoding model']}")
                    st.markdown(f"**Layer**  \n{cfg['Layer']}")
                with col3:
                    st.markdown(f"**Bin width**  \n{cfg['Bin width']}")
                    st.markdown(f"**Receptive field**  \n{cfg['Receptive field']}")


    def _upload_session(self):
        st.header("Session analysis")
        with st.container(border=True):
            st.caption(
                "Upload a recording session or select an existing one, then fit the encoding model "
                "to estimate channel-wise correlations."
            )

            enabled = (
                st.session_state.generator_ready
                and st.session_state.experiment_config is not None
            )
            tab_choice = st.radio(
                "Session source",
                ["Upload session", "Existing sessions"],
                horizontal=True,
                key="session_source_tab",
            )

            if tab_choice == "Upload session":
                uploaded_file = st.file_uploader(
                        "Session file",
                        label_visibility="collapsed",
                        disabled=not enabled,
                        key="session_upload",
                    )

                if uploaded_file is not None:
                    tmp_dir = self.working_dir / "tmp_uploads"
                    tmp_dir.mkdir(exist_ok=True, parents=True)

                    file_path = tmp_dir / uploaded_file.name
                    file_path.write_bytes(uploaded_file.getbuffer())

                    st.session_state.session_file_path = str(file_path)
                    st.session_state.session_source = "upload"
                    st.success(f"Loaded uploaded session: {uploaded_file.name}")

            elif tab_choice == "Existing sessions":
                if not enabled:
                        st.info("Prepare the generator and experiment config first.")
                else:
                    # Replace this with your backend call
                    generator = self._get_generator()
                    sessions_dict = generator.get_available_sessions()

                    if len(sessions_dict) == 0:
                        st.info("No existing sessions available.")
                    else:
                        session_options = {
                            f'Session {sess_id} ({sess_name})' : sess_id 
                            for sess_id, sess_name in sessions_dict.items()
                        }

                        selected_label = st.selectbox(
                            "Select a session",
                            options=list(session_options.keys()),
                            index=None,
                            placeholder="Choose an existing session",
                            key="existing_session_select",
                        )

                        if selected_label is not None:
                            session_id = session_options[selected_label]
                            st.session_state.session_file_path = session_id
                            st.session_state.session_source = "existing"
                            st.success(f"Selected existing session: {selected_label}")

    def _fit_encoding_model(self):
        fit_clicked = st.button(
            "Fit the encoding model",
            use_container_width=True,
            disabled=(st.session_state.session_file_path is None),
        )
        if fit_clicked:
            with st.spinner("Fitting the encoding model..."):
                generator = self._get_generator()
                session_name, corr = generator.fit_encoding_model(
                    st.session_state.session_file_path,
                    st.session_state.session_source
                )

                corr = dict(sorted(corr.items(), key=lambda x: x[1], reverse=True))
                st.session_state.corr_dict = corr
                st.session_state.corr_ready = True
                st.session_state.rec_session_name = session_name

            st.success("Encoding model fit complete.")

    def _display_correlations(self):
        if st.session_state.corr_ready:
            st.subheader("Channel correlations")

            corr_items = list(st.session_state.corr_dict.items())
            corr_rows = [
                {
                    "Rank": i + 1,
                    "Unit ID": ch,
                    "Correlation (\u03C1)": f"{corr:.3f}",
                }
                for i, (ch, corr) in enumerate(corr_items)
            ]

            n = len(corr_rows)
            # dynamic height of container
            if n <= self.max_table_rows:
                height = None   # auto → fits content
            else:
                height = self.max_table_height    # max cap
            with st.container(border=True, height=height):
                top_ch, top_corr = corr_items[0]
                c1, c2 = st.columns([1, 3])

                with c1:
                    st.markdown("**Best channel**")
                    st.markdown(f"### {top_ch}")
                    st.caption(f"Correlation: {top_corr:.3f}")

                with c2:
                    st.dataframe(corr_rows, use_container_width=True, hide_index=True)

    def _stimulus_generation(self):
        st.header("Stimulus generation")

        if not st.session_state.corr_ready:
            if st.session_state.generator_ready:
                st.info("Fit the encoding model to enable stimulus generation.")
            else:
                st.info("Complete experiment setup and session analysis first.")
        else:
            st.caption("Select a channel and generate stimuli.")
            ch_labels = {
                f"{ch} (\u03C1={st.session_state.corr_dict[ch]:.3f})":ch
                for ch in list(st.session_state.corr_dict.keys())
            }
            task_labels = {
                "maximize firing rate (default)": "stretch",
                "maximize one, suppress others": "one-hot",
            }
            duration_labels = {
                "1": 1,
                "2 (default)": 2,
                "3": 3,
                "4": 4,
            }
            n_stimuli_labels = {
                "1": 1,
                "2": 2,
                "4 (default)": 4,
                "8": 8,
                "10": 10,
            }

            with st.form("generate_form"):
                col1, col2 = st.columns(2)
                with col1:
                    unit_label = st.selectbox("Choose unit", list(ch_labels.keys()), index=0)
                    duration_label = st.selectbox("Duration (sec)", duration_labels, index=1)
                with col2:
                    task_label = st.selectbox("Task", task_labels, index=0)
                    n_stimuli_label = st.selectbox("Number of stimuli", n_stimuli_labels, index=2)

                gen_submitted = st.form_submit_button("Generate stimuli")

            if gen_submitted:
                unit_id = ch_labels[unit_label]
                task = task_labels[task_label]
                duration = duration_labels[duration_label]
                n_stimuli = n_stimuli_labels[n_stimuli_label]
                with st.spinner("Generating stimuli..."):
                    generator = self._get_generator()
                    output_dir = self.working_dir / 'outputs' / st.session_state.rec_session_name
                    saved_clips = generator.generate_and_save_stimuli(
                        output_dir,
                        unit_id=unit_id,
                        task=task,
                        duration=duration,
                        n_stimuli=n_stimuli,
                        target_audio=n_stimuli,
                    )
                st.success(f"{n_stimuli} Stimuli generated for unit: '{unit_id}'")
                st.session_state.generated_clips = saved_clips
                st.session_state.generated_output_dir = str(output_dir)

        

    def _download_generated_clips(self):

        output_dir = st.session_state.get("generated_output_dir")

        if output_dir:
            zip_buffer = factory.zip_directory(output_dir)

            st.download_button(
                label="Download all the generated clips",
                data=zip_buffer,
                file_name=f"{st.session_state.rec_session_name}_clips.zip",
                mime="application/zip",
                key="download_all_clips",
            )
