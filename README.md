# LLM Neuron Explorer

Public Streamlit dashboard for visualising measured MLP-channel ablations from the private LLM Synaptic Pruning research project.

Deploy on Streamlit Community Cloud with entrypoint `dashboard.py`. Upload `results.csv` from the Colab experiment. The dashboard does not run a model and does not use simulated data.

Importance is the change in reference-answer loss after disabling a channel. Pilot measurements are not evidence of general task ability or validated specialisation.
