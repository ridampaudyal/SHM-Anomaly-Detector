import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow warnings for cleaner output in Streamlit
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ==============================================================================
# 1. LOAD MODEL AND ASSETS (Cached for performance)
# ==============================================================================

@st.cache_resource # Cache the model and other assets to avoid reloading on every interaction
def load_model_and_assets():
    """
    Loads the pre-trained LSTM Autoencoder model, MinMaxScaler,
    and the training MAE loss data for threshold calculation.
    """
    try:
        # Define file paths relative to the app.py script
        model_path = 'model/lstm_autoencoder_shm_anomaly_detector.h5'
        scaler_path = 'model/scaler.pkl'
        train_loss_path = 'model/train_mae_loss.npy'
        n_steps_path = 'model/n_steps.npy'

        # Load the artifacts
        # Explicitly compile the loaded model to resolve eager execution issues
        model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=MeanSquaredError())

        scaler = joblib.load(scaler_path)
        train_mae_loss = np.load(train_loss_path)
        n_steps = np.load(n_steps_path)[0] # n_steps was saved as a single-element array

        # Calculate the anomaly threshold (e.g., mean + 3 standard deviations of training errors)
        threshold = np.mean(train_mae_loss) + 3 * np.std(train_mae_loss)

        return model, scaler, n_steps, threshold

    except Exception as e:
        st.error(f"🚨 Error loading model or assets: {e}")
        st.info("Please ensure all model files (`.h5`, `.pkl`, `.npy`) are in the `model/` directory.")
        st.stop() # Stop the app if essential files are missing

# Load the resources once when the app starts
model, scaler, n_steps, threshold = load_model_and_assets()

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def create_sequences(data, n_steps):
    """
    Converts a 1D time series array into 3D sequences for LSTM input.
    Args:
        data (np.array): The 1D time series data.
        n_steps (int): The number of time steps in each sequence.
    Returns:
        np.array: A 3D array of shape (samples, n_steps, 1).
    """
    Xs = []
    # Ensure data is treated as a 2D array for slicing if it's 1D initially
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    for i in range(len(data) - n_steps + 1):
        Xs.append(data[i:(i + n_steps), 0]) # Take n_steps from the 1st column
    return np.array(Xs)

def detect_anomaly(raw_data, model, scaler, n_steps, threshold):
    """
    Processes raw vibration data, makes predictions using the autoencoder,
    and identifies anomalies based on the reconstruction error.
    Args:
        raw_data (np.array): The raw 1D vibration data from the user.
        model (tf.keras.Model): The trained LSTM Autoencoder model.
        scaler (sklearn.preprocessing.MinMaxScaler): The fitted data scaler.
        n_steps (int): The sequence length used by the model.
        threshold (float): The anomaly detection threshold.
    Returns:
        tuple: (mae_loss (1D array), anomalies_detected (1D boolean array), scaled_data (1D array))
    """
    # Reshape and scale the new data using the pre-fitted scaler
    scaled_data = scaler.transform(raw_data.reshape(-1, 1))

    # Create sequences from the scaled data
    sequences = create_sequences(scaled_data, n_steps)

    # Ensure sequences have the correct shape for the model (samples, timesteps, features)
    sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)

    # Make predictions (reconstruction)
    reconstructions = model.predict(sequences, verbose=0)

    # Calculate Mean Absolute Error (MAE) for each sequence
    mae_loss = np.mean(np.abs(sequences - reconstructions), axis=1)

    # Compare error to the threshold to find anomalies
    anomalies_detected = mae_loss > threshold

    # Squeeze to ensure 1-dimensional arrays for DataFrame creation and plotting
    return np.squeeze(mae_loss), np.squeeze(anomalies_detected), np.squeeze(scaled_data)


# ==============================================================================
# 3. STREAMLIT APP UI AND LOGIC
# ==============================================================================

st.set_page_config(
    page_title="SHM Anomaly Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and Introduction ---
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🏗️ Structural Health Monitoring: AI-Powered Anomaly Detection 📊</h1>", unsafe_allow_html=True)
st.markdown(
    """
    Welcome to the SHM Anomaly Detector! This application utilizes a pre-trained **LSTM Autoencoder**
    to identify unusual patterns in time-series vibration data. The model was trained exclusively
    on 'healthy' structural vibration data. When it encounters deviations from these learned
    normal patterns, it flags them as potential anomalies, indicating possible damage or unusual events.

    **Upload your vibration data (CSV with a single column) below to get started!**
    """
)
st.markdown("---")

# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Configuration & Model Info")
st.sidebar.info(f"**Anomaly Threshold:** `{threshold:.4f}`\n\n"
                f"This threshold is calculated from the reconstruction errors of the healthy training data (Mean + 3*Std Dev). "
                f"Errors above this value are flagged as anomalies.")
st.sidebar.info(f"**Sequence Length (`n_steps`):** `{n_steps}`") # Kept the value, removed the explanation
st.sidebar.markdown("---")
# Removed: st.sidebar.write("Developed by a Civil Engineer with a passion for AI. Connect on [LinkedIn](https://www.linkedin.com/in/yourprofile)!") # This line is now removed

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "⬆️ Upload your vibration data (CSV format, single column)",
    type=["csv"],
    help="Upload a CSV file where the first column contains your numerical vibration data. No headers needed."
)

# --- Main Content Area Logic ---
if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file, header=None)

        # Basic data validation
        if df.shape[1] != 1:
            st.error("❌ Error: The CSV file must contain a single column of numerical data.")
            st.stop()

        raw_data = df.iloc[:, 0].values.astype(float)

        if len(raw_data) < n_steps:
            st.error(f"❌ Error: The uploaded data must be at least {n_steps} data points long to create sequences.")
            st.stop()

        # Detect anomalies
        with st.spinner("⏳ Detecting anomalies... This may take a moment based on data size."):
            mae_loss, anomalies_detected, scaled_data_for_plot = detect_anomaly(raw_data, model, scaler, n_steps, threshold)

        st.success("✅ Anomaly detection complete!")

        # --- Display Key Metrics ---
        col1, col2, col3 = st.columns(3)
        num_sequences_analyzed = len(mae_loss)
        num_anomalies_detected = np.sum(anomalies_detected)

        with col1:
            st.metric(
                "Total Sequences Analyzed",
                num_sequences_analyzed,
                help="Number of time windows (sequences) processed by the model."
            )
        with col2:
            st.metric(
                "Anomalous Sequences Detected",
                num_anomalies_detected,
                help="Number of sequences where reconstruction error exceeded the threshold."
            )
        with col3:
            if num_sequences_analyzed > 0:
                percentage_anomalous = round((num_anomalies_detected / num_sequences_analyzed) * 100, 2)
                st.metric(
                    "Percentage Anomalous",
                    f"{percentage_anomalous}%",
                    help="Proportion of sequences identified as anomalous."
                )
            else:
                st.metric("Percentage Anomalous", "N/A")

        st.markdown("---")

        # Create a DataFrame for plotting and displaying results
        # Time axis for plots starts from n_steps-1 because the first sequence ends at index n_steps-1
        results_df = pd.DataFrame({
            'time_step': np.arange(len(mae_loss)),
            'vibration_scaled': scaled_data_for_plot[n_steps-1:], # Align with sequence output
            'reconstruction_error': mae_loss,
            'is_anomaly': anomalies_detected
        })

        # --- Visualization Section ---
        st.subheader("📈 Vibration Data with Detected Anomalies")
        fig1, ax1 = plt.subplots(figsize=(15, 6))
        sns.lineplot(data=results_df, x='time_step', y='vibration_scaled', ax=ax1, color='#3498DB', label='Scaled Vibration Data')
        anomalies_plot_data = results_df[results_df['is_anomaly']]
        sns.scatterplot(data=anomalies_plot_data, x='time_step', y='vibration_scaled', ax=ax1, color='#E74C3C', s=100, label='Detected Anomaly', zorder=5, marker='o', alpha=0.8)
        ax1.set_title('Vibration Signal with Detected Anomalies', fontsize=16)
        ax1.set_xlabel('Time Steps', fontsize=12)
        ax1.set_ylabel('Scaled Vibration Amplitude', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig1)

        st.subheader("📉 Reconstruction Error Over Time")
        fig2, ax2 = plt.subplots(figsize=(15, 6))
        sns.lineplot(data=results_df, x='time_step', y='reconstruction_error', ax=ax2, color='#2ECC71', label='Reconstruction Error')
        ax2.axhline(threshold, color='#E74C3C', linestyle='--', label=f'Anomaly Threshold ({threshold:.4f})', linewidth=2)
        anomalies_error_plot_data = results_df[results_df['is_anomaly']]
        sns.scatterplot(data=anomalies_error_plot_data, x='time_step', y='reconstruction_error', ax=ax2, color='#E74C3C', s=100, label='Error Above Threshold', zorder=5, marker='o', alpha=0.8)
        ax2.set_title('Reconstruction Error Over Time', fontsize=16)
        ax2.set_xlabel('Time Steps', fontsize=12)
        ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig2)

        st.info("💡 **Interpretation:** Points highlighted in red indicate sequences where the model's reconstruction error was significantly higher than expected for 'healthy' data, suggesting a potential anomaly.")

    except Exception as e:
        st.error(f"An unexpected error occurred while processing your file: {e}")
        st.warning("Please ensure your CSV contains only numerical data in a single column.")

else:
    st.info("Upload a CSV file using the button above to analyze your vibration data for anomalies.")
    st.markdown("---")
    st.subheader("Try with Sample Data!")
    st.write("Download a sample CSV file to quickly test the app's functionality.")

    # A much shorter, but still sufficient, sample CSV data string (110 data points)
    sample_data_str = """
0.123
0.150
0.110
0.145
0.130
0.100
0.120
0.135
0.115
0.140
0.125
0.105
0.130
0.118
0.142
0.108
0.128
0.138
0.112
0.140
0.120
0.100
0.130
0.120
0.140
0.115
0.135
0.125
0.145
0.110
0.120
0.130
0.110
0.140
0.120
0.100
0.120
0.135
0.115
0.140
0.125
0.105
0.130
0.118
0.142
0.108
0.128
0.138
0.112
0.140
0.120
0.100
0.130
0.120
0.140
0.115
0.135
0.125
0.145
0.110
0.120
0.130
0.110
0.140
0.120
0.100
0.120
0.135
0.115
0.140
0.125
0.105
0.130
0.118
0.142
0.108
0.128
0.138
0.112
0.140
0.120
0.100
0.130
0.120
0.140
0.115
0.135
0.125
0.145
0.110
0.120
0.130
0.110
0.140
0.120
0.100
0.120
0.135
0.115
0.140
0.125
0.105
0.130
0.118
0.142
0.108
0.128
0.138
0.112
0.140
0.120
0.100
"""
    st.download_button(
        label="Download Sample CSV",
        data=sample_data_str,
        file_name="sample_vibration_data.csv",
        mime="text/csv",
        help="This CSV contains a short sample of vibration data for testing."
    )
