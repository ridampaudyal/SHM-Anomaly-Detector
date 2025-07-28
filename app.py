import streamlit as st
import numpy as np
import pandas as pd
import joblib # For loading our saved scaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import seaborn as sns

# Keep the console clean by suppressing TensorFlow's verbose warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# --- 1. Load Our AI Brain (and its helpers) ---
# We're caching these heavy assets so they only load once when the app starts.
# Super important for a snappy user experience!
@st.cache_resource
def load_model_and_assets():
    """
    Grabs our pre-trained LSTM Autoencoder model, the data scaler,
    and the training error data needed to figure out what's 'normal'.
    """
    try:
        # Paths to our saved model and preprocessing tools
        model_path = 'model/lstm_autoencoder_shm_anomaly_detector.h5'
        scaler_path = 'model/scaler.pkl'
        train_loss_path = 'model/train_mae_loss.npy'
        n_steps_path = 'model/n_steps.npy'

        # Load the autoencoder. We re-compile it to make sure it's ready to roll
        # and plays nice with the deployment environment's TensorFlow version.
        model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=MeanSquaredError())

        # Load the scaler. This is critical: we must scale new data exactly as training data was scaled.
        scaler = joblib.load(scaler_path)
        # Load the Mean Absolute Error (MAE) from when we trained on 'healthy' data.
        train_mae_loss = np.load(train_loss_path)
        # Grab the sequence length (n_steps) the model expects.
        n_steps = np.load(n_steps_path)[0]

        # Calculate our anomaly threshold. We're using the common 'mean + 3 standard deviations'
        # rule based on the healthy training errors. Anything above this is suspicious!
        threshold = np.mean(train_mae_loss) + 3 * np.std(train_mae_loss)

        return model, scaler, n_steps, threshold

    except Exception as e:
        # If we can't load the core files, the app can't run. Show a clear error and stop.
        st.error(f"🚨 Oops! Couldn't load the model or its assets. Error: {e}")
        st.info("Make sure all model files (`.h5`, `.pkl`, `.npy`) are in the `model/` directory of your GitHub repo.")
        st.stop()

# Let's get everything loaded up when the app starts
model, scaler, n_steps, threshold = load_model_and_assets()

# --- 2. The Engine Room: Data Processing Functions ---

def create_sequences(data, n_steps):
    """
    Transforms flat time-series data into the windowed (sequential) format
    our LSTM model needs.
    """
    sequences = []
    # Ensure our input data is 2D, even if it starts as 1D. Consistency is key for scaling.
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Slide a window of 'n_steps' across the data to create sequences.
    for i in range(len(data) - n_steps + 1):
        sequences.append(data[i:(i + n_steps), 0])
    return np.array(sequences)

def detect_anomaly(raw_data, model, scaler, n_steps, threshold):
    """
    The main anomaly detection pipeline: preprocesses raw data,
    runs it through the autoencoder, and flags anomalies.
    """
    # Scale the incoming raw data using our pre-fitted scaler.
    scaled_data = scaler.transform(raw_data.reshape(-1, 1))

    # Convert the scaled data into the sequences the LSTM model expects.
    sequences = create_sequences(scaled_data, n_steps)

    # Reshape sequences to (number_of_sequences, sequence_length, number_of_features).
    # 'features' is 1 here because we're dealing with single vibration amplitudes.
    sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)

    # Ask the model to reconstruct the sequences.
    reconstructions = model.predict(sequences, verbose=0)

    # Calculate the Mean Absolute Error (MAE) between the original and reconstructed sequences.
    # A big error means the model struggled to reconstruct, indicating something unusual.
    mae_loss = np.mean(np.abs(sequences - reconstructions), axis=1)

    # Compare the error to our threshold. If it's higher, we've found an anomaly!
    anomalies_detected = mae_loss > threshold

    # Squeeze arrays to 1D for easier plotting and DataFrame creation.
    return np.squeeze(mae_loss), np.squeeze(anomalies_detected), np.squeeze(scaled_data)


# ==============================================================================
# 3. Streamlit App: The User-Facing Interface
# ==============================================================================

# Set up the page: title, wide layout, and expanded sidebar for a good first impression.
st.set_page_config(
    page_title="SHM Anomaly Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Intro ---
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
st.markdown("---") # A nice visual separator

# --- Sidebar: Configuration & Info ---
st.sidebar.header("⚙️ Configuration & Model Info")
st.sidebar.info(f"**Anomaly Threshold:** `{threshold:.4f}`\n\n"
                f"This threshold is calculated from the reconstruction errors of the healthy training data (Mean + 3*Std Dev). "
                f"Errors above this value are flagged as anomalies.")
st.sidebar.info(f"**Sequence Length (`n_steps`):** `{n_steps}`") # Display the sequence length
st.sidebar.markdown("---") # Another visual separator

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "⬆️ Upload your vibration data (CSV format, single column)",
    type=["csv"],
    help="Upload a CSV file where the first column contains your numerical vibration data. No headers needed."
)

# --- Main Logic: What happens when a file is uploaded ---
if uploaded_file is not None:
    try:
        # Read the CSV. We're assuming no header row.
        df = pd.read_csv(uploaded_file, header=None)

        # Quick check: Is it really just one column of numbers?
        if df.shape[1] != 1:
            st.error("❌ Error: Your CSV needs to have just one column of numerical data. Please fix and re-upload!")
            st.stop()

        # Grab the raw data and make sure it's float type.
        raw_data = df.iloc[:, 0].values.astype(float)

        # Is there enough data for our sequences?
        if len(raw_data) < n_steps:
            st.error(f"❌ Error: Not enough data! We need at least {n_steps} data points to create meaningful sequences for analysis.")
            st.stop()

        # Time to detect! Show a spinner to let the user know something's happening.
        with st.spinner("⏳ Detecting anomalies... This might take a moment, depending on your data size."):
            mae_loss, anomalies_detected, scaled_data_for_plot = detect_anomaly(raw_data, model, scaler, n_steps, threshold)

        st.success("✅ Anomaly detection complete!")

        # --- Display Key Performance Indicators ---
        col1, col2, col3 = st.columns(3) # Lay out metrics nicely in columns
        num_sequences_analyzed = len(mae_loss)
        num_anomalies_detected = np.sum(anomalies_detected)

        with col1:
            st.metric(
                "Total Sequences Analyzed",
                num_sequences_analyzed,
                help="The total number of time windows (sequences) the model processed."
            )
        with col2:
            st.metric(
                "Anomalous Sequences Detected",
                num_anomalies_detected,
                help="How many sequences had a reconstruction error above our anomaly threshold."
            )
        with col3:
            if num_sequences_analyzed > 0:
                percentage_anomalous = round((num_anomalies_detected / num_sequences_analyzed) * 100, 2)
                st.metric(
                    "Percentage Anomalous",
                    f"{percentage_anomalous}%",
                    help="The percentage of sequences flagged as anomalous."
                )
            else:
                st.metric("Percentage Anomalous", "N/A")

        st.markdown("---") # Another visual break

        # Prepare a DataFrame for plotting. We align the time steps so the plots make sense.
        # Note: The first 'n_steps-1' data points don't have a full sequence output from the model.
        results_df = pd.DataFrame({
            'time_step': np.arange(len(mae_loss)),
            'vibration_scaled': scaled_data_for_plot[n_steps-1:], # Align with sequence output
            'reconstruction_error': mae_loss,
            'is_anomaly': anomalies_detected
        })

        # --- Visualization: Vibration Data with Anomalies Highlighted ---
        st.subheader("📈 Vibration Data with Detected Anomalies")
        fig1, ax1 = plt.subplots(figsize=(15, 6))
        sns.lineplot(data=results_df, x='time_step', y='vibration_scaled', ax=ax1, color='#3498DB', label='Scaled Vibration Data')
        # Overlay red dots for detected anomalies to make them pop!
        anomalies_plot_data = results_df[results_df['is_anomaly']]
        sns.scatterplot(data=anomalies_plot_data, x='time_step', y='vibration_scaled', ax=ax1, color='#E74C3C', s=100, label='Detected Anomaly', zorder=5, marker='o', alpha=0.8)
        ax1.set_title('Vibration Signal with Detected Anomalies', fontsize=16)
        ax1.set_xlabel('Time Steps', fontsize=12)
        ax1.set_ylabel('Scaled Vibration Amplitude', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig1)

        # --- Visualization: Reconstruction Error Over Time ---
        st.subheader("📉 Reconstruction Error Over Time")
        fig2, ax2 = plt.subplots(figsize=(15, 6))
        sns.lineplot(data=results_df, x='time_step', y='reconstruction_error', ax=ax2, color='#2ECC71', label='Reconstruction Error')
        # Draw our anomaly threshold line for easy comparison.
        ax2.axhline(threshold, color='#E74C3C', linestyle='--', label=f'Anomaly Threshold ({threshold:.4f})', linewidth=2)
        # Overlay red dots where the error crossed the threshold.
        anomalies_error_plot_data = results_df[results_df['is_anomaly']]
        sns.scatterplot(data=anomalies_error_plot_data, x='time_step', y='reconstruction_error', ax=ax2, color='#E74C3C', s=100, label='Error Above Threshold', zorder=5, marker='o', alpha=0.8)
        ax2.set_title('Reconstruction Error Over Time', fontsize=16)
        ax2.set_xlabel('Time Steps', fontsize=12)
        ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig2)

        st.info("💡 **Interpretation:** Red points on these graphs mean the model's reconstruction error was unusually high, suggesting a potential anomaly in your vibration data.")

    except Exception as e:
        # Catch any unexpected errors during file processing or analysis
        st.error(f"An unexpected error popped up while processing your file: {e}")
        st.warning("Double-check your CSV: it should contain only numerical data in a single column and be free of corruption.")

else:
    st.info("Upload a CSV file using the button above to analyze your vibration data for anomalies.")
    st.markdown("---")
    st.subheader("Try with Sample Data!")
    st.write("Download a sample CSV file to quickly test the app's functionality.")

    #sample CSV data string for quick testing.
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
