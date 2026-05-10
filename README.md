# AeroGuard: UAV Forest Fire Detection System

AeroGuard is a Deep Learning-based web application that detects and segments forest fires in real-time using high-resolution Unmanned Aerial Vehicle (UAV) imagery. Built with Python, TensorFlow, and Streamlit, this tool provides an interactive dashboard to upload drone images, visualize fire heatmaps, and calculate the exact percentage of fire coverage.

## 🚀 Features
- **Deep Learning Model:** Utilizes a custom-trained U-Net architecture to perform pixel-wise classification of fire regions.
- **Interactive Dashboard:** Built with Streamlit for a fast, responsive, and user-friendly interface.
- **Real-Time Heatmaps:** Uses OpenCV to overlay dynamic heatmaps indicating fire intensity and spread.
- **Analytics & Metrics:** Calculates the fire coverage area and displays prediction confidence distributions using Pandas and Matplotlib.

## 🛠️ Tech Stack
- **Machine Learning / Computer Vision:** TensorFlow, Keras, OpenCV
- **Data Science:** NumPy, Pandas, Matplotlib, Seaborn
- **Frontend / Web Framework:** Streamlit
- **Language:** Python

## ⚙️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aeroguard-uav-fire.git
   cd aeroguard-uav-fire
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application locally:
   ```bash
   streamlit run app.py
   ```

## 📸 Usage
1. Adjust the **Fire Threshold** and **Mask Opacity** using the configuration sidebar.
2. Upload one or multiple drone images (`.jpg`, `.png`).
3. The system will automatically process the images and display:
   - The original drone image
   - The predicted binary fire segmentation mask
   - The final fire heatmap overlay
4. Scroll down to view the **Detection Analytics**, which includes distribution graphs and a summary dataframe of the fire coverage across all uploaded images.

## 📂 Project Structure
- `app.py`: Main Streamlit application and UI/dashboard logic.
- `fire_unet_final.h5`: Pre-trained U-Net deep learning model weights.
- `requirements.txt`: List of Python dependencies.
