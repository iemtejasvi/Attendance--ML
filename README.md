# ⚠️ VERY IMPORTANT: Clone with Submodules

To clone this repo **with models included**, run this:

```bash
git clone --recursive https://github.com/iemtejasvi/Attendance--ML.git
```

🔁 This ensures the **AI models are downloaded automatically** from Hugging Face into the `models/` folder.  
Then, move them manually:

- Move `tj_doc_model.safetensors` to: `TJModel/tj_doc_parser/`
- Move `tj_ocr_model.safetensors` to: `TJModel/tj_ocr/`

---

# Smart Attendance System

A modern attendance management system that uses AI to detect and process handwritten attendance sheets. Perfect for teachers and educational institutions.

![Smart Attendance System](https://img.shields.io/badge/Smart-Attendance-4361ee)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)
![OCR](https://img.shields.io/badge/OCR-AI%20Powered-orange)

## 🌟 Features

- 📷 **Handwriting Recognition**: Automatically detect and extract names and roll numbers from handwritten attendance sheets  
- 🔍 **Smart Matching**: Fuzzy matching algorithm to handle spelling errors and recognition issues  
- 📊 **Attendance Dashboard**: Visual reports and statistics for monitoring student attendance  
- 📱 **Modern UI**: Clean, responsive interface for a great user experience  
- 📁 **History Management**: Store and access past attendance records  
- 📉 **Trend Analysis**: View attendance patterns over time with interactive charts  
- 📋 **Export Options**: Download attendance data as CSV files  

## 📸 Screenshots

(Add screenshots of your application here)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+  
- CUDA-capable GPU (recommended for faster processing)

### Installation

1. Clone the repository:
   ```bash
   git clone --recursive https://github.com/iemtejasvi/Attendance--ML.git
   cd Attendance--ML
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Move model files as instructed:
   - Move `models/tj_doc_model.safetensors` → `TJModel/tj_doc_parser/`
   - Move `models/tj_ocr_model.safetensors` → `TJModel/tj_ocr/`

5. Run the application:
   ```bash
   python app.py
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## 🛠️ Tech Stack

- **Backend**: Flask  
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5  
- **Database**: SQLite  
- **AI Models**:  
  - Text Detection: CRAFT (Character Region Awareness for Text Detection)  
  - OCR: TrOCR (Transformer-based OCR)  
- **Visualization**: Chart.js  

## 📚 How It Works

1. **Upload**: Teacher uploads a photo of the handwritten attendance sheet  
2. **Processing**:  
   - AI detects text regions in the image  
   - OCR model extracts text from each region  
   - Smart matching algorithm matches extracted text with student database  
3. **Results**: System displays processed attendance with visual indicators  
4. **Storage**: Attendance records are saved to the database for future reference  
5. **Dashboard**: Teachers can view attendance statistics and trends  

## 🧩 Project Structure

```
Attendance--ML/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── students.csv            # Student database
├── uploads/                # Uploaded attendance sheets
├── results/                # Processed results and CSV files
├── TJModel/                # AI model components
│   ├── tj_doc_parser/      # Location for tj_doc_model.safetensors
│   └── tj_ocr/             # Location for tj_ocr_model.safetensors
├── models/                 # Hugging Face submodule (raw .safetensors here)
└── templates/              # HTML templates
    ├── layout.html         # Base template
    ├── index.html          # Home page
    ├── result.html         # Results page
    └── dashboard.html      # Teacher dashboard
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add some amazing feature'`)  
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request  
