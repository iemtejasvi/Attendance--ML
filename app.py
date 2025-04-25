import os
import cv2
import torch
import json
import numpy as np
from PIL import Image
from craft_text_detector import Craft
from transformers import (
    TrOCRProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
)
from safetensors.torch import load_file
import craft_text_detector.craft_utils as craft_utils
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import datetime
import difflib
import sqlite3
import uuid

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['SECRET_KEY'] = 'handwriting-attendance-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['STUDENT_DB'] = 'students.csv'
app.config['ATTENDANCE_DB'] = 'attendance.db'

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize SQLite database for attendance records
def init_db():
    conn = sqlite3.connect(app.config['ATTENDANCE_DB'])
    cursor = conn.cursor()
    
    # Create attendance records table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance_records (
        id TEXT PRIMARY KEY,
        date TEXT,
        class_name TEXT,
        image_path TEXT,
        csv_path TEXT,
        present_count INTEGER,
        total_count INTEGER,
        created_at TEXT
    )
    ''')
    
    # Create attendance details table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance_details (
        id TEXT PRIMARY KEY,
        record_id TEXT,
        student_name TEXT,
        roll_number TEXT,
        status TEXT,
        match_type TEXT,
        confidence REAL,
        FOREIGN KEY (record_id) REFERENCES attendance_records (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Load students database
def load_students():
    if os.path.exists(app.config['STUDENT_DB']):
        return pd.read_csv(app.config['STUDENT_DB'])
    return pd.DataFrame(columns=['Name', 'Roll Number'])

# Initialize student database
students_df = load_students()
print(f"✅ Student database loaded with {len(students_df)} records")

# Load models globally
print("🔄 Loading TJ Text Detector...")
craft_model = Craft(output_dir="tj_output", crop_type="box", cuda=(device == "cuda"))
print("✅ TJ Text Detector Loaded.")

print("🔄 Loading TJOCR model...")
tj_ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")

tj_ocr_model_path = "./TJModel/tj_ocr"
tj_ocr_config = VisionEncoderDecoderConfig.from_pretrained(tj_ocr_model_path)
tj_ocr_model = VisionEncoderDecoderModel(config=tj_ocr_config)

tj_ocr_weights = os.path.join(tj_ocr_model_path, "tj_ocr_model.safetensors")
tj_ocr_model.load_state_dict(load_file(tj_ocr_weights, device=device), strict=False)

tj_ocr_model.to(device)  # Move model to GPU
print("✅ TJOCR Model Loaded on", device)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_text(craft_model, image_path):
    result = craft_model.detect_text(image_path)
    if "boxes" in result:
        result["boxes"] = np.array(result["boxes"], dtype=np.int32)
    return result

def match_student(detected_text, student_df):
    """Match detected text with student database using fuzzy matching"""
    best_match = None
    best_score = 0
    match_type = None
    
    # Check if it's a name
    for _, student in student_df.iterrows():
        # Try to match with name
        name_score = difflib.SequenceMatcher(None, detected_text.lower(), student['Name'].lower()).ratio()
        if name_score > best_score and name_score > 0.6:  # 60% similarity threshold
            best_score = name_score
            best_match = student
            match_type = 'Name'
    
    # If no good name match, check roll numbers
    if best_match is None:
        for _, student in student_df.iterrows():
            # Try to match with roll number
            roll_score = difflib.SequenceMatcher(None, detected_text.lower(), student['Roll Number'].lower()).ratio()
            if roll_score > best_score and roll_score > 0.7:  # 70% similarity threshold for roll numbers (stricter)
                best_score = roll_score
                best_match = student
                match_type = 'Roll Number'
    
    if best_match is not None:
        return {
            'matched': True,
            'student': best_match,
            'confidence': best_score,
            'match_type': match_type
        }
    else:
        return {
            'matched': False
        }

def process_attendance_sheet(image_path):
    try:
        craft_result = detect_text(craft_model, image_path)
        detected_boxes = craft_result["boxes"]
        print(f"🔍 Detected {len(detected_boxes)} text regions.")
    except Exception as e:
        print(f"❌ Text Detection failed: {e}")
        return None

    # Load image with PIL
    pil_image = Image.open(image_path).convert("RGB")
    
    # Create a copy of the image for visualization
    vis_image = cv2.imread(image_path)
    
    extracted_text = []
    for idx, box in enumerate(detected_boxes):
        # Fix for box processing - handle nested arrays correctly
        if isinstance(box, np.ndarray) and box.ndim == 2:  # Box is a 2D array with multiple points
            # Extract the corner points (top-left, bottom-right)
            x_min = np.min(box[:, 0])
            y_min = np.min(box[:, 1])
            x_max = np.max(box[:, 0])
            y_max = np.max(box[:, 1])
        elif len(box) == 4:
            x_min, y_min, x_max, y_max = box
        else:
            # Skip invalid boxes
            print(f"⚠️ Skipping invalid box format: {box}")
            continue
            
        # Convert to integers
        try:
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        except (TypeError, ValueError) as e:
            print(f"⚠️ Error converting box coordinates to integers: {e}")
            print(f"Box: {box}")
            continue
            
        # Ensure valid box dimensions
        if x_min >= x_max or y_min >= y_max or x_max <= 0 or y_max <= 0:
            print(f"⚠️ Invalid box dimensions: {x_min}, {y_min}, {x_max}, {y_max}")
            continue
            
        # Make sure coordinates are within image boundaries
        image_width, image_height = pil_image.size
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_width, x_max)
        y_max = min(image_height, y_max)
        
        # Skip too small regions
        if x_max - x_min < 5 or y_max - y_min < 5:
            continue
            
        cropped_region = pil_image.crop((x_min, y_min, x_max, y_max))
        
        try:
            pixel_values = tj_ocr_processor(cropped_region, return_tensors="pt").pixel_values.to(device)
            generated_ids = tj_ocr_model.generate(pixel_values)
            tj_ocr_text = tj_ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            tj_ocr_text = f"⚠ OCR failed: {e}"
        
        # Simple heuristic to classify text
        digit_count = sum(c.isdigit() for c in tj_ocr_text)
        if digit_count > len(tj_ocr_text) / 2:
            text_type = "Roll Number"
        else:
            text_type = "Name"
            
        # Try to match with student database
        student_match = match_student(tj_ocr_text, students_df)
        
        if student_match['matched']:
            matched_student = student_match['student']
            confidence = student_match['confidence']
            match_type = student_match['match_type']
            
            # Add to extracted text list with matching info
            extracted_text.append({
                "box": [x_min, y_min, x_max, y_max],
                "text": tj_ocr_text,
                "type": text_type,
                "matched": True,
                "student_name": matched_student['Name'],
                "student_roll": matched_student['Roll Number'],
                "match_type": match_type,
                "confidence": confidence
            })
            
            # Use different colors based on match type and confidence
            if confidence > 0.8:  # High confidence match
                color = (0, 255, 0)  # Green
            elif confidence > 0.6:  # Medium confidence
                color = (0, 255, 255)  # Yellow
            else:  # Low confidence
                color = (0, 165, 255)  # Orange
                
            # Draw rectangle and text on the visualization image
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Show matched student name and roll
            if match_type == 'Name':
                display_text = f"{matched_student['Name']} ({matched_student['Roll Number']})"
            else:
                display_text = f"{matched_student['Roll Number']} - {matched_student['Name']}"
                
            cv2.putText(vis_image, display_text[:30], 
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        else:
            # No match found
            extracted_text.append({
                "box": [x_min, y_min, x_max, y_max],
                "text": tj_ocr_text,
                "type": text_type,
                "matched": False
            })
            
            # Use red for unmatched text
            color = (0, 0, 255)  # Red
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(vis_image, f"Unmatched: {tj_ocr_text[:20]}", 
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save visualization image
    output_image_path = os.path.join(app.config['RESULTS_FOLDER'], f"output_{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, vis_image)
    
    # Create attendance DataFrame with full student list
    # Start with marking everyone as absent
    attendance_data = []
    
    # First, add all matched students from extracted text
    matched_students = set()
    for item in extracted_text:
        if item.get('matched', False):
            student_name = item['student_name']
            student_roll = item['student_roll']
            
            if (student_name, student_roll) not in matched_students:
                attendance_data.append({
                    'Name': student_name,
                    'Roll Number': student_roll,
                    'Present': 'Yes',
                    'Match Type': item['match_type'],
                    'Confidence': item['confidence']
                })
                matched_students.add((student_name, student_roll))
    
    # Then add all remaining students as absent
    for _, student in students_df.iterrows():
        if (student['Name'], student['Roll Number']) not in matched_students:
            attendance_data.append({
                'Name': student['Name'],
                'Roll Number': student['Roll Number'],
                'Present': 'No',
                'Match Type': '',
                'Confidence': 0.0
            })
    
    # Get unmatched detected text
    unmatched_data = []
    for item in extracted_text:
        if not item.get('matched', False):
            unmatched_data.append({
                'Text': item['text'],
                'Type': item['type']
            })
    
    # Sort attendance data by roll number and name
    attendance_data = sorted(attendance_data, key=lambda x: (x['Roll Number'], x['Name']))
    
    # Create attendance CSV
    output_csv_path = os.path.join(app.config['RESULTS_FOLDER'], f"attendance_{os.path.basename(image_path).split('.')[0]}.csv")
    pd.DataFrame(attendance_data).to_csv(output_csv_path, index=False)
    
    return {
        'attendance_data': attendance_data,
        'unmatched_data': unmatched_data,
        'visualization_path': f"output_{os.path.basename(image_path)}",
        'csv_filename': f"attendance_{os.path.basename(image_path).split('.')[0]}.csv",
        'present_count': sum(1 for item in attendance_data if item['Present'] == 'Yes'),
        'total_count': len(attendance_data)
    }

# Save attendance record to database
def save_attendance_record(result, date, class_name, image_path):
    record_id = str(uuid.uuid4())
    conn = sqlite3.connect(app.config['ATTENDANCE_DB'])
    cursor = conn.cursor()
    
    # Save attendance record
    cursor.execute('''
    INSERT INTO attendance_records 
    (id, date, class_name, image_path, csv_path, present_count, total_count, created_at) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record_id,
        date,
        class_name,
        image_path,
        result['csv_filename'],
        result['present_count'],
        result['total_count'],
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    
    # Save attendance details
    for student in result['attendance_data']:
        detail_id = str(uuid.uuid4())
        cursor.execute('''
        INSERT INTO attendance_details
        (id, record_id, student_name, roll_number, status, match_type, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            detail_id,
            record_id,
            student['Name'],
            student['Roll Number'],
            student['Present'],
            student.get('Match Type', ''),
            student.get('Confidence', 0.0)
        ))
    
    conn.commit()
    conn.close()
    
    return record_id

# Get recent attendance records
def get_recent_records(limit=5):
    conn = sqlite3.connect(app.config['ATTENDANCE_DB'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, date, class_name, csv_path, present_count, total_count, created_at
    FROM attendance_records
    ORDER BY created_at DESC
    LIMIT ?
    ''', (limit,))
    
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return records

# Get student attendance statistics
def get_student_attendance_stats():
    conn = sqlite3.connect(app.config['ATTENDANCE_DB'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get total attendance records count
    cursor.execute('SELECT COUNT(*) as total FROM attendance_records')
    total_records = cursor.fetchone()['total']
    
    if total_records == 0:
        conn.close()
        return []
    
    # Get student attendance stats
    cursor.execute('''
    SELECT 
        student_name as name, 
        roll_number,
        COUNT(*) as total_checks,
        SUM(CASE WHEN status = 'Yes' THEN 1 ELSE 0 END) as present_count
    FROM attendance_details
    GROUP BY student_name, roll_number
    ORDER BY student_name
    ''')
    
    stats = []
    for row in cursor.fetchall():
        attendance_rate = int((row['present_count'] / row['total_checks']) * 100)
        stats.append({
            'name': row['name'],
            'roll_number': row['roll_number'],
            'attendance_rate': attendance_rate,
            'total_checks': row['total_checks'],
            'present_count': row['present_count']
        })
    
    conn.close()
    return stats

# Calculate average attendance rate
def get_avg_attendance_rate():
    conn = sqlite3.connect(app.config['ATTENDANCE_DB'])
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT AVG(CAST(present_count AS FLOAT) / CAST(total_count AS FLOAT) * 100) as avg_rate
    FROM attendance_records
    ''')
    
    result = cursor.fetchone()
    conn.close()
    
    if result[0] is None:
        return 0
        
    return round(result[0], 1)

@app.route('/')
def index():
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    students_count = len(students_df)
    
    # Get total attendance records count
    conn = sqlite3.connect(app.config['ATTENDANCE_DB'])
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM attendance_records')
    attendance_count = cursor.fetchone()[0]
    conn.close()
    
    # Get recent activity
    recent_activity = get_recent_records(5)
    
    return render_template('index.html', 
                          today_date=today_date,
                          students_count=students_count,
                          attendance_count=attendance_count,
                          recent_activity=recent_activity)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    class_name = request.form.get('class_name', 'Unknown Class')
    date = request.form.get('date', datetime.datetime.now().strftime('%Y-%m-%d'))
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the attendance sheet
        result = process_attendance_sheet(file_path)
        
        if result:
            # Save attendance record to database
            record_id = save_attendance_record(result, date, class_name, file_path)
            
            # Add date and class name to the result for display
            result['date'] = date
            result['class_name'] = class_name
            
            return render_template('result.html', **result)
        else:
            flash('Error processing the attendance sheet')
            return redirect(url_for('index'))
    else:
        flash('Invalid file format. Please upload a JPG or PNG image.')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    # Get statistics for dashboard
    students_count = len(students_df)
    
    conn = sqlite3.connect(app.config['ATTENDANCE_DB'])
    cursor = conn.cursor()
    
    # Get attendance records count
    cursor.execute('SELECT COUNT(*) FROM attendance_records')
    attendance_records = cursor.fetchone()[0]
    
    # Get last processed date
    cursor.execute('SELECT date FROM attendance_records ORDER BY created_at DESC LIMIT 1')
    last_processed_result = cursor.fetchone()
    last_processed = last_processed_result[0] if last_processed_result else 'N/A'
    
    conn.close()
    
    # Get average attendance rate
    avg_attendance_rate = get_avg_attendance_rate()
    
    # Get student-wise attendance stats
    student_attendance = get_student_attendance_stats()
    
    # Get recent records
    recent_records = get_recent_records(10)
    
    return render_template('dashboard.html',
                          students_count=students_count,
                          attendance_records=attendance_records,
                          last_processed=last_processed,
                          avg_attendance_rate=avg_attendance_rate,
                          student_attendance=student_attendance,
                          recent_records=recent_records)

@app.route('/view_attendance/<id>')
def view_attendance(id):
    conn = sqlite3.connect(app.config['ATTENDANCE_DB'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get attendance record
    cursor.execute('SELECT * FROM attendance_records WHERE id = ?', (id,))
    record = dict(cursor.fetchone())
    
    # Get attendance details
    cursor.execute('''
    SELECT student_name, roll_number, status, match_type, confidence
    FROM attendance_details
    WHERE record_id = ?
    ORDER BY roll_number, student_name
    ''', (id,))
    
    attendance_data = []
    for row in cursor.fetchall():
        attendance_data.append({
            'Name': row['student_name'],
            'Roll Number': row['roll_number'],
            'Present': row['status'],
            'Match Type': row['match_type'],
            'Confidence': row['confidence']
        })
    
    conn.close()
    
    # Determine unmatched data (not stored in DB, for simplicity we'll return empty)
    unmatched_data = []
    
    return render_template('result.html',
                          attendance_data=attendance_data,
                          unmatched_data=unmatched_data,
                          visualization_path=os.path.basename(record['image_path']),
                          csv_filename=record['csv_path'],
                          present_count=record['present_count'],
                          total_count=record['total_count'],
                          date=record['date'],
                          class_name=record['class_name'])

@app.route('/api/students', methods=['GET'])
def get_students():
    return jsonify(students_df.to_dict('records'))

@app.route('/results/<filename>')
def results_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 