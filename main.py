print("DEBUG: face.py script execution started...") # For initial run check
import cv2
import mediapipe as mp
import math
import numpy as np
import face_recognition
import pickle
import os
import pandas as pd
from datetime import datetime, timedelta 
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# --- FILE PATHS ---
ENCODINGS_PATH = 'known_face_encodings.pkl'
STUDENT_DETAILS_EXCEL = 'student_details.xlsx'
ATTENDANCE_EXCEL_PATH = 'attendance_log.xlsx'
KNOWN_FACES_DIR = 'known_faces' 

# --- Global Variables for Loaded Data ---
known_face_encodings = []
known_face_roll_numbers = []
student_details_df = None 

# --- Ensure Excel files exist ---
def ensure_excel_file(file_path, columns):
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=columns)
        try:
            df.to_excel(file_path, index=False)
            print(f"Created empty {file_path}")
        except Exception as e:
            print(f"Error creating {file_path}: {e}")
            messagebox.showerror("File Creation Error", f"Could not create {file_path}.\nPlease check permissions or disk space.")

ensure_excel_file(STUDENT_DETAILS_EXCEL, ["RollNo", "Name", "Branch", "Year"])
ensure_excel_file(ATTENDANCE_EXCEL_PATH, ["RollNo", "Name", "Date", "Time"])

# --- Load student details ---
def load_student_details_globally():
    global student_details_df
    try:
        if not os.path.exists(STUDENT_DETAILS_EXCEL) or os.path.getsize(STUDENT_DETAILS_EXCEL) == 0:
            ensure_excel_file(STUDENT_DETAILS_EXCEL, ["RollNo", "Name", "Branch", "Year"])
        
        df = pd.read_excel(STUDENT_DETAILS_EXCEL)
        if 'RollNo' not in df.columns:
            messagebox.showerror("File Error", f"'RollNo' column missing in {STUDENT_DETAILS_EXCEL}.\nPlease ensure it exists with student data.")
            student_details_df = pd.DataFrame(columns=["RollNo", "Name", "Branch", "Year"]).set_index("RollNo")
            return
        
        df['RollNo'] = df['RollNo'].astype(str)
        df.set_index('RollNo', inplace=True, drop=False) 
        student_details_df = df
        print(f"Loaded student details from '{STUDENT_DETAILS_EXCEL}'.")
    except Exception as e:
        messagebox.showerror("File Error", f"Error loading student details from '{STUDENT_DETAILS_EXCEL}': {e}")
        student_details_df = pd.DataFrame(columns=["RollNo", "Name", "Branch", "Year"])
        if 'RollNo' in student_details_df.columns:
             student_details_df['RollNo'] = student_details_df['RollNo'].astype(str)
             student_details_df.set_index('RollNo', inplace=True, drop=False)
        else: 
            student_details_df = pd.DataFrame(columns=["RollNo", "Name", "Branch", "Year"]).set_index("RollNo")

# --- Encoding Management ---
def load_encodings_globally():
    global known_face_encodings, known_face_roll_numbers
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, 'rb') as f:
                data = pickle.load(f)
                known_face_encodings = data['encodings']
                known_face_roll_numbers = [str(rn) for rn in data['roll_numbers']]
                print(f"Loaded {len(known_face_roll_numbers)} known face encodings.")
        except Exception as e:
            print(f"Error loading encodings file '{ENCODINGS_PATH}': {e}")
            known_face_encodings, known_face_roll_numbers = [], []
            messagebox.showwarning("Encoding Error", f"Could not load face encodings: {e}\nFace recognition may be affected.")
    else:
        print(f"Encodings file '{ENCODINGS_PATH}' not found. Enroll faces to enable recognition.")
        known_face_encodings, known_face_roll_numbers = [], []

load_student_details_globally()
load_encodings_globally()

# --- Attendance Logging Functions ---
def log_attendance(roll_no, name):
    ensure_excel_file(ATTENDANCE_EXCEL_PATH, ["RollNo", "Name", "Date", "Time"])
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")
    new_log_entry = {'RollNo': str(roll_no), 'Name': name, 'Date': current_date, 'Time': current_time}
    
    try:
        df_existing = pd.DataFrame(columns=["RollNo", "Name", "Date", "Time"])
        if os.path.exists(ATTENDANCE_EXCEL_PATH) and os.path.getsize(ATTENDANCE_EXCEL_PATH) > 0:
            try:
                df_existing = pd.read_excel(ATTENDANCE_EXCEL_PATH)
                if 'RollNo' in df_existing.columns: df_existing['RollNo'] = df_existing['RollNo'].astype(str)
                if 'Date' in df_existing.columns: df_existing['Date'] = pd.to_datetime(df_existing['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                if 'Time' in df_existing.columns and not pd.api.types.is_string_dtype(df_existing['Time']):
                    df_existing['Time'] = df_existing['Time'].astype(str)
            except Exception as read_e:
                print(f"Warning: Could not parse existing attendance log '{ATTENDANCE_EXCEL_PATH}': {read_e}")

        student_entries = df_existing[df_existing['RollNo'] == str(roll_no)]
        if not student_entries.empty:
            last_entry = student_entries.iloc[-1]
            if 'Date' in last_entry and pd.notna(last_entry['Date']) and 'Time' in last_entry and pd.notna(last_entry['Time']):
                last_date_str = str(last_entry['Date'])
                if isinstance(last_entry['Date'], (datetime, pd.Timestamp)): last_date_str = last_entry['Date'].strftime('%Y-%m-%d')
                last_time_str = str(last_entry['Time'])
                if isinstance(last_entry['Time'], datetime.time): last_time_str = last_entry['Time'].strftime('%H:%M:%S')
                elif isinstance(last_entry['Time'], pd.Timestamp): last_time_str = last_entry['Time'].strftime('%H:%M:%S')
                try:
                    last_dt_str = f"{last_date_str} {last_time_str}"
                    last_dt = datetime.strptime(last_dt_str, "%Y-%m-%d %H:%M:%S")
                    if (now - last_dt) < timedelta(minutes=50): 
                        # print(f"Attendance for {roll_no} ({name}) not allowed. Last entry at {last_dt_str} (less than 50 mins ago).")
                        return False
                except ValueError as ve:
                    print(f"Warning: Date/Time format error for last entry of {roll_no} ('{last_dt_str}'): {ve}. Allowing new log.")
        
        df_new_row = pd.DataFrame([new_log_entry])
        df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
        df_combined.to_excel(ATTENDANCE_EXCEL_PATH, index=False)
        print(f"Attendance logged: RollNo={roll_no}, Name={name}, Date={current_date}, Time={current_time}")
        return True
    except Exception as e:
        print(f"Error writing to main attendance log: {e}")
        # Fallback attempt
        try:
            pd.DataFrame([new_log_entry]).to_excel(ATTENDANCE_EXCEL_PATH, index=False)
            print(f"Attendance logged (fallback write): RollNo={roll_no}, Name={name}, Date={current_date}, Time={current_time}")
            return True
        except Exception as e2:
            print(f"Critical error writing to main attendance log: {e2}")
        return False

# --- Student Details Management ---
def add_new_student_details(roll_no, name, branch, year):
    global student_details_df
    ensure_excel_file(STUDENT_DETAILS_EXCEL, ["RollNo", "Name", "Branch", "Year"])
    df_current_students = pd.DataFrame(columns=["RollNo", "Name", "Branch", "Year"])
    try:
        if os.path.exists(STUDENT_DETAILS_EXCEL) and os.path.getsize(STUDENT_DETAILS_EXCEL) > 0 :
            df_current_students = pd.read_excel(STUDENT_DETAILS_EXCEL)
        if 'RollNo' not in df_current_students.columns:
            df_current_students = pd.DataFrame(columns=["RollNo", "Name", "Branch", "Year"])
        df_current_students['RollNo'] = df_current_students['RollNo'].astype(str)
    except Exception as e:
         print(f"Error reading {STUDENT_DETAILS_EXCEL} for adding student, initializing fresh: {e}")
         df_current_students = pd.DataFrame(columns=["RollNo", "Name", "Branch", "Year"])
         if 'RollNo' not in df_current_students.columns: df_current_students['RollNo'] = pd.Series(dtype='str')

    if str(roll_no) in df_current_students["RollNo"].values:
        messagebox.showerror("Error", f"Student with RollNo {roll_no} already exists.")
        return False
    
    df_new_student_entry = pd.DataFrame([{"RollNo": str(roll_no), "Name": name, "Branch": branch, "Year": str(year)}])
    df_updated_students = pd.concat([df_current_students, df_new_student_entry], ignore_index=True)
    try:
        df_updated_students.to_excel(STUDENT_DETAILS_EXCEL, index=False)
        load_student_details_globally()
        messagebox.showinfo("Success", f"Added new student details: {roll_no} - {name}")
        return True
    except Exception as e:
        messagebox.showerror("Save Error", f"Could not save student details: {e}")
        return False

def mark_present_manual_action(roll_no):
    global student_details_df
    ensure_excel_file(ATTENDANCE_EXCEL_PATH, ["RollNo", "Name", "Date", "Time"])
    str_roll_no = str(roll_no).strip()
    if not str_roll_no: messagebox.showerror("Input Error", "Roll No cannot be empty."); return

    load_student_details_globally()
    if student_details_df is None:
        messagebox.showerror("Load Error", "Student details could not be loaded. Cannot mark attendance.")
        return
    if str_roll_no not in student_details_df.index:
        messagebox.showerror("Student Not Found", f"No student found with RollNo '{str_roll_no}'. Please add details first.")
        return

    name = student_details_df.loc[str_roll_no].get("Name", str_roll_no)
    if log_attendance(str_roll_no, name):
        messagebox.showinfo("Success", f"Manually marked present for: {str_roll_no} - {name}")
    else:
        messagebox.showwarning("Logging Issue", f"Could not log attendance for {str_roll_no} - {name}.\nThey might have been logged recently (within 50 mins) or an error occurred.\nCheck terminal for details.")

def save_new_face_encoding(roll_no, encoding):
    global known_face_encodings, known_face_roll_numbers
    str_roll_no = str(roll_no)
    known_face_encodings.append(encoding)
    known_face_roll_numbers.append(str_roll_no)
    try:
        with open(ENCODINGS_PATH, 'wb') as f:
            pickle.dump({'encodings': known_face_encodings, 'roll_numbers': known_face_roll_numbers}, f)
        print(f"Encoding saved for RollNo: {str_roll_no}")
        load_encodings_globally() 
        return True
    except Exception as e:
        print(f"Error saving encoding for RollNo {str_roll_no}: {e}")
        messagebox.showerror("Encoding Error", f"Could not save face encoding for {str_roll_no}: {e}")
        return False

# --- Tkinter Panel ---
class AttendancePanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Attendance Management Panel")
        self.geometry("700x600") # Adjusted size
        tab_control = ttk.Notebook(self)

        tab_view = ttk.Frame(tab_control); tab_control.add(tab_view, text="View Attendance")
        tab_add = ttk.Frame(tab_control); tab_control.add(tab_add, text="Manage Students")
        tab_mark = ttk.Frame(tab_control); tab_control.add(tab_mark, text="Mark Present Manually")
        tab_control.pack(expand=1, fill="both", padx=5, pady=5)

        # View Attendance Tab
        tree_frame = ttk.Frame(tab_view)
        tree_frame.pack(expand=True, fill="both", padx=10, pady=10)
        cols = ("RollNo", "Name", "Date", "Time")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings")
        for col_name in cols:
            self.tree.heading(col_name, text=col_name); self.tree.column(col_name, width=150, minwidth=100, anchor='w')
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview); vsb.pack(side='right', fill='y')
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview); hsb.pack(side='bottom', fill='x')
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.pack(expand=True, fill="both")
        self.refresh_attendance()
        btn_refresh = ttk.Button(tab_view, text="Refresh Attendance Log", command=self.refresh_attendance)
        btn_refresh.pack(pady=10)

        # Add Student Tab
        add_frame = ttk.LabelFrame(tab_add, text="Step 1: Enter Student Details")
        add_frame.pack(padx=10, pady=10, fill="x")
        self.entries = {}
        details_fields = {"Roll No": None, "Name": None, "Branch": None, "Year": None}
        for i, (text, _) in enumerate(details_fields.items()):
            ttk.Label(add_frame, text=text + ":").grid(row=i, column=0, padx=5, pady=5, sticky="w")
            entry = ttk.Entry(add_frame, width=40)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
            self.entries[text.replace(":", "").replace(" ", "")] = entry
        add_frame.columnconfigure(1, weight=1)
        btn_add_details = ttk.Button(add_frame, text="Save Student Details", command=self.add_student_details_gui_action)
        btn_add_details.grid(row=len(details_fields), column=0, columnspan=2, pady=(10,5))
        
        enroll_face_frame = ttk.LabelFrame(tab_add, text="Step 2: Enroll Face (After Saving Details)")
        enroll_face_frame.pack(padx=10, pady=(5,10), fill="x") # Corrected pady
        ttk.Label(enroll_face_frame, text="Roll No (for face enrollment):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_enroll_face_roll = ttk.Entry(enroll_face_frame, width=25)
        self.entry_enroll_face_roll.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.btn_capture_enroll = ttk.Button(enroll_face_frame, text="Capture & Enroll Face(s)", command=self.capture_and_enroll_face_gui_action, state="disabled")
        self.btn_capture_enroll.grid(row=0, column=2, padx=10, pady=5)
        enroll_face_frame.columnconfigure(1, weight=1)

        # Mark Present Tab
        mark_frame = ttk.LabelFrame(tab_mark, text="Enter RollNo to Mark Present")
        mark_frame.pack(padx=10, pady=10, fill="x")
        ttk.Label(mark_frame, text="Roll No:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_mark_roll = ttk.Entry(mark_frame, width=40)
        self.entry_mark_roll.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        mark_frame.columnconfigure(1, weight=1)
        btn_mark = ttk.Button(mark_frame, text="Mark Present", command=self.mark_present_gui_action)
        btn_mark.grid(row=1, column=0, columnspan=2, pady=10)

    def refresh_attendance(self):
        for row_item in self.tree.get_children(): self.tree.delete(row_item)
        if os.path.exists(ATTENDANCE_EXCEL_PATH) and os.path.getsize(ATTENDANCE_EXCEL_PATH) > 0:
            try:
                df = pd.read_excel(ATTENDANCE_EXCEL_PATH)
                for col in ["RollNo", "Name", "Date", "Time"]:
                    if col in df.columns: df[col] = df[col].astype(str).replace('nan', '', regex=False)
                for _, r_data in df.iterrows():
                    self.tree.insert("", "end", values=(r_data.get("RollNo",""), r_data.get("Name",""), r_data.get("Date",""), r_data.get("Time","")))
            except Exception as e: messagebox.showerror("Log Error", f"Could not read attendance log: {e}")

    def add_student_details_gui_action(self):
        roll = self.entries["RollNo"].get()
        name = self.entries["Name"].get()
        branch = self.entries["Branch"].get()
        year = self.entries["Year"].get()
        if not (roll and name and branch and year):
            messagebox.showerror("Input Error", "All fields for student details are required.")
            return
        if add_new_student_details(roll, name, branch, year):
            for entry_widget in self.entries.values(): entry_widget.delete(0, tk.END)
            self.entry_enroll_face_roll.delete(0, tk.END)
            self.entry_enroll_face_roll.insert(0, roll)
            self.btn_capture_enroll.config(state="normal")

    def mark_present_gui_action(self):
        roll = self.entry_mark_roll.get()
        if not roll: messagebox.showerror("Input Error", "Roll No is required."); return
        mark_present_manual_action(roll)
        self.entry_mark_roll.delete(0, tk.END)
        self.refresh_attendance()

    def capture_and_enroll_face_gui_action(self):
        roll_no_to_enroll = self.entry_enroll_face_roll.get()
        if not roll_no_to_enroll:
            messagebox.showerror("Input Error", "Roll No for face enrollment is required from Step 2.")
            return
        str_roll_no = str(roll_no_to_enroll)
        load_student_details_globally()
        if student_details_df is None or str_roll_no not in student_details_df.index:
            messagebox.showwarning("Student Not Found", f"Details for Roll No {str_roll_no} not found. Please save details first in Step 1.")
            return 

        num_images_to_capture = 3
        captured_encodings_count = 0
        messagebox.showinfo("Face Capture Starting", f"Prepare to capture {num_images_to_capture} images for Roll No: {str_roll_no}.\nFor each, position face and press 'C' in webcam window.\nTry slight pose variations.")

        cap_enroll = cv2.VideoCapture(0)
        if not cap_enroll.isOpened(): messagebox.showerror("Webcam Error", "Could not open webcam for face capture."); return

        for i in range(num_images_to_capture):
            # messagebox.showinfo("Next Capture", f"Image {i+1} of {num_images_to_capture}.\nPress 'C' to capture, 'Q' to abort this image.")
            capture_window_name = f"Enroll Face ({i+1}/{num_images_to_capture}) for {str_roll_no} - 'C': Capture, 'Q': Skip"
            
            image_captured_for_this_iteration = False
            while not image_captured_for_this_iteration:
                ret, frame = cap_enroll.read()
                if not ret: messagebox.showerror("Webcam Error", "Failed to capture frame."); break
                
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Img {i+1}/{num_images_to_capture}. 'C': Capture | 'Q': Skip", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow(capture_window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    if not os.path.exists(KNOWN_FACES_DIR): os.makedirs(KNOWN_FACES_DIR)
                    student_face_dir = os.path.join(KNOWN_FACES_DIR, str_roll_no)
                    if not os.path.exists(student_face_dir): os.makedirs(student_face_dir)
                    
                    img_name = f"{str_roll_no}_img{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    img_path = os.path.join(student_face_dir, img_name)
                    
                    rgb_frame_for_encoding = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(img_path, frame) 
                    print(f"Image {i+1} saved: {img_path}")
                    
                    try:
                        face_locations_new = face_recognition.face_locations(rgb_frame_for_encoding, model="hog")
                        if face_locations_new:
                            new_encoding = face_recognition.face_encodings(rgb_frame_for_encoding, known_face_locations=face_locations_new)[0]
                            save_new_face_encoding(str_roll_no, new_encoding)
                            captured_encodings_count += 1
                            image_captured_for_this_iteration = True 
                            # cv2.destroyWindow(capture_window_name) # Destroy window after successful capture of this image
                        else:
                            messagebox.showwarning("Face Not Found", f"No face detected in image {i+1}. Please try again. Image not saved.")
                            if os.path.exists(img_path): os.remove(img_path)
                    except Exception as e:
                        messagebox.showerror("Encoding Error", f"Could not process face for image {i+1}: {e}")
                        if os.path.exists(img_path): os.remove(img_path)
                    if image_captured_for_this_iteration: 
                        cv2.destroyWindow(capture_window_name) # Destroy window if successful
                        break # Break inner while to go to next i
                
                elif key == ord('q'):
                    messagebox.showinfo("Capture Skipped", f"Capture for image {i+1} skipped by user.")
                    cv2.destroyWindow(capture_window_name)
                    break 
            if not ret or key == ord('q'): break # If webcam fails or user skipped this image, move to end of enroll process

        cap_enroll.release()
        cv2.destroyAllWindows() # Close any remaining OpenCV windows from enrollment

        if captured_encodings_count > 0:
            messagebox.showinfo("Enrollment Update", f"{captured_encodings_count} face(s) for Roll No {str_roll_no} processed and encodings updated.")
        else:
            messagebox.showwarning("Enrollment Note", f"No new face images were successfully captured and encoded for Roll No {str_roll_no} in this session.")


# --- Face Attendance System (OpenCV part) ---
def run_attendance_system():
    global known_face_encodings, known_face_roll_numbers, student_details_df

    load_encodings_globally() 
    load_student_details_globally() 

    if not known_face_encodings:
        messagebox.showwarning("Setup Incomplete", "No known face encodings. Please enroll students.\nFace recognition will be disabled.")

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
    GENDER_BUCKETS = ["Male", "Female"]
    age_net, gender_net = None, None
    try: 
        age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
        print("Age model loaded.")
    except cv2.error as e: 
        print(f"CV Warning: Age model load failed: {e}")
        messagebox.showwarning("Model Warning", "Age detection model not loaded.")
    try: 
        gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
        print("Gender model loaded.")
    except cv2.error as e: 
        print(f"CV Warning: Gender model load failed: {e}")
        messagebox.showwarning("Model Warning", "Gender detection model not loaded.")

    cap = cv2.VideoCapture(0) # Or try 1, -1
    if not cap.isOpened(): 
        messagebox.showerror("Webcam Error", "Could not open webcam. Check connection or if another app is using it."); return
    print("Webcam opened for face attendance process...")
    padding = 20

    with mp_face_mesh.FaceMesh(
        max_num_faces=5, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: print("Ignoring empty camera frame."); continue

            rgb_frame_flipped = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            output_frame = cv2.cvtColor(rgb_frame_flipped, cv2.COLOR_RGB2BGR)
            img_h, img_w, _ = output_frame.shape
            
            process_frame_rgb_mp = np.copy(rgb_frame_flipped)
            process_frame_rgb_mp.flags.writeable = False
            mp_results = face_mesh.process(process_frame_rgb_mp)
            
            face_locations_fr = face_recognition.face_locations(rgb_frame_flipped, model="hog")
            face_encodings_fr = face_recognition.face_encodings(rgb_frame_flipped, face_locations_fr)
            fr_recognized_bboxes_and_names = {} 

            for face_encoding, face_loc_fr in zip(face_encodings_fr, face_locations_fr):
                student_info_display = "Unknown"
                if known_face_encodings: 
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    if len(face_distances) > 0: 
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index] and face_distances[best_match_index] <= 0.55: 
                            roll_no_recognized_fr = known_face_roll_numbers[best_match_index]
                            name_fr = roll_no_recognized_fr 
                            if student_details_df is not None and str(roll_no_recognized_fr) in student_details_df.index:
                                student_data = student_details_df.loc[str(roll_no_recognized_fr)]
                                name_fr = student_data.get('Name', roll_no_recognized_fr)
                                student_info_display = f"{name_fr} ({roll_no_recognized_fr})"
                                log_attendance(roll_no_recognized_fr, name_fr)
                            else:
                                student_info_display = f"RollNo: {roll_no_recognized_fr} (Details N/A)"
                
                top, right, bottom, left = face_loc_fr
                fr_recognized_bboxes_and_names[(left, top, right, bottom)] = student_info_display 
                cv2.rectangle(output_frame, (left, top), (right, bottom), (0, 0, 255), 2) 
                cv2.putText(output_frame, student_info_display, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            if mp_results.multi_face_landmarks:
                for face_landmarks_mp_obj in mp_results.multi_face_landmarks:
                    landmarks_mp = face_landmarks_mp_obj.landmark
                    mood_text, age_text, gender_text = "Mood: Normal", "Age: N/A", "Gender: N/A"
                    
                    x_coords = [l.x for l in landmarks_mp]; y_coords = [l.y for l in landmarks_mp]
                    x_min_mp_norm = min(x_coords); x_max_mp_norm = max(x_coords)
                    y_min_mp_norm = min(y_coords); y_max_mp_norm = max(y_coords)
                    x_min_mp = int(x_min_mp_norm * img_w); x_max_mp = int(x_max_mp_norm * img_w)
                    y_min_mp = int(y_min_mp_norm * img_h); y_max_mp = int(y_max_mp_norm * img_h)
                    face_x1_mp = max(0, x_min_mp - padding); face_y1_mp = max(0, y_min_mp - padding)
                    face_x2_mp = min(img_w - 1, x_max_mp + padding); face_y2_mp = min(img_h - 1, y_max_mp + padding)
                    
                    # --- Mood Detection ---
                    # IMPORTANT: Replace thresholds with YOUR TUNED VALUES
                    if len(landmarks_mp) > 362: 
                        eye_lm_133_x, eye_lm_133_y = int(landmarks_mp[133].x*img_w), int(landmarks_mp[133].y*img_h)
                        eye_lm_362_x, eye_lm_362_y = int(landmarks_mp[362].x*img_w), int(landmarks_mp[362].y*img_h)
                        eye_distance = math.sqrt(pow(eye_lm_362_x-eye_lm_133_x,2)+pow(eye_lm_362_y-eye_lm_133_y,2))
                        if eye_distance > 0:
                            m_lm_61_x,m_lm_61_y = int(landmarks_mp[61].x*img_w),int(landmarks_mp[61].y*img_h)
                            m_lm_291_x,m_lm_291_y = int(landmarks_mp[291].x*img_w),int(landmarks_mp[291].y*img_h)
                            mouth_width = math.sqrt(pow(m_lm_291_x-m_lm_61_x,2)+pow(m_lm_291_y-m_lm_61_y,2))
                            norm_mouth_w = mouth_width/eye_distance; smile_thresh = 1.45 # YOUR SMILE THRESHOLD

                            lip_u_13_y,lip_l_14_y = int(landmarks_mp[13].y*img_h),int(landmarks_mp[14].y*img_h)
                            lip_center_y = (lip_u_13_y+lip_l_14_y)/2.0
                            avg_mouth_corner_y = (m_lm_61_y+m_lm_291_y)/2.0
                            norm_sad_diff = (avg_mouth_corner_y-lip_center_y)/eye_distance; sad_thresh = 0.030 # YOUR SAD THRESHOLD

                            e_in_l_105_x,e_in_l_105_y = int(landmarks_mp[105].x*img_w),int(landmarks_mp[105].y*img_h)
                            e_in_r_334_x,e_in_r_334_y = int(landmarks_mp[334].x*img_w),int(landmarks_mp[334].y*img_h)
                            in_e_dist = math.sqrt(pow(e_in_r_334_x-e_in_l_105_x,2)+pow(e_in_r_334_y-e_in_l_105_y,2))
                            norm_in_e_dist = in_e_dist/eye_distance
                            e_mid_u_l_107_y,e_mid_u_r_336_y=int(landmarks_mp[107].y*img_h),int(landmarks_mp[336].y*img_h)
                            avg_mid_u_e_y = (e_mid_u_l_107_y+e_mid_u_r_336_y)/2.0
                            eye_top_l_159_y,eye_top_r_386_y=int(landmarks_mp[159].y*img_h),int(landmarks_mp[386].y*img_h)
                            avg_eye_top_y = (eye_top_l_159_y+eye_top_r_386_y)/2.0
                            norm_e_h_vs_eye = (avg_mid_u_e_y-avg_eye_top_y)/eye_distance
                            angry_in_e_dist_thresh = 0.35 # YOUR ANGRY THRESHOLD 1
                            angry_e_h_thresh = 0.050     # YOUR ANGRY THRESHOLD 2

                            if norm_mouth_w > smile_thresh: mood_text = "Mood: Happy"
                            elif norm_in_e_dist < angry_in_e_dist_thresh and norm_e_h_vs_eye < angry_e_h_thresh: mood_text = "Mood: Angry"
                            elif norm_sad_diff > sad_thresh: mood_text = "Mood: Sad"
                    # --- End Mood Detection ---
                    
                    current_blob = None
                    if face_x1_mp < face_x2_mp and face_y1_mp < face_y2_mp:
                        face_roi_mp = output_frame[face_y1_mp:face_y2_mp, face_x1_mp:face_x2_mp]
                        if face_roi_mp.size > 0:
                            current_blob = cv2.dnn.blobFromImage(face_roi_mp, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    
                    if age_net is not None and current_blob is not None: 
                        age_net.setInput(current_blob)
                        age_preds = age_net.forward()
                        if age_preds is not None and len(age_preds[0]) == len(AGE_BUCKETS): age_text = f"Age: {AGE_BUCKETS[age_preds[0].argmax()]}"
                        else: age_text = "Age: Pred Error"
                    elif current_blob is None and age_net: age_text = "Age: ROI Invalid"
                    
                    if gender_net is not None and current_blob is not None: 
                        gender_net.setInput(current_blob)
                        gender_preds = gender_net.forward()
                        if gender_preds is not None and len(gender_preds[0]) == len(GENDER_BUCKETS): gender_text = f"Gender: {GENDER_BUCKETS[gender_preds[0].argmax()]}"
                        else: gender_text = "Gender: Pred Error"
                    elif current_blob is None and gender_net: gender_text = "Gender: ROI Invalid"

                    display_mp_attrs = True
                    mp_face_center_x_px = (x_min_mp_norm + x_max_mp_norm) / 2.0 * img_w
                    mp_face_center_y_px = (y_min_mp_norm + y_max_mp_norm) / 2.0 * img_h
                    for fr_bbox_tuple_key in fr_recognized_bboxes_and_names.keys():
                        fr_left, fr_top, fr_right, fr_bottom = fr_bbox_tuple_key
                        if fr_left < mp_face_center_x_px < fr_right and fr_top < mp_face_center_y_px < fr_bottom:
                           display_mp_attrs = False; break
                    
                    if display_mp_attrs:
                        text_y_pos_mp = face_y1_mp - 10 
                        if text_y_pos_mp < 15 : text_y_pos_mp = y_max_mp + 15 
                        cv2.putText(output_frame, mood_text, (face_x1_mp, text_y_pos_mp), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)
                        text_y_pos_mp += 12
                        cv2.putText(output_frame, age_text, (face_x1_mp, text_y_pos_mp), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1)
                        text_y_pos_mp += 12
                        cv2.putText(output_frame, gender_text, (face_x1_mp, text_y_pos_mp), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0),1)

            cv2.imshow('Attendance System - Press Q to Quit', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    print("Face attendance process finished.")

# --- Main Panel ---
if __name__ == "__main__":
    print("DEBUG: Script entered __main__ block.")
    def start_attendance_threaded_action():
        print("DEBUG: 'Start Face Attendance' menu item clicked.")
        attendance_thread = threading.Thread(target=run_attendance_system, daemon=True)
        attendance_thread.start()

    print("DEBUG: About to create AttendancePanel...")
    try:
        panel = AttendancePanel()
        print("DEBUG: AttendancePanel object created successfully.")
        main_menu = tk.Menu(panel) 
        panel.config(menu=main_menu)
        main_menu.add_command(label="Start Face Attendance", command=start_attendance_threaded_action)
        print("DEBUG: Menu configured.")
        print("DEBUG: About to start panel.mainloop()...")
        panel.mainloop()
        print("DEBUG: panel.mainloop() has exited.") 
    except Exception as e:
        print(f"ERROR in __main__ block (GUI setup or mainloop): {e}")
        import traceback
        traceback.print_exc()