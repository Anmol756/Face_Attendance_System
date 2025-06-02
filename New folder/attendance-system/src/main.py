import pandas as pd
from datetime import datetime

# File paths
ATTENDANCE_EXCEL_PATH = 'data/attendance_log.xlsx'
STUDENT_DETAILS_EXCEL_PATH = 'data/student_details.xlsx'

def ensure_excel_files_exist():
    # Ensure attendance log Excel file exists
    if not os.path.exists(ATTENDANCE_EXCEL_PATH):
        df = pd.DataFrame(columns=["RollNo", "Name", "Date", "Time"])
        df.to_excel(ATTENDANCE_EXCEL_PATH, index=False)

    # Ensure student details Excel file exists
    if not os.path.exists(STUDENT_DETAILS_EXCEL_PATH):
        df = pd.DataFrame(columns=["RollNo", "Name", "Branch", "Year"])
        df.to_excel(STUDENT_DETAILS_EXCEL_PATH, index=False)

def log_attendance(roll_no, name):
    ensure_excel_files_exist()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")
    df_new_row = pd.DataFrame([{
        'RollNo': roll_no,
        'Name': name,
        'Date': current_date,
        'Time': current_time
    }])
    df_existing = pd.read_excel(ATTENDANCE_EXCEL_PATH)
    already_logged = ((df_existing['RollNo'] == roll_no) & (df_existing['Date'] == current_date)).any()
    if already_logged:
        return False
    df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
    df_combined.to_excel(ATTENDANCE_EXCEL_PATH, index=False)
    return True

def load_student_details():
    df = pd.read_excel(STUDENT_DETAILS_EXCEL_PATH)
    df.set_index('RollNo', inplace=True)
    return df

def main():
    ensure_excel_files_exist()
    student_details_df = load_student_details()
    # Additional logic for attendance management can be added here

if __name__ == "__main__":
    main()