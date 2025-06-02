import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
STUDENT_DETAILS_EXCEL = 'student_details.xlsx'
ATTENDANCE_EXCEL_PATH = 'attendance_log.xlsx'
MAIN_LOGO_PATH = 'ITM_Logo.png' # Your existing main logo
ANNIVERSARY_LOGO_PATH = 'logo28.jpg' # Path to your new anniversary logo

# --- PAGE SETUP ---
st.set_page_config(page_title="Attendance Report Dashboard", layout="wide")

# --- HEADER SECTION WITH LOGOS AND TITLE ---
# Create two columns: one for the main content (main logo, title), one for the corner logo
header_col1, header_col2 = st.columns([0.85, 0.15]) # Adjust ratio: 85% for main, 15% for corner

with header_col1:
    if os.path.exists(MAIN_LOGO_PATH):
        st.image(MAIN_LOGO_PATH, width=650) # Adjust width as needed for main logo
    else:
        st.warning(f"Main logo image not found at: {MAIN_LOGO_PATH}")
    
    st.title("Attendance Report Dashboard")
    st.write("Institute of Technology & Management, Gwalior")
    st.write(f"Generates reports based on `{ATTENDANCE_EXCEL_PATH}` and `{STUDENT_DETAILS_EXCEL}`.")

with header_col2:
    if os.path.exists(ANNIVERSARY_LOGO_PATH):
        st.image(ANNIVERSARY_LOGO_PATH, width=600) # Adjust width for smaller corner logo
    else:
        st.warning(f"Anniversary logo not found at: {ANNIVERSARY_LOGO_PATH}")

st.markdown("---") # Adds a horizontal line separator

# ... (rest of your existing Streamlit code: load_data, sidebar, report generation, etc.) ...
# Ensure the load_data, all_students_df, attendance_df_full definitions are below this header.

# --- Helper function to load data --- (Keep your existing function)
# Make sure this function is defined before it's called below
def load_data(file_path, is_student_details=False):
    # ... (your existing load_data function code) ...
    if os.path.exists(file_path):
        if os.path.getsize(file_path) > 0:
            try:
                df = pd.read_excel(file_path)
                if 'RollNo' in df.columns:
                    df['RollNo'] = df['RollNo'].astype(str) 
                if is_student_details and 'RollNo' in df.columns:
                    pass # Already handled RollNo above
                elif 'Date' in df.columns: 
                    try:
                        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                    except Exception: # Keep as is if conversion fails
                        pass 
                return df
            except Exception as e:
                st.error(f"Error reading Excel file '{file_path}': {e}")
                return pd.DataFrame() 
        else:
            # st.warning(f"File '{file_path}' is empty.") # Optional: can be noisy
            return pd.DataFrame()
    else:
        st.error(f"File '{file_path}' not found.")
        return pd.DataFrame()

# --- Load initial data ---
all_students_df = load_data(STUDENT_DETAILS_EXCEL, is_student_details=True)
attendance_df_full = load_data(ATTENDANCE_EXCEL_PATH)

# ... (rest of your Streamlit app: sidebar, report generation logic, full attendance log display, refresh button) ...

st.sidebar.header("Report Filters")

# --- Date Selector ---
if not attendance_df_full.empty and 'Date' in attendance_df_full.columns:
    attendance_df_full['Date'] = attendance_df_full['Date'].astype(str)
    available_dates = sorted(attendance_df_full['Date'].unique(), reverse=True)
    if not available_dates:
        selected_date = st.sidebar.date_input("Select a date for report:", datetime.today()).strftime('%Y-%m-%d')
    else:
        selected_date = st.sidebar.selectbox("Select Date for Report:", options=["All Dates"] + available_dates) # Added "All Dates"
else:
    st.sidebar.warning("Attendance log is empty or missing 'Date' column.")
    selected_date = st.sidebar.date_input("Select a date for report:", datetime.today()).strftime('%Y-%m-%d')


if st.sidebar.button("Generate Report for Selected Date"):
    if selected_date:
        st.header(f"Attendance Report for: {selected_date if selected_date != 'All Dates' else 'All Recorded Dates'}")

        if all_students_df.empty or 'RollNo' not in all_students_df.columns:
            st.error(f"Could not load student details or 'RollNo' column missing in '{STUDENT_DETAILS_EXCEL}'. Cannot generate full report.")
        else:
            total_students = len(all_students_df['RollNo'].unique())
            
            if selected_date == "All Dates":
                daily_attendance_df = attendance_df_full # Consider all attendance
                # For summary, you might want to summarize daily stats if "All Dates" is selected
                # For now, this will make "Present" count all unique students ever present.
                # A more meaningful "All Dates" report might be a summary per day or overall participation.
                # Let's adjust this to pick the LATEST date if "All Dates" is too vague for summary
                if available_dates: # if there are dates in the log
                    st.info(f"Showing detailed Present/Absent for the latest available date: {available_dates[0]} when 'All Dates' is selected for summary.")
                    report_date_for_details = available_dates[0]
                    daily_attendance_df_for_details = attendance_df_full[attendance_df_full['Date'] == report_date_for_details]
                else: # No dates in log, can't show present/absent details
                    daily_attendance_df_for_details = pd.DataFrame()

            else: # A specific date is selected
                daily_attendance_df_for_details = attendance_df_full[attendance_df_full['Date'] == selected_date]
            
            if daily_attendance_df_for_details.empty and selected_date != "All Dates":
                st.warning(f"No attendance records found for {selected_date}.")
                present_students_rollnos = set()
            elif not daily_attendance_df_for_details.empty :
                daily_attendance_df_for_details['RollNo'] = daily_attendance_df_for_details['RollNo'].astype(str)
                present_students_rollnos = set(daily_attendance_df_for_details['RollNo'].unique())
            else: # Case for "All Dates" and no records at all
                 present_students_rollnos = set()


            present_count = len(present_students_rollnos)
            absent_count = total_students - present_count if total_students >= present_count else 0 # ensure absent_count is not negative

            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Registered Students", total_students)
            col2.metric(f"Present ({selected_date if selected_date != 'All Dates' else report_date_for_details if available_dates else 'N/A'})", present_count)
            col3.metric(f"Absent ({selected_date if selected_date != 'All Dates' else report_date_for_details if available_dates else 'N/A'})", absent_count)

            st.markdown("---")
            st.subheader(f"Present Students ({selected_date if selected_date != 'All Dates' else report_date_for_details if available_dates else 'N/A'})")
            if present_count > 0:
                present_students_details = all_students_df[all_students_df['RollNo'].isin(list(present_students_rollnos))][['RollNo', 'Name']]
                if not present_students_details.empty:
                    st.dataframe(present_students_details.reset_index(drop=True), use_container_width=True)
                else:
                    st.write("Details for present students could not be fully retrieved.")
            else:
                st.info("No students were marked present on the selected date scope.")
            
            st.markdown("---")
            st.subheader(f"Absent Students ({selected_date if selected_date != 'All Dates' else report_date_for_details if available_dates else 'N/A'})")
            if absent_count > 0:
                all_rollnos = set(all_students_df['RollNo'].astype(str).unique())
                absent_students_rollnos = list(all_rollnos - present_students_rollnos)
                if absent_students_rollnos:
                    absent_students_details = all_students_df[all_students_df['RollNo'].isin(absent_students_rollnos)][['RollNo', 'Name']]
                    if not absent_students_details.empty:
                        st.dataframe(absent_students_details.reset_index(drop=True), use_container_width=True)
                    else: st.write("Could not retrieve full details for all absent students.")
                else: st.info("All registered students were present.")
            elif total_students > 0 : st.info("All registered students were present on the selected date scope.")
            # else: st.info("No registered students to determine absentees.") # Redundant if total_students is 0

    else:
        st.info("Please select a date and click 'Generate Report'.")

st.markdown("---")
st.subheader("Full Attendance Log (Raw Data)")
if not attendance_df_full.empty:
    attendance_df_full_display = attendance_df_full.copy()
    if 'RollNo' in attendance_df_full_display.columns:
        attendance_df_full_display['RollNo'] = attendance_df_full_display['RollNo'].astype(str)
    st.dataframe(attendance_df_full_display, use_container_width=True)
else:
    st.info(f"The main attendance log file '{ATTENDANCE_EXCEL_PATH}' is empty or could not be loaded.")

if st.sidebar.button("Refresh All Data from Files"): # Changed button text slightly
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("This app displays attendance data. To take attendance, run the main Face Attendance System (face.py).")