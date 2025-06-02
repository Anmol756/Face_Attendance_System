# Attendance System

This project is an attendance management system designed to facilitate the logging of student attendance and management of student details. It utilizes Excel files for data storage and provides a user-friendly interface for interaction.

## Project Structure

```
attendance-system
├── src
│   ├── main.py          # Entry point for the attendance system
│   └── utils.py         # Utility functions for handling Excel files
├── data
│   ├── attendance_log.xlsx  # Logs attendance records
│   └── student_details.xlsx  # Stores student details
├── requirements.txt      # Lists project dependencies
└── README.md             # Documentation for the project
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd attendance-system
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare Excel files**:
   The project includes two Excel files located in the `data` directory:
   - `attendance_log.xlsx`: This file will be used to log attendance records.
   - `student_details.xlsx`: This file will store details of students.

   If these files do not exist, they will be created automatically when the application runs for the first time.

## Usage

To run the attendance system, execute the following command in your terminal:
```
python src/main.py
```

Follow the on-screen instructions to manage student details and log attendance.

## Dependencies

This project requires the following Python packages:
- pandas
- openpyxl
- tkinter
- face_recognition
- opencv-python
- mediapipe

Make sure to install these packages using the `requirements.txt` file provided.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.