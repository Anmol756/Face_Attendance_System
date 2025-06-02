import face_recognition
import os
import pickle

# Path to the directory containing subdirectories of known faces (named by RollNo)
KNOWN_FACES_DIR = 'known_faces'
# File to save the encodings to
ENCODINGS_PATH = 'known_face_encodings.pkl' # Will store encodings and roll_numbers

print(f"Loading known faces from {KNOWN_FACES_DIR}...")

known_encodings = []
known_roll_numbers = [] # Changed from known_names

# Loop through each person (folder named by RollNo) in the known_faces directory
for roll_no_folder_name in os.listdir(KNOWN_FACES_DIR):
    person_dir_path = os.path.join(KNOWN_FACES_DIR, roll_no_folder_name)
    
    if not os.path.isdir(person_dir_path):
        continue

    # The folder name is the roll number
    roll_no = roll_no_folder_name 
    print(f"Processing images for Roll No: {roll_no}")
    
    for image_name in os.listdir(person_dir_path):
        image_path = os.path.join(person_dir_path, image_name)
        
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image, model="hog") 

            if face_locations:
                encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
                
                if encodings:
                    known_encodings.append(encodings[0])
                    known_roll_numbers.append(roll_no) # Store roll_no
                    print(f"  + Encoded {image_name} for Roll No: {roll_no}")
                else:
                    print(f"  - No face encodings found in {image_name} (Roll No: {roll_no}). Face might be unclear or too small.")
            else:
                print(f"  - No faces found in {image_name} (Roll No: {roll_no}).")
        except Exception as e:
            print(f"  ! Error processing {image_path}: {e}")

print(f"\nFound {len(known_encodings)} encodings.")
if known_encodings:
    print(f"Saving encodings to {ENCODINGS_PATH}...")
    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump({'encodings': known_encodings, 'roll_numbers': known_roll_numbers}, f) # Save roll_numbers
    print("Encodings saved successfully!")
else:
    print("No encodings were generated. Please check your known_faces directory and images.")