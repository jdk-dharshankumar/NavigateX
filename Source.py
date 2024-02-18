import cv2
from pyzbar.pyzbar import decode
import csv
import face_recognition
import numpy as np
import os
from datetime import datetime

print("Welcome To Navigate X \t")

def scan_barcodes(frame, scanned_codes):
    try:
        barcodes = decode(frame)
        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            if barcode_data not in scanned_codes:
                print("Registration Number:", barcode_data)
                scanned_codes.append(barcode_data)
    except Exception as e:
        print("Error scanning barcode:", e)
    return scanned_codes

def main(subject_name):  # Add subject_name as an argument
    try:
        # Face recognition setup
        CurrentFolder = os.getcwd()
        image = os.path.join(CurrentFolder, 'rahul.png')
        image2 = os.path.join(CurrentFolder, 'yuva.png')
        image3= os.path.join(CurrentFolder, 'dharsh.png')
        image4 =os.path.join(CurrentFolder, 'ravi.png')
        image5 =os.path.join(CurrentFolder, 'steve.png')
        
        person1_name = "URK23CO2019"
        person1_image = face_recognition.load_image_file(image)
        person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

        person2_name = "URK23CO2028"
        person2_image = face_recognition.load_image_file(image2)
        person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

        person3_name = "URK23AI1150 "
        person3_image = face_recognition.load_image_file(image3)
        person3_face_encoding = face_recognition.face_encodings(person3_image)[0]

        person4_name = "URK23AI1008 "
        person4_image = face_recognition.load_image_file(image4)
        person4_face_encoding = face_recognition.face_encodings(person4_image)[0]

        person5_name = "URK23AI1152 "
        person5_image = face_recognition.load_image_file(image5)
        person5_face_encoding = face_recognition.face_encodings(person5_image)[0]

        known_face_encodings = [
            person1_face_encoding,
            person2_face_encoding,
            person3_face_encoding,
            person4_face_encoding,
            person5_face_encoding,
        ]
        known_face_names = [
            person1_name,
            person2_name,
            person3_name,
            person4_name,
            person5_name
        ]

        face_counts = {name: 0 for name in known_face_names}  # Initialize face count dictionary

        # CSV file setup
        csv_file_name = f"Attendance_{subject_name}.csv"  # Concatenate subject_name with the file name
        csv_file = open(csv_file_name, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Name', 'Registration Number', 'Date', 'Time', 'Verification', 'Face Scanned Count'])  # Add new column

        scanned_codes = []
        last_recognized_face = None

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Failed to open camera.")
            return

        print("Camera opened successfully.")

        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to capture frame.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            if face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
                name = "Not Registered"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    if last_recognized_face != name:
                        scanned_codes = []
                    last_recognized_face = name
                    face_names.append(name)

                    scanned_codes = scan_barcodes(frame, scanned_codes.copy())

                    for barcode_data in list(scanned_codes):
                        if barcode_data == name:
                            verification = "Verified"
                            face_counts[name] += 1  # Update face count
                            csv_writer.writerow([name, barcode_data, datetime.now().date().strftime("%Y-%m-%d"),
                                                 datetime.now().strftime("%H:%M:%S"), verification, face_counts[name]])  # Include face count
                            print("Verified:", name, "Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            scanned_codes.remove(barcode_data)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow('Attendance System', frame)

            key = cv2.waitKey(1)
            if key == ord('c'):
                print("Exiting loop...")
                break

    except Exception as e:
        print("Error occurred:", e)

    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            print("Camera released.")
        cv2.destroyAllWindows()
        print("Windows destroyed.")
        csv_file.close()
        print(f"Data Saved To '{csv_file_name}'.")
        print("Thank You")

if __name__ == "__main__":
    subject_name = input("Enter subject name: ")  # Get subject name from user input
    main(subject_name)
