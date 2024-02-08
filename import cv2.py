import cv2
import face_recognition

# Load an image with faces
image = face_recognition.load_image_file("example.jpg")

# Find all face locations in the image
face_locations = face_recognition.face_locations(image)

# Load a pre-trained face recognition model
face_encodings = face_recognition.face_encodings(image, face_locations)

# Load a sample image for comparison
known_image = face_recognition.load_image_file("example2.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Compare faces in the image with the known face
for face_encoding in face_encodings:
    # Compare the face encoding with the known face encoding
    match = face_recognition.compare_faces([known_face_encoding], face_encoding)

    if match[0]:
        print("Found a matching face!")
    else:
        print("No match found.")

# Display the image with face locations
for top, right, bottom, left in face_locations:
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imshow("Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
