import cv2
from random import randrange

print("img, video, stream, or webcam?")

user_choice = input().lower()

if user_choice == str("img"):
    trained_face_data = cv2.CascadeClassifier(r'D:/Organize these/Library/GITHUB/Data-Science-Notes-n-Projects/Projects/ML-AI_Projects/skfirst_facedetector/haarcascade_frontalface_default.xml')
    
    img = cv2.imread(r'D:/Organize these/Library/GITHUB/Data-Science-Notes-n-Projects/Projects/ML-AI_Projects/skfirst_facedetector/rdj.jpg')

    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    print(face_coordinates)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 4)

    cv2.imshow('Inor Nam', img)
    cv2.waitKey()

elif user_choice == "video" or user_choice == "stream":
    trained_face_data = cv2.CascadeClassifier(r'D:/Organize these/Library/GITHUB/Data-Science-Notes-n-Projects/Projects/ML-AI_Projects/skfirst_facedetector/haarcascade_frontalface_default.xml')

    if trained_face_data.empty():
        print("Error: Could not load the face detection classifier.")
        exit()

    vid_stream = cv2.VideoCapture(r'D:/Organize these/Library/GITHUB/Data-Science-Notes-n-Projects/Projects/ML-AI_Projects/skfirst_facedetector/vidtest.mp4')

    if not vid_stream.isOpened():
        print("Error: Could not open the video.")
        exit()

    while True:
        # Read the current frame
        successful_frame_read, frame = vid_stream.read()

        # Check if the frame reading was successful
        if not successful_frame_read:
            print("Error: Could not read a frame from the video capture.")
            break

        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscaled frame
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 4)

        # Display the frame with detected faces
        cv2.imshow('Face Detection', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #d close the display window
    vid_stream.release()
    cv2.destroyAllWindows()

else:
    # Load the pre-trained face detection classifier
    trained_face_data = cv2.CascadeClassifier(r'D:/Organize these/Library/GITHUB/Data-Science-Notes-n-Projects/Projects/ML-AI_Projects/skfirst_facedetector/haarcascade_frontalface_default.xml')

    # Check if the classifier loaded successfully
    if trained_face_data.empty():
        print("Error: Could not load the face detection classifier.")
        exit()

    # Using zero allows the VideoCapture function to use the default webcam.
    # You can change the parameter to a video path if you want to process a video file.
    wc = cv2.VideoCapture(0)

    # Check if the video capture object was successfully opened
    if not wc.isOpened():
        print("Error: Could not open the video capture.")
        exit()

    # Iterate forever over frames
    while True:
        # Read the current frame
        successful_frame_read, frame = wc.read()

        # Check if the frame reading was successful
        if not successful_frame_read:
            print("Error: Could not read a frame from the video capture.")
            break  # Exit the loop if frame capture fails

        # Convert the frame to grayscale for face detection
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscaled frame
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 4)

        # Display the frame with detected faces
        cv2.imshow('Face Detection', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #d close the display window
    wc.release()
    cv2.destroyAllWindows()
    
print("gg")