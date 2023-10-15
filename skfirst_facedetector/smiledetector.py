import cv2
from random import randrange

face_detector = cv2.CascadeClassifier(r'D:/Organize these/Library/GITHUB/Data-Science-Notes-n-Projects/Projects/ML-AI_Projects/skfirst_facedetector/haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier(r'D:\Organize these\Library\GITHUB\Data-Science-Notes-n-Projects\Projects\ML-AI_Projects\skfirst_facedetector\smile_cascade.xml')
    

print("video, stream, or webcam?")

user_choice = input().lower()

if user_choice == "video" or user_choice == "stream":
    if face_detector.empty():
        print("Error: Could not load the face detection classifier.")
        exit()
    elif smile_detector.empty():
        print("Error: Could not load the smile detection classifier.")
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
        face_coordinates = face_detector.detectMultiScale(grayscaled_frame)
        
        print("Face Coordinates: ",face_coordinates)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 4)
            the_face = frame[y:y+h, x:x+w]
            grayscaled_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
            smiles = smile_detector.detectMultiScale(grayscaled_face)
            
            #face = frame[y:y+h, x:x+w]
            #face_grayscale we= cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #smile = smile_detector(face_grayscale, 1.7, 28)
            for (x_, y_, w_, h_) in smiles:
                cv2.rectangle(frame, (x_, y_), (x_ + w_, y_ + h_), (randrange(256), randrange(256), randrange(256)), 4)
            
                
            
        # Display the frame with detected faces
        cv2.imshow('SMILE!', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #d close the display window
    vid_stream.release()
    cv2.destroyAllWindows()

else:

    # Check if the classifier loaded successfully
    if face_detector.empty():
        print("Error: Could not load the face detection classifier.")
        exit()
    elif smile_detector.empty():
        print("Error: Could not load the smile detection classifier.")
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
        face_coordinates = face_detector.detectMultiScale(grayscaled_frame, 1.3, 5)
        
        print("Face Coordinates: ",face_coordinates)
        
        # Draw rectangles around the detected faces
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 4)
            the_face = frame[y:y+h, x:x+w]
            grayscaled_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
            smiles = smile_detector.detectMultiScale(grayscaled_face)
            #face = frame[y:y+h, x:x+w]
            #face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        #smile = smile_detector(grayscaled_frame, 1.7, 28)
            
            for (x_, y_, w_, h_) in smiles:
                cv2.rectangle(frame, (x_, y_), (x_ + w_, y_ + h_), (randrange(256), randrange(256), randrange(256)), 4)

        cv2.imshow('SMILE!', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #d close the display window
    wc.release()
    cv2.destroyAllWindows()
    
print("gg")