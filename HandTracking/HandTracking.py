import cv2
import mediapipe as mp
import time

# Acess to the webcam
cap = cv2.VideoCapture(0)

# Hand model
handModel = mp.solutions.hands
# static_image_mode=False (useful to work with live stream) max_num_hands=2 (Number of hands detection)
# min_detection_confidence=0.5 (If the score is lower than 0.5, the model will not consider the target as a hand)
# min_tracking_confidence=0.5 (After the target have been detected as a hand, the system will track it. If the score is lower than 0.5, the model will do the hand detection process again)
hands = handModel.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Hand landmarks model
# Module used to draw the landmarks of a hand
handLandmarksModel = mp.solutions.drawing_utils

# Previous and current time
pTime = 0
cTime = 0

while True:
    # Read a frame
    sucess, img = cap.read()

    # Convert the frame (image) to a RGB format to work with it
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the object (frame/image) for hand detection
    results = hands.process(imgRGB)

    # Extract the information we had processed from the object (frame/image)
    # First of all, we check if the detection model worked
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        # We have to check if we have multiple hands. If this were the case, we extract the information one by one
        # hand_landmarks will represent a hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Based on the landmarks, we have identifiers for each part of the hand
            # Each identifiers have its own coordinates to locate it.
            # We will refer to the identifier as id and the coordinates (x,y,z) as a landmark that we will treat it like an object
            # The coordinates are given in decimal values and they indicates a ratio (porcentage) of the image.
            # Suppose we have a landmark with coordinates (landmark.x, landmark.y) = (0.5, 0.3), then this would means it’s halfway across the width and 30% down from the top.
            # If we multiply for the height and the width, the we will get the value in pixels
            for id, landmark in enumerate(hand_landmarks.landmark):
                # print(id, landmark)
                # We get the height and width of the image. Moreover, we also get the channels. In our case, we are working with a BGR image,then we have three channels
                # Fun fact: RGB (Red, Green, Blue) and BGR (Blue, Green, Red) are similar,but not the same, in particular, the order is not the same.
                # The reason why OpenCV use BGR is due to in the past we used to use this format, but not now. However, OpenCV decided to continue using this format
                h, w, c = img.shape
                # We get the position (pixels). We have to casting to an integer
                px, py = int(landmark.x*w), int(landmark.y*h)
                #print(id, px, py)
                # Let's say that we will print a circle over the landmark 8
                if id == 4:
                    cv2.circle(img, (px, py), 15, (255,0,255), cv2.FILLED)
                if id == 8:
                    cv2.circle(img, (px, py), 15, (255,0,255), cv2.FILLED)
                if id == 12:
                    cv2.circle(img, (px, py), 15, (255,0,255), cv2.FILLED)
                if id == 16:
                    cv2.circle(img, (px, py), 15, (255,0,255), cv2.FILLED)
                if id == 20:
                    cv2.circle(img, (px, py), 15, (255,0,255), cv2.FILLED)

            # Based on the information received from the detection hand model, we draw the landmarks of the hand
            # If we add the handModel.HAND_CONNECTIONS, then we will also show the connections between the landmarks
            handLandmarksModel.draw_landmarks(img, hand_landmarks, handModel.HAND_CONNECTIONS)

    # Show the frame per second (FPS)
    # time function return a value in seconds
    # cTime - pTime give us the time taken to process one frame
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # FPS are expressed with decimals, then we casting to an integer and convert it to a string variable
    # (10, 70) Specifies the position (coordinates) where the text will be placed. In this case, it’s at pixel coordinates (10, 70) from the top-left corner of the image.
    # cv2.FONT_HERSHEY_PLAIN: Specifies the font type for the text (in this case, a simple plain font). When we use this font type, the text appears without any additional styling (such as bold or italic). It’s a straightforward and simple font choice.
    # 3: The font scale (size) of the text.
    # 255, 0, 255): The color of the text in BGR format (here, it’s a shade of magenta)
    # 3: The thickness of the text. Indicates how bold or thick the text appears when added to the image
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 255), 3)

    # Show the frame
    cv2.imshow('img', img)

    # Wait 1ms for a key press
    # If that time pass, then take the next frame
    # If x <= 0 in waitKey, then it waits indefinitely for a key press
    cv2.waitKey(1)
