# import cv2
# import numpy as np

# # Load the Haar Cascade Classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load emoji image with transparency (RGBA)
# emoji = cv2.imread('emoji.png', cv2.IMREAD_UNCHANGED)

# # Ensure the emoji has 4 channels (RGBA)
# if emoji.shape[2] == 4:
#     # Extract the alpha channel (transparency)
#     alpha_channel = emoji[:, :, 3]
#     # Create a 3-channel emoji (without alpha)
#     emoji_rgb = emoji[:, :, :3]
# else:
#     emoji_rgb = emoji
#     alpha_channel = np.ones_like(emoji_rgb[:, :, 0]) * 255  # If no alpha, use a full opacity mask

# # Get the webcam feed
# cap = cv2.VideoCapture(0)  # 0 for default camera

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Convert the frame to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
#     # Loop through all detected faces
#     for (x, y, w, h) in faces:
#         # Resize the emoji to fit the detected face
#         emoji_resized = cv2.resize(emoji_rgb, (w, h))
#         alpha_resized = cv2.resize(alpha_channel, (w, h))

#         # Create a mask from the resized emoji alpha channel
#         _, mask = cv2.threshold(alpha_resized, 1, 255, cv2.THRESH_BINARY)
#         mask_inv = cv2.bitwise_not(mask)

#         # Extract the region of interest (ROI) from the frame where the emoji will be placed
#         roi = frame[y:y+h, x:x+w]

#         # Create the background of the emoji in the ROI (black out the area where the emoji will go)
#         frame_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

#         # Take only the emoji region from the emoji image
#         emoji_fg = cv2.bitwise_and(emoji_resized, emoji_resized, mask=mask)

#         # Add the emoji to the frame
#         dst = cv2.add(frame_bg, emoji_fg)

#         # Place the final result back into the frame
#         frame[y:y+h, x:x+w] = dst

#     # Display the resulting frame with emoji overlay
#     cv2.imshow('Video Feed with Emoji Overlay', frame)

#     # Exit the video window by pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture object and close windows
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import requests
# import base64

# # Your API key for Ready Player Me
# API_KEY = "sk_live_wKUBsBTFSmnxEWE4R6JnjzF3VcWg-qrlpGlu"

# # Ready Player Me API endpoint for avatar generation
# API_URL = "https://api.readyplayer.me/v1/avatars/generate"

# # Function to send face image to Ready Player Me API
# def generate_avatar(face_image):
#     # Encode the face image as JPEG
#     _, img_encoded = cv2.imencode('.jpg', face_image)
#     # Convert to base64
#     img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

#     # Create the JSON payload
#     payload = {
#         "image": img_base64,
#         "options": {
#             "type": "fullbody"  # Specify the type of avatar (if applicable)
#         }
#     }

#     headers = {
#         'Authorization': f'Bearer {API_KEY}',
#         'Content-Type': 'application/json',
#     }

#     # Make the API request
#     response = requests.post(API_URL, headers=headers, json=payload)

#     if response.status_code == 200:
#         avatar_url = response.json().get('url')
#         print(f"Avatar URL: {avatar_url}")
#         return avatar_url
#     else:
#         print(f"Error: {response.status_code}, {response.text}")
#         return None

# # Get the webcam feed
# cap = cv2.VideoCapture(0)  # 0 for default camera

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         face = frame[y:y+h, x:x+w]  # Crop the detected face

#         # Send the face to Ready Player Me API to generate an avatar
#         avatar_url = generate_avatar(face)

#         if avatar_url:
#             # Load the avatar from the URL
#             avatar_img = requests.get(avatar_url, stream=True).raw
#             avatar = cv2.imdecode(np.asarray(bytearray(avatar_img.read()), dtype=np.uint8), cv2.IMREAD_COLOR)

#             # Resize the avatar to match the face size
#             avatar_resized = cv2.resize(avatar, (w, h))

#             # Overlay the avatar onto the original frame
#             frame[y:y+h, x:x+w] = avatar_resized

#     cv2.imshow('Video Feed with Avatar Overlay', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# # sk_live_wKUBsBTFSmnxEWE4R6JnjzF3VcWg-qrlpGlu



import cv2
import numpy as np
from fer import FER  # Import FER library for emotion detection

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load different emojis based on emotion
emoji_happy = cv2.imread('emoji.png', cv2.IMREAD_UNCHANGED)
emoji_angry = cv2.imread('emoji_angry.png', cv2.IMREAD_UNCHANGED)
emoji_sad = cv2.imread('emoji_sad.png', cv2.IMREAD_UNCHANGED)
emoji_surprised = cv2.imread('emoji_surprised.png', cv2.IMREAD_UNCHANGED)

# Ensure the emojis have 4 channels (RGBA)
def preprocess_emoji(emoji):
    if emoji.shape[2] == 4:
        alpha_channel = emoji[:, :, 3]
        emoji_rgb = emoji[:, :, :3]
    else:
        emoji_rgb = emoji
        alpha_channel = np.ones_like(emoji_rgb[:, :, 0]) * 255
    return emoji_rgb, alpha_channel

emoji_happy_rgb, alpha_happy = preprocess_emoji(emoji_happy)
emoji_angry_rgb, alpha_angry = preprocess_emoji(emoji_angry)
emoji_sad_rgb, alpha_sad = preprocess_emoji(emoji_sad)
emoji_surprised_rgb, alpha_surprised = preprocess_emoji(emoji_surprised)

# Get the webcam feed
cap = cv2.VideoCapture(0)  # 0 for default camera

# Initialize the emotion detector
emotion_detector = FER()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Extract the face ROI for emotion detection
        face_roi = frame[y:y+h, x:x+w]

        # Get the dominant emotion of the face
        emotion, score = emotion_detector.top_emotion(face_roi)

        # Select the corresponding emoji based on the emotion
        if emotion == 'happy':
            emoji_rgb, alpha_channel = emoji_happy_rgb, alpha_happy
        elif emotion == 'angry':
            emoji_rgb, alpha_channel = emoji_angry_rgb, alpha_angry
        elif emotion == 'sad':
            emoji_rgb, alpha_channel = emoji_sad_rgb, alpha_sad
        elif emotion == 'surprised':
            emoji_rgb, alpha_channel = emoji_surprised_rgb, alpha_surprised
        else:
            emoji_rgb, alpha_channel = emoji_happy_rgb, alpha_happy  #  if the emotion is not recognized or neutral let it be happy

        # Resize the emoji to fit the detected face
        emoji_resized = cv2.resize(emoji_rgb, (w, h))
        alpha_resized = cv2.resize(alpha_channel, (w, h))

        # Create a mask from the resized emoji alpha channel
        _, mask = cv2.threshold(alpha_resized, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Extract the region of interest (ROI) from the frame where the emoji will be placed
        roi = frame[y:y+h, x:x+w]

        # Create the background of the emoji in the ROI (black out the area where the emoji will go)
        frame_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Take only the emoji region from the emoji image
        emoji_fg = cv2.bitwise_and(emoji_resized, emoji_resized, mask=mask)

        # Add the emoji to the frame
        dst = cv2.add(frame_bg, emoji_fg)

        # Place the final result back into the frame
        frame[y:y+h, x:x+w] = dst

    # Display the resulting frame with emoji overlay
    cv2.imshow('Video Feed with Emoji Overlay', frame)

    # Exit the video window by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close windows
cap.release()
cv2.destroyAllWindows()
