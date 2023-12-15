import cv2
import numpy as np
import pickle
from tensorflow import keras

def preprocess_image(img):
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Invert the image if necessary
    if np.mean(img) > 127:
        img = 255 - img

    # Normalize the image
    img = img / 255.0

    # Reshape the image to fit the model input
    img = img.reshape(1, 28, 28)

    return img

def predict_digit(img, model):
    processed_image = preprocess_image(img)
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction[0])
    return predicted_label


# Load your trained model
with open('mnsit.pkl', 'rb') as f:
    model = pickle.load(f)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Display the captured frame
        cv2.imshow('Frame', frame)

        # Press 'p' to predict the digit in the frame
        # if cv2.waitKey(1) & 0xFF == ord('p'):
        #     predicted_label = predict_digit(frame, model)
        #     print(f"Predicted Label: {predicted_label}")
        predicted_label = predict_digit(frame, model)
            # Overlay the prediction on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Predicted: {predicted_label}', (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame with prediction
        cv2.imshow('Frame', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
