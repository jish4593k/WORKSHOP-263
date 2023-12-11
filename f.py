import cv2
import numpy as np
import os
from datetime import datetime
import face_recognition
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import ttk, messagebox

# Load the MobileNetV2 model for face recognition
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
face_model = Model(inputs=base_model.input, outputs=x)

# Load face images for recognition
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to find face encodings
def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = find_encodings(images)
print('Encoding Complete')

# Function to mark attendance
def mark_attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# ElasticNet Regression
def elastic_net_regression(X, y, alpha=0.5, l1_ratio=0.5):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X, y)
    return model

# Neural Network Regression
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

def train_neural_network(X, y, epochs=1000, lr=0.01):
    input_size = X.shape[1]
    model = NeuralNetwork(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        inputs = torch.tensor(X.values, dtype=torch.float32)
        targets = torch.tensor(y.values, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    return model

# GUI for attendance marking
def create_gui(name, window):
    def on_mark_attendance():
        mark_attendance(name)
        messagebox.showinfo("Attendance Marked", f"Attendance for {name} marked successfully!")

    mark_button = ttk.Button(window, text="Mark Attendance", command=on_mark_attendance)
    mark_button.pack()

# Main video capture loop
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Perform regression prediction using ElasticNet
            # This part may need to be adapted based on your specific regression task
            X_regression = np.array([[x1, y1, x2, y2]])  # Placeholder for face position features
            elastic_net_model = elastic_net_regression(X_regression, np.array([0]))  # Placeholder for labels
            regression_prediction = elastic_net_model.predict(X_regression)[0]
            print(f"ElasticNet Regression Prediction: {regression_prediction}")

            # Perform regression prediction using Neural Network
            # This part may need to be adapted based on your specific regression task
            X_regression_nn = pd.DataFrame([[x1, y1, x2, y2]])  # Placeholder for face position features
            neural_network_model = train_neural_network(X_regression_nn, pd.DataFrame([0]))  # Placeholder for labels
            nn_regression_prediction = neural_network_model(torch.tensor(X_regression_nn.values, dtype=torch.float32)).detach().numpy()[0][0]
            print(f"Neural Network Regression Prediction: {nn_regression_prediction}")

            # Display GUI for marking attendance
            root = tk.Tk()
            root.title("Attendance Marking")

            ttk.Label(root, text=f"Predicted Quality: {regression_prediction}").pack()
            ttk.Label(root, text=f"NN Regression Prediction: {nn_regression_prediction}").pack()

            create_gui(name, root)

            root.mainloop()

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
