from typing import Concatenate

import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pickle
from torch.distributed.checkpoint import load_state_dict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import os
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

# Uses Nvidia gpu to make things faster if you have one
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 48

# Transforms image to grayscale, correct size, and normalizes
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # for grayscale
])
transformtrain = transforms.Compose([
transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]), # for grayscale
    transforms.RandomHorizontalFlip(p=1.0),
    #transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10)
])

# Datasets
train_dir = 'archiveDataset/train'
test_dir = 'archiveDataset/test'

train_original_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
train_transformed_data = datasets.ImageFolder(root=train_dir,transform=transformtrain)
train_total_dataset = ConcatDataset([train_original_dataset,train_transformed_data])
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_total_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define CNN model, its inheriting
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        # cnn layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1 ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # hidden layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # 4 emotion classes: angry, happy, sad, surprise
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def training():
    # Initialize model
    model = EmotionCNN().to(device)

    # Loss: CrossEntropyLoss uses a mixture of softmax and logloss
    loss_calc = nn.CrossEntropyLoss()
    # adam optimizer tracks previous weight adjustments so that adjustments have momentum
    # it will adjust the learning rate for individual weights based on previous movements
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_calc(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

    # Testing accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # saves new model to be used later without recreating it
    # is saved to the main project folder
    torch.save(model.state_dict(), input("\nName of model"))





# -----------------------------------------
# 2. Feature Extractor (removes last FC)
# -----------------------------------------
class EmotionFeatureExtractor(nn.Module):
    def __init__(self, cnn_model):
        super(EmotionFeatureExtractor, self).__init__()
        self.conv_layers = cnn_model.conv_layers
        self.flatten = nn.Flatten()
        self.fc1 = list(cnn_model.fc_layers.children())[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# -----------------------------------------
# 3. Feature Extraction Function
# -----------------------------------------
def extract_features(dataloader, modelSVM):
    features = []
    labels = []
    modelSVM.eval()
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)
            output = modelSVM(images)
            features.append(output.cpu().numpy())
            labels.append(targets.numpy())
    return np.vstack(features), np.concatenate(labels)

# -----------------------------------------
# 4. Training the CNN
# -----------------------------------------
def train_cnn(train_loader, test_loader, save_path):
    modelSVM = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelSVM.parameters(), lr=0.001)
    num_epochs = 50

    for epoch in range(num_epochs):
        modelSVM.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = modelSVM(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Test Accuracy
    modelSVM.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = modelSVM(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy (CNN only): {100 * correct / total:.2f}%")

    torch.save(modelSVM.state_dict(), save_path)
    return modelSVM

# -----------------------------------------
# 5. Run CNN → Extract Features → Train SVM
# -----------------------------------------
def run_cnn_svm(train_loader, test_loader):
    cnn_save_path = "emotion_cnn_neutral_v2.pth"

    # Step 1: Train CNN
    print("\n[1] Training CNN...")
    modelSVM = train_cnn(train_loader, test_loader, cnn_save_path)

    # Step 2: Load CNN and convert to feature extractor
    print("\n[2] Extracting features from CNN...")
    modelSVM.load_state_dict(torch.load(cnn_save_path))
    feature_model = EmotionFeatureExtractor(modelSVM).to(device)

    # Step 3: Extract features
    X_train_features, y_train = extract_features(train_loader, feature_model)
    X_test_features, y_test = extract_features(test_loader, feature_model)

    # Step 4: Train and evaluate SVM
    print("\n[3] Training SVM on CNN features...")
    param_grid = {
        'C': [.1,1,10,15],
        'gamma': ['scale',.01],
        'kernel': ['rbf']
    }

    svm = SVC()
    grid = GridSearchCV(svm, param_grid, cv=3, verbose=3, n_jobs=-1)
    grid.fit(X_train_features, y_train)

    print("Best SVM params:", grid.best_params_)
    y_pred = grid.predict(X_test_features)

    print("\n[4] SVM Results:")
    print("SVM Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def run_svm(test_loader, mod_pth):
    model_svm = EmotionCNN().to(device)
    model_svm.load_state_dict(torch.load(mod_pth))
    model_svm.eval()

    feature_model = EmotionFeatureExtractor(model_svm).to(device)

    # Step 3: Extract features
    X_train_features, y_train = extract_features(train_loader, feature_model)
    X_test_features, y_test = extract_features(test_loader, feature_model)

    # Step 4: Train and evaluate SVM
    print("\n[3] Training SVM on CNN features...")
    param_grid = {
        'C': [.1,1,10],
        'gamma': ['scale',.01],
        'kernel': ['rbf']
    }

    svm = SVC(probability=True)
    print("Parameter grid:", param_grid)
    grid = GridSearchCV(svm, param_grid, cv=3, verbose=3, n_jobs=-1)
    grid.fit(X_train_features, y_train)

    print("Best SVM params:", grid.best_params_)
    y_pred = grid.predict(X_test_features)

    print("\n[4] SVM Results:")
    print("SVM Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    with open("svm_model_neutral_v2", 'wb') as f:
        pickle.dump(grid.best_estimator_, f)
    print("SVM model saved to:", "svm_model_neutral_v2")

# -----------------------------------------
# Entry point
# -----------------------------------------











def video():
    mod = EmotionCNN().to(device)
    mod.load_state_dict(torch.load("emotion_cnn_neutral_v2.pth"))
    mod.eval()
    feature_model = EmotionFeatureExtractor(mod).to(device)
    with open("svm_model_neutral_v2", 'rb') as f:
        svm_model = pickle.load(f)
    emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]
    # 0 is used for default camera, try 1 if it doesn't work
    camera = cv.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open camera.")
        exit()

    # Use OpenCV's Haar cascade for face detection.
    haarcascade_path = os.path.join(cv.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
    face_cascade = cv.CascadeClassifier(haarcascade_path)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        padding = 50

        for (x, y, a, b) in faces:
            # Expand the detected face box with some padding.
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + a + padding, frame.shape[1])
            y2 = min(y + b + padding, frame.shape[0])

            # Crop the face from the gray scale frame.
            face = gray[y:y + a, x:x + b]
            face_resized = cv.resize(face, (48, 48))  # Resize to match model input size.
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=(0, -1))  # shape becomes (1, 48, 48, 1)
            face_tensor = torch.tensor(face_input, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            # Now face_tensor shape is (1, 1, 48, 48)

            # Extract features using the CNN feature extractor.
            with torch.no_grad():
                features = feature_model(face_tensor)
            features_np = features.cpu().numpy()

            # Get prediction probabilities from the SVM. This requires the model was trained with probability=True.
            probs = svm_model.predict_proba(features_np)[0]
            predicted_index = np.argmax(probs)
            emotion = emotion_labels[predicted_index]

            # Draw the bounding box and the predicted emotion label.
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.putText(frame, emotion, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.9, (0, 255, 0), 2)

            # Draw the confidence levels for each emotion below the bounding box.
            start_y = y2 + 20  # Starting Y coordinate for the text.
            for i, label in enumerate(emotion_labels):
                text = f"{label}: {probs[i] * 100:.1f}%"
                cv.putText(frame, text, (x1, start_y + i * 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show the frame with overlays.
        cv.imshow("Facial Expression Recognition", frame)

        # Press '0' to stop the program.
        if cv.waitKey(100) & 0xFF == ord('0'):
            break

    # Release the camera resources.
    camera.release()
    cv.destroyAllWindows()















while True:
    response = int(input("What would you like do?\n0: Quit\n"
                         "1: Train a new CNN model\n2: Test specific image\n3: Video test\n4: Train a new CNN and SVM\n"))
    match response:
        case 0:
            break
        case 1:
            training()
        case 2:
            model = EmotionCNN().to(device)
            model.load_state_dict(torch.load("256_out_4L_AC83.93"))
            img = Image.open("images/sad.jpg").convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

            # Run the model
            output = model(input_tensor)
            prediction = output.argmax(dim=1)

            print("Predicted class:", prediction.item())
            break
        case 3:
            print("Press the 0 key in the video program to end")
            video()
            break
        case 4:
            run_cnn_svm(train_loader, test_loader)
            break
        case 5:
            run_svm(train_loader,"emotion_cnn_neutral_v2.pth")
            break






