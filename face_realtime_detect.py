import cv2
import torch
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, UnidentifiedImageError

# Load the pre-trained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=0)

person_dir = "./hcp.jpg"
person_name = "Hoang Cong Phu"
database = {}

def euclidean_distance(embedding1, embedding2):
    return torch.dist(embedding1, embedding2).item()

def find_closest_match(embedding, database, threshold=1.1):
    closest_distance = float('inf')
    closest_name = None
    for name, db_embedding in database.items():
        distance = euclidean_distance(embedding, db_embedding)
        if distance < closest_distance:
            closest_distance = distance
            closest_name = name
    if closest_distance < threshold:
        return closest_name, closest_distance
    else:
        return None, closest_distance


embeddings = []
frame = cv2.imread(person_dir)
embeddings = []
faces, _ = mtcnn.detect(frame)
if faces is not None:
    for face in faces:
        bbox = list(map(int,face.tolist()))
        face_img = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        face_img = cv2.resize(face_img, (160, 160))
        face_img = np.transpose(face_img, (2, 0, 1)) 
        face_img = torch.tensor(face_img, dtype=torch.float32)
        face_img = (face_img - 127.5) / 128.0  
        face_img = face_img.unsqueeze(0) 
        embedding = model(face_img).detach()
        embeddings.append(embedding)
        print(embeddings)
        print(len(embedding))
        database[person_name] = torch.stack(embeddings).mean(dim=0)

print(database)
        
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

while True:
    ret, frame = cap.read()
    if ret == True:
        faces, _ = mtcnn.detect(frame)
        if faces is not None:
            for face in faces:
                bbox = list(map(int,face.tolist()))
                face_box = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
                face_img = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                face_img = cv2.resize(face_img, (160, 160))
                face_img = np.transpose(face_img, (2, 0, 1)) 
                face_img = torch.tensor(face_img, dtype=torch.float32)
                face_img = (face_img - 127.5) / 128.0  
                face_img = face_img.unsqueeze(0) 
                new_embedding = model(face_img).detach() 
                name, _ = find_closest_match(new_embedding, database)
                print(name)
        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()