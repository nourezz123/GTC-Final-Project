import os
import cv2
import numpy as np
import face_recognition
import pickle

EMBEDDING_FILE = "face_embeddings.pkl"


def extract_embeddings_from_folder(dataset_dir):
    embeddings = []
    labels = []

    print(f"Loading images from: {dataset_dir}")
    for person_name in os.listdir(dataset_dir):
        person_folder = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"Processing person images: {person_name}")
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)

            if len(encodings) == 0:
                print(f"No face in photo: {image_path}")
                continue

            embeddings.append(encodings[0])
            labels.append(person_name)

    print(f"extracted {len(embeddings)} embedding")
    return embeddings, labels


def save_embeddings(embeddings, labels):
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, 'rb') as f:
            existing = pickle.load(f)

        existing_embeddings = np.array(existing['embeddings'])
        existing_labels = np.array(existing['labels'])
        new_embeddings = np.array(embeddings)
        new_labels = np.array(labels)

        embeddings = np.concatenate([existing_embeddings, new_embeddings])
        labels = np.concatenate([existing_labels, new_labels])
    else:
        embeddings = np.array(embeddings)
        labels = np.array(labels)

    with open(EMBEDDING_FILE, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, f)
    print("The model and encoder label have been saved")


def load_embeddings():
    if not os.path.exists(EMBEDDING_FILE):
        print("No embeddings file found")
        return [], []
    with open(EMBEDDING_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['labels']




def predict_images_in_folder(test_dir, known_embeddings, known_labels, tolerance=0.5):
    print(f"Testing images from: {test_dir}")
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

    y_true, y_pred = [], []

    for image_name in image_files:
        image_path = os.path.join(test_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        if len(encodings) == 0:
            continue

        for (box, encoding) in zip(boxes, encodings):
            distances = face_recognition.face_distance(known_embeddings, encoding)
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]

            if best_distance < tolerance:
                name = known_labels[best_match_index]
                confidence = (1.0 - best_distance) * 100
                if confidence < 30:
                    name = "Unknown"
            else:
                name = "Unknown"

            y_true.append(os.path.basename(os.path.dirname(image_path)))
            y_pred.append(name)

            top, right, bottom, left = box
            cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 255), 2)
            cv2.putText(image, f"{name}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        display = cv2.resize(image, (800, 600)) if image.shape[1] > 800 else image
        cv2.imshow("Prediction", display)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()



def predict_video(video_path, known_embeddings, known_labels, tolerance=0.6, use_camera=False):
    if use_camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps) if fps > 0 else 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for (box, encoding) in zip(boxes, encodings):
            distances = face_recognition.face_distance(known_embeddings, encoding)
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]

            if best_distance < tolerance:
                name = known_labels[best_match_index]
                confidence = (1.0 - best_distance) * 100
                if confidence < 30:
                    name = "Unknown"
            else:
                name = "Unknown"

            top, right, bottom, left = box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{name}" if name != "Unknown" else "Unknown"
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        display = cv2.resize(frame, (800, 600)) if frame.shape[1] > 800 else frame
        cv2.imshow("Face Recognition Video", display)

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mode = input("Choose mode (train / test / add  / camera): ").strip().lower()
    if mode == "train":
        dataset_path = "LFW/lfw-deepfunneled/lfw-deepfunneled"
        embeddings, labels = extract_embeddings_from_folder(dataset_path)
        if embeddings:
            save_embeddings(embeddings, labels)
        else:
            print("There is not enough data for training.")

    elif mode == "add":
        dataset_path = "LFW/lfw-deepfunneled/lfw_try/Dwayne_Johnson"
        new_embeddings, new_labels = extract_embeddings_from_folder(dataset_path)
        if new_embeddings:
            old_embeddings, old_labels = load_embeddings()
            embeddings = np.concatenate([np.array(old_embeddings), np.array(new_embeddings)])
            labels = np.concatenate([np.array(old_labels), np.array(new_labels)])
            save_embeddings(embeddings, labels)
            print("New faces added successfully.")

    elif mode == "camera":
        embeddings, labels = load_embeddings()
        if len(embeddings) > 0:
            predict_video(None, embeddings, labels, use_camera=True)

    elif mode == "test":
        test_dir = "LFW/lfw-deepfunneled/lfw_test"
        embeddings, labels = load_embeddings()
        if len(embeddings) > 0:
            predict_images_in_folder(test_dir, embeddings, labels)

    else:
        print("Invalid mode. Choose: train / test / add / camera")

