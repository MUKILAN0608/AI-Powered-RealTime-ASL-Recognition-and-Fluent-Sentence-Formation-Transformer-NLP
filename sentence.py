import cv2
import pickle
import mediapipe as mp
import numpy as np
import sys
import pyttsx3
import threading
import queue
import spacy
import time
from spellchecker import SpellChecker

sys.stdout.reconfigure(encoding='utf-8')

class SignLanguageRecognizer:
    def __init__(self, model_path='sign_language_model.pkl'):
        self.model = self.load_model(model_path)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                                         min_detection_confidence=0.85, min_tracking_confidence=0.85)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not access the camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cv2.namedWindow("Sign Language Recognition", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Sign Language Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.prediction_text = "Waiting for sign..."
        self.last_prediction = None
        self.word_buffer = []
        self.sentence_buffer = []
        self.letter_start_time = None
        self.tts_queue = queue.Queue()
        self.tts_engine = pyttsx3.init()
        threading.Thread(target=self.process_tts, daemon=True).start()
        
        self.nlp = spacy.load("en_core_web_sm")
        self.spell_checker = SpellChecker()
        self.common_words = {"I", "the", "a", "an", "you", "he", "she", "it", "we", "they"}

    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Model file '{model_path}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def preprocess_landmarks(self, hand_landmarks):
        min_vals = np.min([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], axis=0)
        return [[lm.x - min_vals[0], lm.y - min_vals[1], lm.z - min_vals[2]] for lm in hand_landmarks.landmark]

    def predict_sign(self, landmarks):
        if len(landmarks) == 21:
            prediction = self.model.predict([np.array(landmarks).flatten()])[0]
            return prediction if prediction.isalpha() and len(prediction) == 1 else None
        return None

    def process_tts(self):
        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            self.tts_engine.say(text.strip().capitalize())
            self.tts_engine.runAndWait()

    def refine_prediction(self, text):
        if len(text) < 2 or len(text) > 15:
            return ""
        if not text.isalpha():
            return ""
        if any(text.count(char) > len(text) // 2 for char in set(text)):
            return ""
        if text in self.spell_checker or text in self.common_words:
            return text
        corrected_word = self.spell_checker.correction(text)
        if corrected_word and corrected_word in self.spell_checker.known([corrected_word]):
            return corrected_word
        return ""

    def display_prediction(self, frame, word, sentence, hand_landmarks=None):
        cv2.putText(frame, "Sign Language Recognition", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Current Word: {''.join(self.word_buffer)}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Formed Sentence: {sentence}", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)
        
        if hand_landmarks and word:
            x, y = int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0] - 20)
            cv2.putText(frame, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    def run(self):
        print("Starting Sign Language Recognition...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Warning: Failed to capture frame.")
                continue
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            hand_landmarks = None
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                prediction = self.predict_sign(self.preprocess_landmarks(hand_landmarks))
                if prediction:
                    if prediction == self.last_prediction:
                        if self.letter_start_time and (time.time() - self.letter_start_time) >= 2:
                            self.word_buffer.append(prediction)
                            self.letter_start_time = None
                    else:
                        self.letter_start_time = time.time()
                    self.last_prediction = prediction
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                formed_word = "".join(self.word_buffer)
                if formed_word:
                    corrected_word = self.refine_prediction(formed_word)
                    if corrected_word or formed_word in self.common_words:
                        self.sentence_buffer.append(corrected_word if corrected_word else formed_word)
                self.word_buffer = []
            elif key == ord('\r'):
                sentence_display = " ".join(self.sentence_buffer)
                self.tts_queue.put(sentence_display)
                self.sentence_buffer = []

            sentence_display = " ".join(self.sentence_buffer)
            self.display_prediction(frame, self.last_prediction or "Waiting...", sentence_display, hand_landmarks)
            cv2.imshow("Sign Language Recognition", frame)
        self.cleanup()
    
    def cleanup(self):
        print("Closing application...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.tts_queue.put(None)

if __name__ == "__main__":
    recognizer = SignLanguageRecognizer()
    recognizer.run()
