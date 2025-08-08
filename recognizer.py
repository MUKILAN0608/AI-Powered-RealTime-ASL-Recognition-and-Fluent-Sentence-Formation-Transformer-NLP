import cv2
import pickle
import mediapipe as mp
import numpy as np
import threading
import queue
import pyttsx3
import time
from spellchecker import SpellChecker

class SignLanguageTextProcessor:
    def __init__(self):
        self.spell_checker = SpellChecker()
        self.common_words = {"I", "the", "a", "an", "you", "he", "she", "it", "we", "they"}
        self.word_buffer = []
        self.sentence_buffer = []
    
    def refine_prediction(self, text):
        if len(text) < 2 or len(text) > 15 or not text.isalpha():
            return ""
        if any(text.count(char) > len(text) // 2 for char in set(text)):
            return ""
        if text in self.spell_checker or text in self.common_words:
            return text
        corrected_word = self.spell_checker.correction(text)
        return corrected_word if corrected_word in self.spell_checker.known([corrected_word]) else ""
    
    def add_letter(self, letter):
        self.word_buffer.append(letter)
    
    def finalize_word(self):
        formed_word = "".join(self.word_buffer)
        if formed_word:
            corrected_word = self.refine_prediction(formed_word)
            if corrected_word or formed_word in self.common_words:
                self.sentence_buffer.append(corrected_word if corrected_word else formed_word)
        self.word_buffer = []
    
    def finalize_sentence(self):
        sentence = " ".join(self.sentence_buffer)
        self.sentence_buffer = []
        return sentence
    
    def get_current_word(self):
        return "".join(self.word_buffer)
    
    def get_current_sentence(self):
        return " ".join(self.sentence_buffer)

class SignLanguageRecognizer:
    def __init__(self, model_path='sign_language_model.pkl'):
        self.model = self.load_model(model_path)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                                         min_detection_confidence=0.85, min_tracking_confidence=0.85)
        
        self.video_source = 0
        self.cap = cv2.VideoCapture(self.video_source)
        self.text_processor = SignLanguageTextProcessor()
        self.last_prediction = None
        self.letter_start_time = None
        
        self.tts_queue = queue.Queue()
        self.tts_engine = pyttsx3.init()
        threading.Thread(target=self.process_tts, daemon=True).start()

    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
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

    def generate_frames(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                prediction = self.predict_sign(self.preprocess_landmarks(hand_landmarks))
                if prediction:
                    if prediction == self.last_prediction:
                        if self.letter_start_time and (time.time() - self.letter_start_time) >= 2:
                            self.text_processor.add_letter(prediction)
                            self.letter_start_time = None
                    else:
                        self.letter_start_time = time.time()
                    self.last_prediction = prediction
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            current_word = self.text_processor.get_current_word()
            current_sentence = self.text_processor.get_current_sentence()
            cv2.putText(frame, f"Word: {current_word}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, f"Sentence: {current_sentence}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def run(self):
        return self.generate_frames()
