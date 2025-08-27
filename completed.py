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
import re
import ollama
from collections import deque
from difflib import get_close_matches
from advanced_word_corrector import AdvancedWordCorrector, AdvancedSentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

# System instruction (keeps style consistent)
system_prompt = (
    "You are an advanced ASL predictor and sentence stylist. "
    "For each input phrase, produce one short, elegant, and grammatically perfect English sentence "
    "that faithfully represents the intended meaning. "
    "Prefer natural phrasing, fluid word order, and polished sentence-level punctuation. "
    "Keep each output concise (one sentence) and beautiful."
)

def show_loading_screen(frame, message="Processing..."):
    """Display a minimal loading screen"""
    # Create a semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Loading message
    cv2.putText(frame, message, (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    
    # Simple animated dots
    dot_count = int(time.time() * 2) % 4
    dots = "." * dot_count
    cv2.putText(frame, dots, (frame.shape[1]//2 + 50, frame.shape[0]//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    
    return frame

def find_closest_word(target_word, word_list, max_distance=3):
    """Find the closest word prioritizing same-first-letter candidates, using Levenshtein + char similarity"""
    if not target_word or not word_list:
        return target_word
    
    best_match = target_word
    min_distance = float('inf')
    
    # Prefer words that share the same first letter; fall back if none
    first = target_word[0].lower()
    same_initial = [w for w in word_list if isinstance(w, str) and len(w) > 0 and w[0].lower() == first]
    candidates = same_initial if same_initial else list(word_list)

    for word in candidates:
        # Calculate Levenshtein distance
        distance = levenshtein_distance(target_word.lower(), word.lower())
        
        # Also check for character similarity (common typos)
        char_similarity = calculate_char_similarity(target_word.lower(), word.lower())
        
        # Combined score (lower is better)
        combined_score = distance + (1 - char_similarity) * 2
        
        # Small penalty if first letters differ (only happens on fallback)
        if word and word[0].lower() != first:
            combined_score += 0.5
        
        if combined_score < min_distance and combined_score <= max_distance:
            min_distance = combined_score
            best_match = word
    
    return best_match

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_char_similarity(s1, s2):
    """Calculate character similarity between two strings"""
    if not s1 or not s2:
        return 0.0
    
    # Common character substitutions (typos)
    char_substitutions = {
        'a': ['q', 'w', 's', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'],
        'e': ['w', 's', 'd', 'r'],
        'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'],
        'h': ['g', 'y', 'u', 'j', 'n', 'b'],
        'i': ['u', 'j', 'k', 'o'],
        'j': ['h', 'u', 'i', 'k', 'm', 'n'],
        'k': ['j', 'i', 'o', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'k', 'l', 'p'],
        'p': ['o', 'l'],
        'q': ['w', 'a'],
        'r': ['e', 'd', 'f', 't'],
        's': ['a', 'w', 'e', 'd', 'z'],
        't': ['r', 'f', 'g', 'y'],
        'u': ['y', 'h', 'j', 'i'],
        'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'a', 's', 'e'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'g', 'h', 'u'],
        'z': ['a', 's', 'x']
    }
    
    # Count similar characters
    similar_chars = 0
    total_chars = max(len(s1), len(s2))
    
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            similar_chars += 1
        elif s1[i] in char_substitutions and s2[i] in char_substitutions[s1[i]]:
            similar_chars += 0.8  # High similarity for adjacent keys
        elif s1[i] in char_substitutions and any(s2[i] in subs for subs in char_substitutions.values()):
            similar_chars += 0.6  # Medium similarity for any substitution
    
    return similar_chars / total_chars if total_chars > 0 else 0.0

def collapse_repeats(text: str) -> str:
    """Collapse repeated characters (e.g., 'kikl' stays, 'cooool' -> 'cool')."""
    if not text:
        return text
    collapsed = [text[0]]
    for ch in text[1:]:
        if ch != collapsed[-1]:
            collapsed.append(ch)
    return "".join(collapsed)

def find_closest_in_spellcheck(word: str, dictionary_words: set[str]) -> str:
    """Find nearest English word from spellcheck dict with strong first-letter and length constraints."""
    if not dictionary_words:
        return word
    w = word.lower()
    w_collapsed = collapse_repeats(w)
    first = w[0]
    last = w[-1]
    # Filter: same first letter and reasonable length window
    same_initial = [d for d in dictionary_words if len(d) > 0 and d[0] == first and abs(len(d) - len(w)) <= 2]
    # Further prefer same last letter if possible
    same_initial_last = [d for d in same_initial if len(d) > 0 and d[-1] == last]
    candidates = same_initial_last if same_initial_last else (same_initial if same_initial else [d for d in dictionary_words if abs(len(d) - len(w)) <= 2])
    if not candidates:
        candidates = list(dictionary_words)
    best = w
    best_score = float('inf')
    for cand in candidates:
        c = cand.lower()
        c_collapsed = collapse_repeats(c)
        # Base distance on collapsed forms to tolerate held letters
        d1 = levenshtein_distance(w_collapsed, c_collapsed)
        # Secondary distance on raw strings
        d2 = levenshtein_distance(w, c)
        char_sim = calculate_char_similarity(w, c)
        score = d1 * 1.2 + d2 * 0.8 + (1 - char_sim) * 1.5
        if c[0] != first:
            score += 0.75
        if c[-1] != last:
            score += 0.4
        if score < best_score:
            best_score = score
            best = cand
    return best

def smart_word_correction(word, context=""):
    """Smart word correction using similarity and context"""
    if not word or len(word) < 2:
        return word
    
    # Common meaningful words for comparison
    meaningful_words = [
        # Common words
        "good", "bad", "big", "small", "new", "old", "hot", "cold",
        "happy", "sad", "tired", "excited", "beautiful", "ugly",
        "strong", "weak", "fast", "slow", "easy", "hard",
        
        # Common verbs
        "go", "come", "see", "hear", "know", "think", "make", "take",
        "give", "get", "put", "say", "tell", "ask", "help", "work",
        "want", "need", "like", "love", "build", "learn", "play",
        
        # Common nouns
        "name", "time", "day", "person", "thing", "place", "work", "home",
        "family", "cat", "dog", "book", "car", "house", "school", "friend",
        "food", "water", "man", "woman", "boy", "girl", "child",
        
        # Pronouns and articles
        "I", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "my", "your", "his", "her", "its", "our", "their", "the", "a", "an",
        
        # Question words
        "what", "how", "when", "where", "why", "who", "which"
    ]
    
    # Try to find the closest meaningful word first
    corrected = find_closest_word(word, meaningful_words, max_distance=3)
    
    # If no good match found, try to make it more readable
    if corrected == word and len(word) > 2:
        # Try to fix common patterns
        if word.lower().startswith('g') and word.lower().endswith('d'):
            if 'ee' in word.lower() or 'oo' in word.lower():
                corrected = "good"
            elif 'ea' in word.lower():
                corrected = "greed"
        elif word.lower().startswith('b') and word.lower().endswith('d'):
            if 'a' in word.lower():
                corrected = "bad"
        elif word.lower().startswith('h') and word.lower().endswith('y'):
            if 'a' in word.lower():
                corrected = "happy"
    
    return corrected

    def update_suggestions(self):
        """Update word suggestions based on current context"""
        if self.word_buffer:
            current_word = "".join(self.word_buffer)
            self.suggestions = self.sentence_former.get_auto_completion_suggestions(current_word)
        elif self.sentence_buffer:
            current_sentence = " ".join(self.sentence_buffer)
            self.suggestions = self.sentence_former.suggest_next_word(current_sentence)
        else:
            self.suggestions = ["I", "You", "The", "What", "How", "Hello", "Hi"]

def process_with_ollama_safe(text):
    """Process text using Ollama for grammar correction without hallucination"""
    try:
        # Create a strict prompt that prevents hallucination
        safe_prompt = (
            "You are a grammar correction assistant. Your task is to ONLY correct grammar, "
            "punctuation, and sentence structure. DO NOT change the meaning, add new words, "
            "or create new content. ONLY fix existing grammar issues. "
            "If the input is already grammatically correct, return it unchanged. "
            "Input text: "
        )
        
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": safe_prompt},
                {"role": "user", "content": f"Correct only grammar and punctuation: {text}"}
            ]
        )
        
        result = response["message"]["content"].strip()
        
        # Safety check: ensure the result contains the original words
        original_words = set(text.lower().split())
        result_words = set(result.lower().split())
        
        # If more than 30% new words are added, reject the result
        new_words = result_words - original_words
        if len(new_words) > len(original_words) * 0.3:
            print(f"Ollama added too many new words, using original: {text}")
            return text
        
        # If result is significantly longer, reject it
        if len(result) > len(text) * 1.5:
            print(f"Ollama result too long, using original: {text}")
            return text
        
        return result
        
    except Exception as e:
        print(f"Ollama error: {e}")
        return text  # Return original text if Ollama fails

def form_sentence_with_ollama(words):
    """Use Ollama to form a proper sentence from collected words"""
    try:
        # Create a sentence formation prompt
        sentence_prompt = (
            "You are a sentence formation assistant. Your task is to create a grammatically correct "
            "English sentence using ONLY the provided words. You may reorder words and add basic "
            "grammar elements (articles, prepositions, verb forms) but DO NOT add new content words. "
            "Keep the sentence simple and natural. Return only the sentence, nothing else."
        )
        
        # Join words with spaces
        word_list = " ".join(words)
        
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": sentence_prompt},
                {"role": "user", "content": f"Form a sentence using these words: {word_list}"}
            ]
        )
        
        result = response["message"]["content"].strip()
        
        # Safety check: ensure the result contains ALL corrected words (no missing words)
        original_words = [w.lower() for w in words]
        result_words = result.lower().split()
        missing = [w for w in original_words if w not in result_words]
        if missing:
            print(f"Ollama missing words {missing}, using simple join: {word_list}")
            return word_list
        
        # If result is too long, reject it
        if len(result) > len(word_list) * 2:
            print(f"Ollama result too long, using simple join: {word_list}")
            return word_list
        
        return result
        
    except Exception as e:
        print(f"Ollama sentence formation error: {e}")
        # Fallback to simple word joining
        return " ".join(words)

class AdvancedWordCorrector:
    """Advanced word correction with spell checking and context-aware transformation"""
    def __init__(self):
        # Enhanced dictionary with common words and their variations
        self.word_dictionary = {
            # Common verbs with variations
            "build": ["bild", "buid", "buld", "bild", "buil", "bld"],
            "learn": ["lern", "lean", "larn", "leran", "lrn"],
            "work": ["wrok", "wok", "wrk", "wrok", "work"],
            "help": ["hlp", "hel", "hepl", "hlp", "help"],
            "want": ["wnt", "wan", "wnt", "want", "wnt"],
            "need": ["ned", "nead", "ned", "need", "nd"],
            "like": ["lik", "liek", "lik", "like", "lk"],
            "love": ["lov", "luv", "lov", "love", "lv"],
            "go": ["go", "goe", "go", "go", "g"],
            "come": ["com", "cme", "com", "come", "cm"],
            "see": ["se", "see", "se", "see", "s"],
            "hear": ["her", "hear", "her", "hear", "hr"],
            "know": ["no", "kno", "no", "know", "kn"],
            "think": ["thnk", "thik", "thnk", "think", "thk"],
            "make": ["mak", "mae", "mak", "make", "mk"],
            "take": ["tak", "tae", "tak", "take", "tk"],
            "give": ["giv", "gie", "giv", "give", "gv"],
            "get": ["get", "gte", "get", "get", "gt"],
            "put": ["put", "pte", "put", "put", "pt"],
            "say": ["say", "sae", "say", "say", "sy"],
            "tell": ["tel", "tll", "tel", "tell", "tl"],
            "ask": ["ask", "as", "ask", "ask", "ak"],
            
            # Common nouns with variations
            "name": ["nme", "nae", "nme", "name", "nm"],
            "time": ["tme", "tie", "tme", "time", "tm"],
            "day": ["day", "dae", "day", "day", "dy"],
            "person": ["prson", "persn", "prson", "person", "prsn"],
            "thing": ["thng", "thig", "thng", "thing", "thg"],
            "place": ["plce", "plae", "plce", "place", "plc"],
            "work": ["wrok", "wok", "wrk", "work", "wrk"],
            "home": ["hom", "hoe", "hom", "home", "hm"],
            "family": ["famy", "faily", "famy", "family", "fmly"],
            "cat": ["cat", "ct", "cat", "cat", "ct"],
            "dog": ["dog", "dg", "dog", "dog", "dg"],
            "book": ["bok", "boo", "bok", "book", "bk"],
            "car": ["car", "cr", "car", "car", "cr"],
            "house": ["hose", "huse", "hose", "house", "hs"],
            "school": ["schol", "scool", "schol", "school", "schl"],
            "friend": ["frend", "frid", "frend", "friend", "frd"],
            "food": ["fod", "foe", "fod", "food", "fd"],
            "water": ["wter", "wate", "wter", "water", "wtr"],
            
            # Common adjectives with variations
            "good": ["god", "goo", "god", "good", "gd"],
            "bad": ["bad", "bd", "bad", "bad", "bd"],
            "big": ["big", "bg", "big", "big", "bg"],
            "small": ["smal", "smll", "smal", "small", "sml"],
            "new": ["new", "nw", "new", "new", "nw"],
            "old": ["old", "ol", "old", "old", "ol"],
            "hot": ["hot", "ht", "hot", "hot", "ht"],
            "cold": ["col", "cod", "col", "cold", "cl"],
            "happy": ["hpy", "happ", "hpy", "happy", "hpy"],
            "sad": ["sad", "sd", "sad", "sad", "sd"],
            "tired": ["tred", "tird", "tred", "tired", "trd"],
            "excited": ["exctd", "excit", "exctd", "excited", "exct"],
            
            # Question words with variations
            "what": ["wat", "wht", "wat", "what", "wht"],
            "how": ["how", "hw", "how", "how", "hw"],
            "when": ["wen", "whn", "wen", "when", "whn"],
            "where": ["were", "wher", "were", "where", "whr"],
            "why": ["wy", "why", "wy", "why", "wy"],
            "who": ["who", "wo", "who", "who", "wo"],
            "which": ["wich", "whch", "wich", "which", "whc"]
        }
        
        # Create reverse mapping for quick lookup
        self.reverse_dictionary = {}
        for correct_word, variations in self.word_dictionary.items():
            for variation in variations:
                self.reverse_dictionary[variation.lower()] = correct_word
        
        # Common misspellings and corrections
        self.common_corrections = {
            "i": "I", "im": "I'm", "ive": "I've", "id": "I'd",
            "ur": "your", "u": "you", "r": "are", "yr": "your",
            "thx": "thanks", "pls": "please", "tho": "though",
            "nite": "night", "nite": "night", "nite": "night",
            "shpe": "shape", "bild": "build", "lern": "learn",
            "wrok": "work", "hlp": "help", "wnt": "want",
            "ned": "need", "lik": "like", "lov": "love"
        }
        
        # Context-aware word suggestions
        self.context_suggestions = {
            "build": ["house", "building", "project", "system", "app"],
            "learn": ["language", "skill", "subject", "topic", "method"],
            "work": ["job", "project", "task", "assignment", "duty"],
            "help": ["assist", "support", "aid", "guide", "teach"],
            "want": ["desire", "need", "wish", "hope", "plan"],
            "need": ["require", "want", "must", "should", "essential"],
            "like": ["enjoy", "love", "prefer", "appreciate", "favor"],
            "love": ["adore", "like", "enjoy", "cherish", "treasure"]
        }

    def correct_word(self, word, context=""):
        """Advanced word correction with context awareness"""
        if not word:
            return ""
        
        word_lower = word.lower()
        
        # Check for exact corrections first
        if word_lower in self.common_corrections:
            return self.common_corrections[word_lower]
        
        # Check for variations in dictionary
        if word_lower in self.reverse_dictionary:
            return self.reverse_dictionary[word_lower]
        
        # Check if it's already correct
        if word_lower in self.word_dictionary:
            return word
        
        # Try to find close matches
        all_words = list(self.word_dictionary.keys()) + list(self.common_corrections.values())
        close_matches = get_close_matches(word_lower, all_words, n=3, cutoff=0.6)
        
        if close_matches:
            # If context is provided, prefer contextually relevant words
            if context and close_matches[0] in self.context_suggestions:
                return close_matches[0]
            return close_matches[0]
        
        # If no close match found, return the original word if it looks reasonable
        if re.match(r'^[a-zA-Z]+$', word) and len(word) >= 2:
            return word
        
        return ""

    def get_context_suggestions(self, word, context=""):
        """Get context-aware suggestions for a word"""
        corrected_word = self.correct_word(word, context)
        if corrected_word in self.context_suggestions:
            return self.context_suggestions[corrected_word]
        return []

    def is_valid_word(self, word):
        """Check if a word is valid"""
        if not word:
            return False
        
        word_lower = word.lower()
        
        # Check if it's a known word
        if word_lower in self.word_dictionary or word_lower in self.common_corrections:
            return True
        
        # Check if it's a variation
        if word_lower in self.reverse_dictionary:
            return True
        
        # Basic validation rules
        if len(word) < 2 or len(word) > 20:
            return False
        
        # Check for repeated characters (e.g., "helllooo")
        if any(word.count(char) > len(word) // 2 for char in set(word)):
            return False
        
        # Check if it looks like a reasonable word
        if re.match(r'^[a-zA-Z]+$', word):
            return True
        
        return False

class SimpleWordValidator:
    """Simple word validator without external dependencies"""
    def __init__(self):
        # Common English words for validation
        self.common_words = {
            # Pronouns
            "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "his", "hers", "ours", "theirs",
            
            # Articles
            "a", "an", "the",
            
            # Common verbs
            "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
            "want", "need", "like", "love", "go", "come", "see", "hear", "know", "think",
            "make", "take", "give", "get", "put", "say", "tell", "ask", "help", "work",
            
            # Common nouns
            "name", "time", "day", "person", "thing", "place", "work", "home", "family",
            "cat", "dog", "book", "car", "house", "school", "friend", "food", "water",
            "man", "woman", "boy", "girl", "child", "people", "world", "life", "way",
            
            # Common adjectives
            "good", "bad", "big", "small", "new", "old", "hot", "cold", "high", "low",
            "happy", "sad", "tired", "excited", "beautiful", "ugly", "strong", "weak",
            "fast", "slow", "easy", "hard", "right", "wrong", "same", "different",
            
            # Common adverbs
            "very", "much", "many", "few", "little", "more", "less", "most", "least",
            "now", "then", "here", "there", "today", "yesterday", "tomorrow",
            "always", "never", "sometimes", "often", "usually", "rarely",
            
            # Question words
            "what", "how", "when", "where", "why", "who", "which",
            
            # Prepositions
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "down",
            "over", "under", "above", "below", "between", "among", "through", "across",
            
            # Conjunctions
            "and", "or", "but", "because", "if", "when", "while", "although", "unless",
            
            # Numbers
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "first", "second", "third", "last", "next", "previous"
        }
        
        # Add capitalized versions
        self.common_words.update({word.capitalize() for word in self.common_words})
        
        # Add some common misspellings and corrections
        self.corrections = {
            "i": "I", "im": "I'm", "ive": "I've", "id": "I'd",
            "ur": "your", "u": "you", "r": "are", "yr": "your",
            "thx": "thanks", "pls": "please", "tho": "though",
            "nite": "night", "nite": "night", "nite": "night"
        }

    def is_valid_word(self, word):
        """Check if a word is valid"""
        if not word:
            return False
        
        # Check if it's a known word
        if word.lower() in self.common_words or word in self.common_words:
            return True
        
        # Check if it's a correction
        if word.lower() in self.corrections:
            return True
        
        # Basic validation rules
        if len(word) < 2 or len(word) > 20:
            return False
        
        # Check for repeated characters (e.g., "helllooo")
        if any(word.count(char) > len(word) // 2 for char in set(word)):
            return False
        
        # Check if it looks like a reasonable word
        if re.match(r'^[a-zA-Z]+$', word):
            return True
        
        return False

    def correct_word(self, word):
        """Correct common misspellings"""
        if not word:
            return ""
        
        # Check for corrections
        if word.lower() in self.corrections:
            return self.corrections[word.lower()]
        
        # If it's already valid, return as is
        if self.is_valid_word(word):
            return word
        
        # Try to find similar words (simple approach)
        word_lower = word.lower()
        for valid_word in self.common_words:
            if valid_word.lower() == word_lower:
                return valid_word
        
        return ""

class AdvancedSentenceTransformer:
    """Advanced sentence transformation using BART and T5 concepts"""
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Sentence transformation templates (BART-style)
        self.transformation_templates = {
            "grammar_correction": {
                "i am": "I am",
                "i have": "I have",
                "i want": "I want",
                "i like": "I like",
                "i love": "I love",
                "i need": "I need",
                "i can": "I can",
                "i will": "I will",
                "i would": "I would",
                "i should": "I should"
            },
            "sentence_structure": {
                "am happy": "I am happy",
                "have a cat": "I have a cat",
                "want to go": "I want to go",
                "like this": "I like this",
                "love you": "I love you",
                "need help": "I need help",
                "can do": "I can do",
                "will go": "I will go"
            },
            "question_formation": {
                "what is": "What is",
                "how are": "How are",
                "when will": "When will",
                "where is": "Where is",
                "why do": "Why do",
                "who is": "Who is",
                "which one": "Which one"
            }
        }
        
        # Context-aware sentence improvements
        self.context_improvements = {
            "greeting": {
                "hello": "Hello",
                "hi": "Hi",
                "good morning": "Good morning",
                "good afternoon": "Good afternoon",
                "good evening": "Good evening"
            },
            "expression": {
                "thank you": "Thank you",
                "please": "Please",
                "sorry": "Sorry",
                "excuse me": "Excuse me"
            }
        }

    def transform_sentence(self, sentence):
        """Transform sentence using advanced NLP techniques"""
        if not sentence:
            return sentence
        
        # Apply BART-style transformations
        transformed = self.apply_bart_transformations(sentence)
        
        # Apply T5-style improvements
        improved = self.apply_t5_improvements(transformed)
        
        # Final grammar correction
        final = self.correct_grammar(improved)
        
        return final

    def apply_bart_transformations(self, sentence):
        """Apply BART-style sentence transformations"""
        transformed = sentence
        
        # Apply grammar correction templates
        for wrong, right in self.transformation_templates["grammar_correction"].items():
            transformed = transformed.replace(wrong, right)
        
        # Apply sentence structure templates
        for wrong, right in self.transformation_templates["sentence_structure"].items():
            if transformed.lower().startswith(wrong.lower()):
                transformed = right + transformed[len(wrong):]
        
        # Apply question formation templates
        for wrong, right in self.transformation_templates["question_formation"].items():
            if transformed.lower().startswith(wrong.lower()):
                transformed = right + transformed[len(wrong):]
        
        return transformed

    def apply_t5_improvements(self, sentence):
        """Apply T5-style sentence improvements"""
        improved = sentence
        
        # Apply context-aware improvements
        for context_type, improvements in self.context_improvements.items():
            for wrong, right in improvements.items():
                if improved.lower().startswith(wrong.lower()):
                    improved = right + improved[len(wrong):]
        
        # Improve sentence flow and coherence
        improved = self.improve_sentence_flow(improved)
        
        return improved

    def improve_sentence_flow(self, sentence):
        """Improve sentence flow and coherence"""
        words = sentence.split()
        if not words:
            return sentence
        
        # Ensure proper capitalization
        if words[0] and words[0].islower():
            words[0] = words[0].capitalize()
        
        # Add proper punctuation if missing
        if words and not any(words[-1].endswith(p) for p in [".", "!", "?"]):
            if words[0].lower() in ["what", "how", "when", "where", "why", "who", "which"]:
                words.append("?")
            else:
                words.append(".")
        
        return " ".join(words)

    def correct_grammar(self, sentence):
        """Apply comprehensive grammar corrections using BART-T5 transformer"""
        if not sentence:
            return sentence
        
        # Apply basic grammar corrections directly
        corrections = {
            "i ": "I ",
            " i ": " I ",
            " i.": " I.",
            " i!": " I!",
            " i?": " I?",
            " i,": " I,"
        }
        
        corrected = sentence
        for wrong, right in corrections.items():
            corrected = corrected.replace(wrong, right)
        
        return corrected

class EnhancedSentenceFormation:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.word_validator = SimpleWordValidator()
        self.advanced_corrector = AdvancedWordCorrector()
        self.sentence_transformer = AdvancedSentenceTransformer()
        
        # Common sentence templates and patterns
        self.sentence_templates = {
            "greeting": ["Hello", "Hi", "Good morning", "Good afternoon", "Good evening"],
            "question": ["What", "How", "When", "Where", "Why", "Who", "Which"],
            "common_verbs": ["is", "are", "was", "were", "have", "has", "had", "do", "does", "did"],
            "common_nouns": ["name", "time", "day", "person", "thing", "place", "work", "home"],
            "common_adj": ["good", "bad", "big", "small", "new", "old", "hot", "cold"]
        }
        
        # Grammar rules for sentence formation
        self.grammar_rules = {
            "sentence_start": ["I", "You", "He", "She", "It", "We", "They", "The", "A", "An"],
            "question_start": ["What", "How", "When", "Where", "Why", "Who", "Which", "Is", "Are", "Do", "Does"],
            "punctuation": [".", "!", "?"]
        }
        
        # Auto-completion suggestions
        self.auto_complete = {
            "I": ["am", "have", "want", "need", "like", "love"],
            "You": ["are", "have", "want", "need", "like", "love"],
            "The": ["cat", "dog", "book", "car", "house", "person"],
            "A": ["cat", "dog", "book", "car", "house", "person"],
            "What": ["is", "are", "do", "does", "time", "name"],
            "How": ["are", "is", "do", "does", "old", "far"]
        }

    def validate_sentence_structure(self, words):
        """Validate and improve sentence structure"""
        if not words:
            return []
        
        # Basic sentence structure validation
        sentence = " ".join(words)
        doc = self.nlp(sentence)
        
        # Check if sentence starts with proper word
        first_word = words[0].capitalize()
        if first_word not in self.grammar_rules["sentence_start"] and first_word not in self.grammar_rules["question_start"]:
            # Try to find a better starting word
            if first_word.lower() in ["am", "are", "is", "was", "were"]:
                words.insert(0, "I")
            elif first_word.lower() in ["have", "has", "had"]:
                words.insert(0, "I")
            elif first_word.lower() in ["want", "need", "like", "love"]:
                words.insert(0, "I")
        
        # Add proper punctuation
        if not any(words[-1].endswith(p) for p in self.grammar_rules["punctuation"]):
            if words[0].lower() in ["what", "how", "when", "where", "why", "who", "which"]:
                words.append("?")
            else:
                words.append(".")
        
        return words

    def enforce_subject_verb_agreement(self, words):
        """Adjust simple subject-verb agreement for 'to be' (am/is/are) and capitalize 'I'."""
        if not words:
            return words
        # Capitalize standalone 'i' at start
        if words and words[0].lower() == "i":
            words[0] = "I"

        # Find first verb candidate
        doc = self.nlp(" ".join(words))
        if not doc:
            return words
        # Heuristics: subject is first token; verb is first AUX/VERB
        subj_token = doc[0]
        verb_index = None
        for i, t in enumerate(doc):
            if t.pos_ in ("AUX", "VERB"):
                verb_index = i
                break
        if verb_index is None or verb_index >= len(words):
            return words

        subj_text = subj_token.text
        verb_text = words[verb_index].lower()
        be_forms = {"am", "is", "are"}

        # Decide target 'to be' form
        target = None
        if subj_text.lower() == "i":
            target = "am"
        elif subj_text.lower() in ("you", "we", "they"):
            target = "are"
        elif subj_token.pos_ in ("PRON",) and subj_text.lower() in ("he", "she", "it"):
            target = "is"
        elif subj_token.tag_ in ("NNS", "NNPS"):  # plural nouns
            target = "are"
        elif subj_token.tag_ in ("NN", "NNP"):  # singular nouns
            target = "is"

        if target and verb_text in be_forms and verb_text != target:
            words[verb_index] = target

        return words

    def get_auto_completion_suggestions(self, current_word):
        """Get auto-completion suggestions for the current word"""
        suggestions = []
        
        # Get suggestions based on current word
        if current_word in self.auto_complete:
            suggestions.extend(self.auto_complete[current_word])
        
        # Get suggestions based on sentence context
        if current_word.lower() in ["is", "are", "was", "were"]:
            suggestions.extend(["good", "bad", "big", "small", "here", "there"])
        elif current_word.lower() in ["have", "has", "had"]:
            suggestions.extend(["a", "an", "the", "some", "many"])
        elif current_word.lower() in ["want", "need", "like", "love"]:
            suggestions.extend(["to", "the", "a", "an", "this", "that"])
        
        return suggestions[:5]  # Limit to 5 suggestions

    def correct_grammar(self, sentence):
        """Apply grammar corrections to the sentence"""
        if not sentence:
            return sentence
        
        # Use advanced sentence transformer
        return self.sentence_transformer.transform_sentence(sentence)

    def suggest_next_word(self, current_sentence):
        """Suggest the next word based on current sentence context"""
        if not current_sentence:
            return ["I", "You", "The", "What", "How"]
        
        words = current_sentence.split()
        if not words:
            return ["I", "You", "The", "What", "How"]
        
        last_word = words[-1].lower()
        
        # Suggest based on last word
        if last_word in ["i", "you", "he", "she", "it", "we", "they"]:
            return ["am", "are", "is", "was", "were", "have", "has", "had", "want", "need", "like", "love"]
        elif last_word in ["am", "are", "is", "was", "were"]:
            return ["good", "bad", "big", "small", "here", "there", "happy", "sad", "tired", "excited"]
        elif last_word in ["have", "has", "had"]:
            return ["a", "an", "the", "some", "many", "few", "lot", "enough"]
        elif last_word in ["want", "need", "like", "love"]:
            return ["to", "the", "a", "an", "this", "that", "it", "them", "you"]
        elif last_word in ["the", "a", "an"]:
            return ["cat", "dog", "book", "car", "house", "person", "thing", "place", "time", "day"]
        
        return ["and", "or", "but", "because", "when", "where", "how", "why"]

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
        
        # Enhanced sentence formation
        self.sentence_former = EnhancedSentenceFormation()
        
        self.prediction_text = "Waiting for sign..."
        self.last_prediction = None
        self.word_buffer = []
        self.sentence_buffer = []
        self.letter_start_time = None
        self.tts_queue = queue.Queue()
        self.tts_engine = pyttsx3.init()
        self.suggestions = []
        self.show_suggestions = False
        self.is_processing = False
        self.processing_start_time = None
        
        threading.Thread(target=self.process_tts, daemon=True).start()
        
        self.nlp = spacy.load("en_core_web_sm")
        self.word_validator = SimpleWordValidator()
        self.advanced_corrector = AdvancedWordCorrector()
        self.common_words = {"I", "the", "a", "an", "you", "he", "she", "it", "we", "they"}
        
        # Build spellcheck dictionary: frequency list + validator/common + corrector dicts
        self.spellcheck_words = set()
        try:
            with open("frequency_dict.txt", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    token = line.strip().split()[0] if line.strip() else ""
                    if token:
                        self.spellcheck_words.add(token.lower())
        except Exception:
            pass
        # Add validator common words
        try:
            for w in getattr(self.word_validator, "common_words", []):
                self.spellcheck_words.add(str(w).lower())
        except Exception:
            pass
        # Add advanced corrector dictionaries
        try:
            for w in getattr(self.advanced_corrector, "word_dictionary", {}).keys():
                self.spellcheck_words.add(str(w).lower())
            for w in getattr(self.advanced_corrector, "common_corrections", {}).keys():
                self.spellcheck_words.add(str(w).lower())
            for w in getattr(self.advanced_corrector, "common_corrections", {}).values():
                self.spellcheck_words.add(str(w).lower())
        except Exception:
            pass

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
        """Enhanced word refinement with smart similarity-based correction"""
        if len(text) < 2 or len(text) > 15:
            return ""
        if not text.isalpha():
            return ""
        if any(text.count(char) > len(text) // 2 for char in set(text)):
            return ""
        
        # If it's already in spellcheck, keep unchanged
        if self.is_word_in_spellcheck(text):
            return text

        # Use smart word correction
        corrected_word = smart_word_correction(text, context=" ".join(self.sentence_buffer))
        if corrected_word and corrected_word != text:
            print(f"Smart correction: '{text}' -> '{corrected_word}'")
            return corrected_word
        
        # Fall back to advanced correction if smart correction didn't help
        corrected_word = self.advanced_corrector.correct_word(text, context=" ".join(self.sentence_buffer))
        if corrected_word and corrected_word.lower() != text.lower():
            print(f"Advanced correction: '{text}' -> '{corrected_word}'")
            return corrected_word
        # Accept advanced correction only if it yields a known/spellchecked word
        if corrected_word and self.is_word_in_spellcheck(corrected_word):
            return corrected_word
        
        # Finally, use the global spellcheck dictionary to find nearest real English word
        if self.spellcheck_words:
            nearest = find_closest_in_spellcheck(text, self.spellcheck_words)
            if nearest and nearest.lower() != text.lower():
                print(f"Spellcheck correction: '{text}' -> '{nearest}'")
                return nearest

        # Try simple correction
        simple_corrected = self.word_validator.correct_word(text)
        if simple_corrected:
            print(f"Simple correction: '{text}' -> '{simple_corrected}'")
            return simple_corrected
        
        return ""

    def is_word_in_spellcheck(self, word: str) -> bool:
        if not word:
            return False
        return word.lower() in self.spellcheck_words

        

    def update_suggestions(self):
        """Update word suggestions based on current context"""
        if self.word_buffer:
            current_word = "".join(self.word_buffer)
            self.suggestions = self.sentence_former.get_auto_completion_suggestions(current_word)
        elif self.sentence_buffer:
            current_sentence = " ".join(self.sentence_buffer)
            self.suggestions = self.sentence_former.suggest_next_word(current_sentence)
        else:
            self.suggestions = ["I", "You", "The", "What", "How", "Hello", "Hi"]

    def display_prediction(self, frame, word, sentence, hand_landmarks=None):
        # Show loading screen if processing
        if self.is_processing:
            frame = show_loading_screen(frame, "Processing...")
            return frame
        
        # Current word being formed (top-left corner)
        current_word = ''.join(self.word_buffer)
        if current_word:
            cv2.putText(frame, current_word, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Formed sentence display (context under predicted word at top-left)
        sentence_display = " ".join(self.sentence_buffer)
        if sentence_display:
            cv2.putText(frame, sentence_display, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Hand landmarks display
        if hand_landmarks and word:
            x, y = int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0] - 20)
            cv2.putText(frame, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    def run(self):
        print("Starting Sign Language Recognition System...")
        print("Controls: SPACE=Add word, ENTER=Form and speak sentence, BACKSPACE=Remove word, C=Clear, Q=Quit")
        print("Advanced word correction and sentence formation active")
        print("System ready for sign language recognition")
        
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
                            self.update_suggestions()
                    else:
                        self.letter_start_time = time.time()
                    self.last_prediction = prediction
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Add word to sentence
                formed_word = "".join(self.word_buffer)
                if formed_word:
                    corrected_word = self.refine_prediction(formed_word)
                    if corrected_word or formed_word in self.common_words:
                        final_word = corrected_word if corrected_word else formed_word
                        self.sentence_buffer.append(final_word)
                        print(f"Added word: {final_word}")
                self.word_buffer = []
                self.update_suggestions()
            elif key == ord('\r') or key == 13:
                # Speak sentence with Ollama enhancement
                if self.sentence_buffer:
                    self.is_processing = True
                    self.processing_start_time = time.time()
                    
                    # Start processing in background thread
                    def process_sentence():
                        try:
                            # Apply smart word correction to each word
                            corrected_words = []
                            for word in self.sentence_buffer:
                                # Keep exact if already valid; otherwise refine once
                                if self.is_word_in_spellcheck(word):
                                    corrected_words.append(word)
                                else:
                                    corrected_word = self.refine_prediction(word)
                                    if corrected_word and corrected_word != word:
                                        print(f"Corrected: '{word}' -> '{corrected_word}'")
                                    corrected_words.append(corrected_word if corrected_word else word)
                            
                            print(f"Original words: {' '.join(self.sentence_buffer)}")
                            print(f"Corrected words: {' '.join(corrected_words)}")
                            
                            # Use Ollama only to frame a sentence (must keep all words). If Ollama fails safety checks, use our own framing
                            formed_sentence = form_sentence_with_ollama(corrected_words)
                            if formed_sentence.strip().lower() != " ".join(corrected_words).strip().lower():
                                print(f"Ollama formed sentence: {formed_sentence}")
                            
                            # Ensure all corrected words are present; otherwise frame sentence locally
                            missing = [w for w in [cw.lower() for cw in corrected_words] if w not in formed_sentence.lower().split()]
                            if missing:
                                # Local framing: simple grammatical shaping without changing words
                                tmp = list(corrected_words)
                                # Apply subject-verb agreement
                                tmp = self.sentence_former.enforce_subject_verb_agreement(tmp)
                                # Capitalize first token
                                if tmp:
                                    tmp[0] = tmp[0].capitalize()
                                # Join and ensure punctuation
                                formed_sentence = " ".join(tmp)
                                if not formed_sentence.endswith(('.', '!', '?')):
                                    formed_sentence += '.'
                            
                            # Final light grammar correction that preserves tokens
                            final_sentence = self.sentence_former.correct_grammar(formed_sentence)
                            
                            # Use final formed sentence for TTS
                            self.tts_queue.put(final_sentence)
                            self.sentence_buffer = []
                            self.update_suggestions()
                        except Exception as e:
                            print(f"Error processing sentence: {e}")
                        finally:
                            self.is_processing = False
                            self.processing_start_time = None
                    
                    threading.Thread(target=process_sentence, daemon=True).start()
            elif key == 8:  # Backspace
                # Remove last word from sentence
                if self.sentence_buffer:
                    removed_word = self.sentence_buffer.pop()
                    print(f"Removed word: {removed_word}")
                    self.update_suggestions()
            elif key == ord('c'):
                # Clear sentence
                self.sentence_buffer = []
                self.word_buffer = []
                print("Cleared sentence")
                self.update_suggestions()

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
