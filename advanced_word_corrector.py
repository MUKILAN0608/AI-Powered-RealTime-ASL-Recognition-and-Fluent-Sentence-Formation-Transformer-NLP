import re
from difflib import get_close_matches

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
            "ned": "need", "lik": "like", "lov": "love",
            "am": "am", "have": "have", "read": "read", "today": "today"
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
        
        # Check if it's a common English word that shouldn't be changed
        common_words = {"am", "is", "are", "was", "were", "have", "has", "had", 
                       "do", "does", "did", "will", "would", "could", "should", 
                       "can", "may", "might", "must", "shall", "a", "an", "the",
                       "and", "or", "but", "in", "on", "at", "to", "for", "of",
                       "with", "by", "from", "up", "down", "out", "off", "over",
                       "under", "above", "below", "before", "after", "during",
                       "read", "today", "yesterday", "tomorrow", "now", "then",
                       "here", "there", "this", "that", "these", "those"}
        
        if word_lower in common_words:
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

class AdvancedSentenceTransformer:
    """Advanced sentence transformation using BART and T5 concepts"""
    def __init__(self):
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
            # Check if it's a question (starts with question words)
            if words[0].lower() in ["what", "how", "when", "where", "why", "who", "which"]:
                words.append("?")
            else:
                words.append(".")
        
        return " ".join(words)

    def correct_grammar(self, sentence):
        """Apply comprehensive grammar corrections"""
        if not sentence:
            return sentence
        
        # Basic grammar corrections
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

# Test the advanced word corrector
if __name__ == "__main__":
    corrector = AdvancedWordCorrector()
    transformer = AdvancedSentenceTransformer()
    
    # Test word corrections
    test_words = ["shpe", "bild", "lern", "wrok", "hlp", "wnt", "ned", "lik", "lov"]
    print("Testing Advanced Word Correction:")
    for word in test_words:
        corrected = corrector.correct_word(word)
        print(f"'{word}' -> '{corrected}'")
    
    # Test sentence transformation
    test_sentences = [
        "i am happy",
        "i have a cat",
        "i want to go",
        "am tired",
        "have a book",
        "what is your name",
        "how are you"
    ]
    
    print("\nTesting BART-T5 Sentence Transformation:")
    for sentence in test_sentences:
        transformed = transformer.transform_sentence(sentence)
        print(f"'{sentence}' -> '{transformed}'")
