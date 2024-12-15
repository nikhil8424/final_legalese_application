import os
from flask import Flask, render_template, request, flash, jsonify
from werkzeug.utils import secure_filename
import PyPDF2
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator
from collections import defaultdict
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
import spacy
import nltk
from nltk.corpus import wordnet
from transformers import pipeline
from collections import Counter
from data_list import MAHARASHTRA_CITIES, legal_keywords  # Import from data_lists.py


nltk.download('punkt')
nltk.download('punkt_tab')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Store files in this folder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit for uploads
app.config['SECRET_KEY'] = 'your_secret_key' # for flash messages
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')



# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to summarize text using Sumy
def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Adjust the number of sentences in the summary
    summary_text = "\n".join([str(sentence) for sentence in summary])
    return summary_text


# Function to extract text from image using pytesseract
def extract_text_from_image(image_path, language):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=language)
    return text

# Function to extract text from PDF using PyPDF2
def pdf_to_text(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ''  # Handle None if text extraction fails
    return text

# Function to convert PDF to images and then extract text
def pdf_to_images_and_text(pdf_path, language):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image, lang=language)
    return text

# Function to handle file upload and text extraction
def extract_text(file_path, language_code):
    if file_path.lower().endswith('.pdf'):
        if language_code == "eng":
            return pdf_to_text(file_path)
        else:
            return pdf_to_images_and_text(file_path, language_code)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return extract_text_from_image(file_path, language_code)
    else:
        raise ValueError("Only PDF and image files are supported.")

# Function to translate extracted text to English
def translate_to_english(text):
    try:
        translator = GoogleTranslator(target='en')
        lines = text.split('\n')
        batch_size = 100
        translated_text = ""
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            batch_text = "\n".join(batch)
            translated_batch = translator.translate(batch_text)
            translated_text += translated_batch + "\n"
        return translated_text.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Function to extract laws and sections from the text
def extract_laws_and_sections(text):
    pattern = re.compile(r'\bSection\s+\d+\b', re.IGNORECASE)
    sections = pattern.findall(text)
    return sections

# Function to identify document type and extract sections
def identify_document_type(text):
    text = text.lower()
    keyword_counts = defaultdict(int)
    for doc_type, keywords in legal_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text:
                keyword_counts[doc_type] += 1
    
    sections = extract_laws_and_sections(text)
    
    if keyword_counts:
        max_type = max(keyword_counts, key=keyword_counts.get)
        return max_type, sections, keyword_counts
    else:
        return 'Unknown Document Type', sections, keyword_counts

def extract_dates(text):
    date_patterns = [
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)[\s\n]*\d{1,2},[\s\n]*\d{4}",
        r"\d{1,2}[\s\n]*(?:January|February|March|April|May|June|July|August|September|October|November|December)[,\s\n]*\d{4}",
        r"\d{4}[-/]\d{2}[-/]\d{2}",
        r"\d{1,2}[-/]\d{2}[-/]\d{4}",
        r"\d{2}[-/]\d{2}[-/]\d{4}",
        r"\d{1,2}[\s\n]*(?:January|February|March|April|May|June|July|August|September|October|November|December)[\s\n]*\d{4}",
    ]
    
    dates_found = set()
    for pattern in date_patterns:
        dates_found.update(match.group() for match in re.finditer(pattern, text))

    return list(dates_found)

def extract_organization_names(text):
    organization_suffixes = [
        r"Pvt\.? Ltd\.?", r"Ltd\.?", r"LLP", r"LP", r"PA", r"PC", r"NPO", r"NGO", r"Foundation", r"Properties",
        r"Co-op", r"Cooperative Society", r"Trust", r"Section 8 Company", r"Inc\.?", r"Corp\.?", r"LLC", r"PLC",
        r"GmbH", r"S\.A\.", r"S\.R\.L\.", r"A\.G\.", r"KGaA"
    ]

    organization_pattern = r"\b[A-Z][A-Za-z\s&'-]*?\b(?:\s(?:'|\b[A-Z][A-Za-z\s&'-]+?\b))*(?:" + "|".join(organization_suffixes) + r")\b"
    
    cleaned_text = clean_text(text)
    
    organizations = set()
    for match in re.finditer(organization_pattern, cleaned_text):
        org_name = match.group().strip()
        if len(org_name.split()) <= 6 and not re.search(r'\b(?:dispute|arising|relating|agreement|binding|arbitration|rules|association|remainder|document|omitted|brevity|terms|ownership|website|code|warranty|defects|limitations|liability|witness|whereof|parties|executed|date|first|written|above|inc)\b', org_name, re.IGNORECASE):
            organizations.add(org_name)
    
    return list(organizations)

def extract_city_names(text, MAHARASHTRA_CITIES):
    city_pattern = r'\b(?:' + '|'.join(re.escape(city) for city in MAHARASHTRA_CITIES) + r')\b'
    cities_found = set()
    
    cleaned_text = clean_text(text)
    
    for match in re.finditer(city_pattern, cleaned_text, re.IGNORECASE):
        city_name = match.group().strip()
        if city_name in MAHARASHTRA_CITIES and len(city_name) > 1:
            cities_found.add(city_name)
    
    return list(cities_found)

def extract_names(text):
    doc = nlp(text)
    filtered_names = set()  # Use a set to avoid duplicates

    # Extract names using NER
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            # Normalize and clean names
            normalized_name = ' '.join(name.split())
            if len(normalized_name.split()) > 1 and not re.match(r'\b(?:This|Witness|Whereof|Sealed|Witnesseth|Principal|a|b|c|d)\b', normalized_name):
                filtered_names.add(normalized_name)

    # Convert set to list
    filtered_names = list(filtered_names)
    
    # Additional cleaning to remove unusual cases
    filtered_names = [name for name in filtered_names if re.match(r'^[A-Za-z\s]+$', name)]

    return filtered_names

def clean_text(text):
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
        return text
    
#function to convert legal language into simple english language
class TextSimplifier:
    def __init__(self):
        self.pos_map = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
        try:
            self.paraphraser = pipeline("text2text-generation", model="t5-small")  # Specify the model here
            print("Model initialized successfully.")
        except Exception as e:
            print(f"Error initializing model: {e}")
            self.paraphraser = None
    
    def find_simple_synonym(self, word, pos_tag):
       try:
           
           synonyms = wordnet.synsets(word)
           if synonyms:
                filtered_synonyms = [lemma.name().replace('_', ' ') for syn in synonyms 
                                    for lemma in syn.lemmas() if syn.pos() == self.pos_map.get(pos_tag, 'n')]
                if filtered_synonyms:
                    common_synonym = Counter(filtered_synonyms).most_common(1)[0][0]
                    return common_synonym
           return word
       except Exception as e:
            print(f"Error with WordNet: {e}")
            return word

    def paraphrase_sentence(self, sentence):
        if self.paraphraser:
            try:
                paraphrased = self.paraphraser(sentence, max_length=100, num_return_sequences=1)
                return paraphrased[0]['generated_text']
            except Exception as e:
                print(f"Paraphrasing error: {e}")
                return sentence
        else:
            print("Paraphrasing is unavailable due to initialization issues. Skipping...")
            return sentence

    def simplify_sentence(self, sentence):
        simplified_sentence = []
        for token in sentence:
            pos_tag = token.pos_
            if pos_tag in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                simplified_word = self.find_simple_synonym(token.text, pos_tag)
                simplified_sentence.append(simplified_word)
            else:
                simplified_sentence.append(token.text)
        return ' '.join(simplified_sentence)

    def simplify_text(self, text):
        doc = nlp(text)
        simplified_sentences = []

        for sentence in doc.sents:
            simplified_sentence = self.simplify_sentence(sentence)
            simplified_text = self.paraphrase_sentence(simplified_sentence)
            simplified_sentences.append(simplified_text)

        return ' '.join(simplified_sentences)
        
def simplify_analyze_text(legal_text):
    # Simplify the legal text
    simplifier = TextSimplifier()
    simplified_text = simplifier.simplify_text(legal_text)
    
    # Return the results
    return simplified_text


# Routes
@app.route("/", methods=["GET"])
def index():
    language_options = ["English", "Marathi", "Hindi", "Tamil"]
    return render_template("index.html", language_options = language_options)

@app.route("/upload", methods=["POST"])
def upload():
    extracted_text = ""
    results = {}
    if 'file' not in request.files:
        flash("No file part", 'error')
        return jsonify({"error": "No file part"})
        
    file = request.files['file']
    selected_language = request.form.get('language')
    
    if file.filename == '':
        flash('No selected file', 'error')
        return jsonify({"error": "No selected file"})
    
    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            language_dict = {
                "Marathi": "mar",
                "English": "eng",
                "Hindi": "hin",
                "Tamil": "tam"
            }
            language_code = language_dict.get(selected_language)

            if not language_code:
                flash("Please select a valid language.", 'error')
                return jsonify({"error": "Please select a valid language."})

            extracted_text = extract_text(file_path, language_code)
            translated_text = translate_to_english(extracted_text)
            results["translated_text"] = translated_text
            return jsonify(results)

        except Exception as e:
            flash(f"Error processing file: {e}", "error")
            return jsonify({"error": f"Error processing file: {e}"})

@app.route("/process", methods=["POST"])
def process():
    action = request.form['action']
    translated_text = request.form['translated_text']
    results = {}
    
    if action == "Identify":
        doc_type, sections, keyword_counts = identify_document_type(translated_text)
        sections_text = "\n".join(sections) if sections else "No sections found."
        keyword_count_details = "\n".join([f"{keyword}: {count}" for keyword, count in keyword_counts.items()])
        results["document_type"] = doc_type
        results["sections"] = sections_text
        results["keyword_counts"] = keyword_count_details
        results["action"] = "Identify"
    elif action == "Key Points":
        dates = extract_dates(translated_text)
        organizations = extract_organization_names(translated_text)
        city_names = extract_city_names(translated_text, MAHARASHTRA_CITIES)
        names = extract_names(translated_text)

        # Create detailed messages for each extracted item
        dates_text = "\n".join([f"{i}. Date: {date}" for i, date in enumerate(dates, start=1)]) if dates else "No dates found."
        organizations_text = "\n".join([f"{i}. Organization: {org}" for i, org in enumerate(organizations, start=1)]) if organizations else "No organization names found."
        cities_text = "\n".join([f"{i}. Location: {city}" for i, city in enumerate(city_names, start=1)]) if city_names else "No city names found."
        names_text = "\n".join([f"{i}. Name: {name}" for i, name in enumerate(names, start=1)]) if names else "No human names found."
        results["dates"] = dates_text
        results["organizations"] = organizations_text
        results["cities"] = cities_text
        results["names"] = names_text
        results["action"] = "Key Points"
    elif action == "Summary":
        summary = summarize_text(translated_text)
        results["summary"] = summary if summary else "No Summary"
        results["action"] = "Summary"
    elif action == "Plain Language":
        simplified_text = simplify_analyze_text(translated_text)
        results["simplified_text"] = simplified_text
        results["action"] = "Plain Language"

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)