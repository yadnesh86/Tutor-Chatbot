import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set up SSL context for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
try:
    with open(file_path, "r") as file:
        intents = json.load(file)
except Exception as e:
    st.error(f"Error loading intents file: {e}")

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter

    # Inject custom CSS for better UI
    st.markdown("""
    <style>
    body {
        background-color: #2c3e50;
        color: #ecf0f1;
    }

    .stApp {
        background-color: #2c3e50;
        color: #ecf0f1;
    }

    .sidebar .sidebar-content {
        background-color: #34495e;
        color: #ecf0f1;
    }

    .custom-input-container input {
        width: 100%;
        color: #2c3e50;
        background-color: #ecf0f1;
        border: 2px solid #95a5a6;
        border-radius: 5px;
        padding: 8px;
        outline: none;
        font-size: 16px;
    }

    .stTextArea textarea {
        background-color: #ecf0f1;
        color: #2c3e50;
        border: 2px solid #95a5a6;
        border-radius: 5px;
        padding: 8px;
        font-size: 16px;
    }

    .stButton button {
        background-color: #2980b9;
        color: white;
        border-radius: 5px;
    }

    h1 {
        color: #ecf0f1;
        text-align: center;
    }

    h2, h3 {
        color: #1abc9c;
    }

    .stMarkdown {
        padding-top: 30px;
        font-size: 18px;
        color: #ecf0f1;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Set the title for the page
    st.title("Tutor Chatbot")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Empowering learning with Technology: Your Guide to Educational Tools and Innovation.")

        # Check if the chat_log.csv file exists
        if not os.path.exists('chat_log.csv'):
            try:
                with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
            except Exception as e:
                st.error(f"Error creating chat log file: {e}")

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Write chat to CSV log
            try:
                with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([user_input_str, response, timestamp])
            except Exception as e:
                st.error(f"Error writing to chat log file: {e}")

            # Check if the chatbot response contains "goodbye"
            if any(phrase in response.lower() for phrase in ['goodbye', 'bye']):
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error reading chat log: {e}")

    elif choice == "About":
        st.write("This chatbot is designed to assist students with their Education related queries.")
        st.subheader("Features:")
        st.write(""" 
        - Provides answers to frequently asked questions about Educational Tools and Innovation.
        - Saves conversation history for review.
        - Built using Natural Language Processing (NLP) techniques.
        """)

if __name__ == '__main__':
    main()
