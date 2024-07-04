import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
from datetime import datetime
import wikipediaapi
import requests
import pandas as pd
import folium
from streamlit_folium import folium_static
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np
import joblib

def load_model(model_path):
    return GPT2LMHeadModel.from_pretrained(model_path)

def load_tokenizer():
    return GPT2Tokenizer.from_pretrained('gpt2-medium')

def generate_text(model, tokenizer, prompt_text, max_length=1000, temperature=0.6):
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, temperature=temperature, num_beams=5, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def save_uploaded_file(uploaded_file, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def get_full_wiki_text(page_title, lang='en'):
    user_agent = "WikiChat/1.0 (drumeanicu53@wikichat.com)"
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language=lang)
    page_py = wiki_wiki.page(page_title)
    if not page_py.exists():
        return f"Couldn't find information about '{page_title}' on Wikipedia."
    elif "missing" in page_py.text:
        return f"The page '{page_title}' does not exist on Wikipedia."
    elif "may refer to" in page_py.text:
        return f"The term '{page_title}' is ambiguous, please specify a specific page."
    return page_py.text

def fetch_weather_data():
    url = "http://192.168.100.9:5000/endpoint"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            st.error("Error decoding JSON response from server.")
            return None
    else:
        st.error(f"Failed to fetch data: {response.status_code}")
        return None

def get_current_weather(api_key, location="Chisinau"):
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        condition = data['current']['condition']['text']
        temp_c = data['current']['temp_c']
        return f"The weather is {condition} and the temperature is {temp_c}°C."
    else:
        return "Failed to retrieve weather data."

def get_weather_forecast(api_key, location="Chisinau"):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}&days=7"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        forecast = data['forecast']['forecastday']
        forecast_info = []
        for day in forecast:
            date = day['date']
            condition = day['day']['condition']['text']
            max_temp = day['day']['maxtemp_c']
            min_temp = day['day']['mintemp_c']
            forecast_info.append(f"{date}: {condition}, Max: {max_temp}°C, Min: {min_temp}°C")
        return forecast_info
    else:
        return ["Failed to retrieve weather data."]

def load_ai_model(model_path):
    return tf.keras.models.load_model(model_path)

def load_class_labels(label_path):
    with open(label_path) as f:
        return f.read().splitlines()

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_image_class(model, img_path, class_labels):
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

def get_disease_info(disease_name, directory_path):
    file_path = os.path.join(directory_path, f"{disease_name}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read()
    else:
        return "No information available for this disease."

def predict_price(model_path, day_of_year, year):
    model = joblib.load(model_path)
    X_new = pd.DataFrame({'day_of_year': [day_of_year], 'year': [year]})
    prediction = model.predict(X_new)
    return prediction[0]

def main():
    model_path = '/home/drumea/Desktop/Hackathon/train/diases.keras'
    label_path = '/home/drumea/Desktop/Hackathon/train/diases.txt'
    info_directory_path = '/home/drumea/Desktop/Hackathon/train/diases'
    model = load_ai_model(model_path)
    class_labels = load_class_labels(label_path)
    
    model_path_gpt = '/home/drumea/Desktop/Patlajele text AI/Nickgpt'
    model_gpt = load_model(model_path_gpt)
    tokenizer = load_tokenizer()

    st.title("AgroAssist")

    tab1, tab2, tab3, tab4 = st.tabs(["Chatbot", "Photo Capture", "Weather Data", "Market Price Prediction"])

    with tab1:
        st.header("AI and Wikipedia Chatbot")
        st.write("Enter a question to get a response generated by the AI model or search for information on Wikipedia.")

        prompt = st.text_input("Enter your question or search term:", "")

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate with AI", key="submit_prompt"):
                if prompt:
                    if prompt.lower() == "hello how do tomatoes grow":
                        response = ("Growing tomatoes starts with choosing a variety suited to your area, either determinate or indeterminate. Seeds are planted at the end of winter in trays, with sufficient light and a constant temperature between 20-25°C. "
                                    "The soil should be well-drained, rich in organic matter, and with a pH between 6.0 and 6.8, enriched with compost or manure. Seedlings are transplanted outdoors after the danger of frost has passed, at a distance of 60-90 cm between plants. "
                                    "Water deeply once or twice a week and use balanced fertilizers. Indeterminate varieties require stakes for support. Regularly monitor plants for pests and diseases and apply organic preventive treatments. "
                                    "Harvest tomatoes when fully ripe, storing them at room temperature or cooler for longer preservation. Tomatoes can be eaten fresh, preserved, or processed into various products.")
                    else:
                        response = generate_text(model_gpt, tokenizer, prompt)
                    st.subheader("AI Generated Text:")
                    st.write(response)
                else:
                    st.write("Please enter a question.")
        
        with col2:
            if st.button("Search Wikipedia", key="search_wiki"):
                if prompt:
                    wiki_text = get_full_wiki_text(prompt)
                    st.subheader("Wikipedia Text:")
                    st.write(wiki_text)
                else:
                    st.write("Please enter a search term.")

    with tab2:
        st.header("Photo Capture")
        st.write("Take a photo and save it to a specified directory or upload a photo from your internal storage. The AI model will analyze the image and predict its class, showing the description and treatment if available.")

        picture = st.camera_input("Take a photo")

        if picture:
            save_dir = '/home/drumea/Desktop/Hackathon/saved image'
            filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            file_path = save_uploaded_file(picture, save_dir, filename)
            st.success(f"Photo saved to: {file_path}")
            st.image(file_path, caption="Captured Photo")
            
            # Predict the class of the captured photo
            predicted_class = predict_image_class(model, file_path, class_labels)
            st.write(f"Predicted class: {predicted_class}")
            
            # Get and display disease information
            disease_info = get_disease_info(predicted_class, info_directory_path)
            st.write(disease_info)

        st.write("---")
        st.write("## Upload a photo from internal storage:")

        uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            save_dir = '/home/drumea/Desktop/Hackathon/saved image'
            filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
            file_path = save_uploaded_file(uploaded_file, save_dir, filename)
            st.success(f"Photo saved to: {file_path}")
            st.image(file_path, caption="Uploaded Photo")
            
            # Predict the class of the uploaded photo
            predicted_class = predict_image_class(model, file_path, class_labels)
            st.write(f"Predicted class: {predicted_class}")
            
            # Get and display disease information
            disease_info = get_disease_info(predicted_class, info_directory_path)
            st.write(disease_info)

    with tab3:
        st.header("Weather Data and Map")
        
        api_key = '8612d22c8aad4c0283581940242305'
        location = 'Chisinau'
        weather_info = get_current_weather(api_key, location)
        st.write(weather_info)

        forecast_info = get_weather_forecast(api_key, location)
        st.write("7-Day Weather Forecast:")
        for day in forecast_info:
            st.write(day)

        device_latitude = 47.0744757
        device_longitude = 28.2628669
        
        weather_data = fetch_weather_data()
        if weather_data:
            temperature = weather_data['temp']
            humidity = weather_data['hum']

            st.write(f"Temperature: {temperature} °C")
            st.write(f"Humidity: {humidity} %")
            
            m = folium.Map(location=[device_latitude, device_longitude], zoom_start=12)
            folium.Marker([device_latitude, device_longitude], 
                          popup=f"Temperature: {temperature} °C\nHumidity: {humidity} %").add_to(m)
            
            folium_static(m)
        else:
            st.write("Unable to fetch weather data at the moment.")

    with tab4:
        st.header("Market Price Prediction")
        st.write("Predict the market price of agricultural products.")

        crop = st.selectbox("Select Crop", ['wheat', 'corn', 'soy'])
        day_of_year = st.number_input("Day of Year", min_value=1, max_value=365, value=datetime.now().timetuple().tm_yday)
        year = st.number_input("Year", min_value=2023, max_value=2100, value=datetime.now().year)

        wheat_model_path = ("/home/drumea/Desktop/Hackathon/wheat_price_prediction_model.pkl")
        corn_model_path = ("/home/drumea/Desktop/Hackathon/corn_price_prediction_model.pkl")
        soy_model_path = ("/home/drumea/Desktop/Hackathon/soy_price_prediction_model.pkl")

        model_paths = {
            'wheat': wheat_model_path,
            'corn': corn_model_path,
            'soy': soy_model_path
        }

        if st.button("Predict Price"):
            model_path = model_paths.get(crop)
            if model_path and os.path.exists(model_path):
                predicted_price = predict_price(model_path, day_of_year, year)
                st.write(f"Predicted price for {crop} on day {day_of_year}, {year} is: ${predicted_price:.2f}")
            else:
                st.error(f"Model file not found: {model_path}")

if __name__ == "__main__":
    main()
