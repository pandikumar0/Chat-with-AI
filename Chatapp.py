from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import google.ai.generativelanguage as glm
import io

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))          # for configuration of gemini api

def get_gemini_response(query):
    model = genai.GenerativeModel("gemini-pro")                 # for question and query systems               
    chat = model.start_chat(history = [])
    response = chat.send_message(query, stream=True)            # steam = True for asynchronous handling
    return response

def image_to_byte_array(image):                                 # Systems using Byte array for Image, convertion
    ByteArr = io.BytesIO()
    image.save(ByteArr, format=image.format)
    ByteArr = ByteArr.getvalue()
    return ByteArr

def get_gemini_response_image(image_prompt, image):             # Gemini response on image and image prompt
    model = genai.GenerativeModel("gemini-pro-vision")
    response = model.generate_content(
        glm.Content(
            parts =[
                glm.Part(text= image_prompt),
                glm.Part(
                    inline_data = glm.Blob(
                        mime_type = "image/jpeg",
                        data = image_to_byte_array(image)
                    )
                )
            ]
        )
    )
    return response

    
    
st.set_page_config(page_title = "Visual Question Answering System")                     # Default Loading style of Page

gemini_pro, gemini_vision_pro = st.tabs(["Gemini Pro", "Gemini Pro Vision"])
def main():
    with gemini_pro:

        st.header("Chat with Gemini AI")

        input = st.text_input("Input: ", key = "input")
        submit = st.button("Ask Question")

        if submit and input:
            response = get_gemini_response(input)                       # Get Response from API
            st.subheader("The response is: ")
            for chunk in response:
                st.write(chunk.text)

    with gemini_vision_pro:
        st.header("Interact with Image")

        image_prompt = st.text_input("Input: ", key = "input_img")

        file = st.file_uploader("Choose the Image", accept_multiple_files = False, type=['png', 'jpg', 'jpeg'])

        if file:
            st.image(Image.open(file))

        submit = st.button("Get Response")

        if submit and file:
                if image_prompt!="":
                    image = Image.open(file)
                    response = get_gemini_response_image(image_prompt, image)

                    response.resolve()

                    st.write(response.text)

if __name__ == "__main__":
    main()

