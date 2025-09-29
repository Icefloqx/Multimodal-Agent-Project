#Import Libraries
import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_groq import ChatGroq
from IPython.display import display, Image as IPyImage
from PIL import Image
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration

#Load the API
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    print("Groq API key loaded successfully!")
else:
    print("Error: Missing Groq API key in .env file")

#Define state structure
class State(TypedDict):
    query: str
    image_path: str
    image_description: str
    ocr_text: str
    response: str

#Initialize the Groq LLM 
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

#testing the model
result = llm.invoke("what is a cat")
result.content

#initialize the BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_image(state: State) -> State:
    """Generate a caption for the image using BLIP"""
    try:
        raw_image = Image.open(state["image_path"]).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt")
        out = blip_model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        caption = f"Could not caption image: {str(e)}"
    return {"image_description": caption}

def extract_text(state: State) -> State:
    """Extract visible text from the image using OCR"""
    try:
        raw_image = Image.open(state["image_path"])
        ocr_result = pytesseract.image_to_string(raw_image)
    except Exception as e:
        ocr_result = f"OCR failed: {str(e)}"
    return {"ocr_text": ocr_result.strip()}

#Initialise Groq
def reason_with_groq(state: State) -> State:
    """Use Groq LLM to reason on query + image description + OCR text"""
    prompt = ChatPromptTemplate.from_template("""
    You are a multimodal AI assistant. First describe the image, then answer the user's query.
    
    - Image description: {image_description}
    - OCR text: {ocr_text}
    - User query: {query}
    
    Provide a thoughtful, detailed response.
    """)
    chain = prompt | llm
    response = chain.invoke({
        "image_description": state["image_description"],
        "ocr_text": state["ocr_text"],
        "query": state["query"]
    }).content
    return {"response": response}

#Crafting the workflow
workflow = StateGraph(State)

workflow.add_node("describe_image", describe_image)
workflow.add_node("extract_text", extract_text)
workflow.add_node("reason_with_groq", reason_with_groq)

workflow.set_entry_point("describe_image")
workflow.add_edge("describe_image", "extract_text")
workflow.add_edge("extract_text", "reason_with_groq")
workflow.add_edge("reason_with_groq", END)

app = workflow.compile()

#Running the agent
def run_agent(query: str, image_path: str) -> dict:
    result = app.invoke({"query": query, "image_path": image_path})
    return {
        "image_description": result["image_description"],
        "ocr_text": result["ocr_text"],
        "response": result["response"]
    }

import gradio as gr

def multimodal_agent_ui(query, image):
    if image is not None:  # Case 1: Image provided
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        result = run_agent(query, temp_path)
        return f""" **Image Description:** {result['image_description']}  
**OCR Text:** {result['ocr_text']}  
**Answer:** {result['response']}"""
    
    else:  #Only text question
        # Just use Groq LLM directly
        prompt = f"User question: {query}\n\nAnswer thoughtfully as an AI assistant."
        response = llm.invoke(prompt).content
        return f"**Answer (Text-only):** {response}"

# Create UI
demo = gr.Interface(
    fn=multimodal_agent_ui,
    theme='Yntec/HaleyCH_Theme_Orange_Green',
    inputs=[
        gr.Textbox(label="Your Question"),
        gr.Image(type="pil", label="Upload an Image (Optional)")
    ],
    outputs=gr.Markdown(label="Agent Response"),
    title=" Multimodal AI Agent",
    description="Ask any question. Optionally upload an image for multimodal reasoning."
)

# Launch UI
if __name__ == "__main__":
    demo.launch()