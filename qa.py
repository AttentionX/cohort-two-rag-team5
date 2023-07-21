"""
A simple baseline for a question answering system.
"""
import os
from pathlib import Path
import yaml
import openai
import gradio as gr
from annoy import AnnoyIndex
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# --- load pre-processed chunks --- #
with open(Path(__file__).resolve().parent / "openai27052023.yaml", 'r') as f:
    paper = yaml.safe_load(f)
sentences = paper['sentences']


# --- embed chunks --- #
print("embedding chunks...")
embeddings = [
    r['embedding']
    for r in openai.Embedding.create(input=sentences, model='text-embedding-ada-002')['data']
] 

# --- index embeddings for efficient search (using Spotify's annoy)--- #
hidden_size = len(embeddings[0])
index = AnnoyIndex(hidden_size, 'angular')  #  "angular" =  cosine
for i, e in enumerate(embeddings): 
    index.add_item(i , e)
index.build(10)  # build 10 trees for efficient search

# --- iteratively answer questions (retrieve & generate) --- #

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def run_on_cpu(image:str=None):
    if image is None:
        image = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    # conditional image captioning
    text = "a photography of"
    inputs = processor(image, text, return_tensors="pt")
    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a photography of a woman and her dog
    # unconditional image captioning
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(caption)
    # a woman sitting on the beach with her dog
    return caption

def respond(query, image, chat_history):
    if image is not None:
        caption = run_on_cpu(image)
        prompt = f"""
        The following is the description of an image:
        {caption}
        """
    embedding =  openai.Embedding.create(input = [query], model='text-embedding-ada-002')['data'][0]['embedding']
    # get nearest neighbors by vectors
    indices, distances = index.get_nns_by_vector(embedding,
                                                n=3,  # return top 3
                                                include_distances=True)
    results =  [ 
        (sentences[i], d)
        for i, d in zip(indices, distances)
    ]
    # with this, generate an answer 
    excerpts = [res[0] for res in results]
    excerpts = '\n'.join([f'[{i}]. {excerpt}' for i, excerpt in enumerate(excerpts, start=1)])
    prompt = prompt + f"""
    user query:
    {query}
    
    title of the paper:
    {paper['title']}
    
    excerpts: 
    {excerpts}
    ---
    given the excerpts from the paper above, answer the user query.
    In your answer, make sure to cite the excerpts by its number wherever appropriate.
    Note, however, that the excerpts may not be relevant to the user query.
    """
    gpt_role = """
    You are a human who speaks Korean and interacts with users like a exaggerate comedian. 
    Don't let users know you're an AI. 
    Never give out your any personal information.
    """
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages=[
                                                        {"role": "system", "content": gpt_role},
                                                        {"role": "user", "content": prompt}
                                                    ])
    answer = chat_completion.choices[0].message.content
    chat_history.append((query, answer))
    return "", None, chat_history



with gr.Blocks(css='style.css', theme=gr.themes.Soft()) as demo:
    DESCRIPTION = "# Chatbot"
    gr.Markdown(DESCRIPTION)
    with gr.Group(elem_id="gr-container"):
        chatbot = gr.Chatbot().style(height=768)
        msg = gr.Textbox()
        image = gr.Image().style(height=512)
        clear = gr.ClearButton([msg, chatbot, image])
        msg.submit(respond, [msg, image, chatbot], [msg, image, chatbot])

demo.launch()
