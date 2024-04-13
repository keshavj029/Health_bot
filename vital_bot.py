from fastapi import FastAPI, HTTPException
import os
from pydantic import BaseModel
from langchain_community.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ['GOOGLE_API_KEY'] = 'AIzaSyDpZ0Wv2pVRnaUyDDPyx13PejTPU5wN1W8'
llm = GooglePalm(temperature=0.3)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

class InputData(BaseModel):
    question: str

@app.post("/predict")
@app.exception_handler(ValueError)

async def generate_text(data:InputData):

    title_template = PromptTemplate(
        input_variables=['chat_history', 'question'],
        template = "By using context from memory {chat_history}, Act as a mental and physical therapist and answer the following  : {question} "

    )

    title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key='output' , memory=memory)
    response = title_chain({'question': data.question})
    return {'content': response['output']}
