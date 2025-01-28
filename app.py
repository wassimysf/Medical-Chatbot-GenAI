from flask import Flask, render_template, request
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_name = "medaibot"

docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize LLM and chains
llm = OpenAI(temperature=0.4, max_tokens=500)

# Define the system prompt (updated for your use case)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "the answer concise."
    "\n\n"
    "{context}\n\n"
    "Question: {input}"
)

# Create the prompt using the system_prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
])

# Create the question-answer chain with the updated prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create the retrieval-augmented generation (RAG) chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Invalid input"
    
    # Retrieve the context (i.e., related documents from the vector store)
    context = retriever.retrieve(msg)
    
    # Run the RAG chain with both 'input' and 'context'
    response = rag_chain.invoke({"input": msg, "context": context})
    
    return str(response.get("answer", "I could not process your query."))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
