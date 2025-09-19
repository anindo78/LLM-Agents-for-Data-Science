'''
Step 2: Load Documents in Python
Use LangChain's document loaders to read the PDF content. PyPDFLoader is straightforward for simple PDFs. UnstructuredPDFLoader
 (requires unstructured[pdf]) can handle more complex layouts but has more dependencies.
'''

import os
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATA_DIR = os.getenv('DATA_DIR', 'data')
PDF_PATH = os.path.join(DATA_DIR, 'llama2.pdf')

def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# documents = load_documents() # Call this later


'''
Split Documents

Large documents need to be split into smaller chunks suitable for embedding and retrieval. 
The RecursiveCharacterTextSplitter attempts to split text semantically (at paragraphs, sentences, and so on)
before resorting to fixed-size splits. chunk_size determines the maximum size of each chunk (in characters),
and chunk_overlap specifies how many characters should overlap between consecutive chunks to maintain context.
'''

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   is_separator_regex=False)
    split_docs = text_splitter.split_documents(documents)
    return split_docs



# chunks = split_documents(documents) # Call this later


'''
Choose and Configure Embedding Model

Embeddings transform text into vectors 
(lists of numbers) such that semantically similar text chunks have vectors that are close together in multi-dimensional space.

Option A: (Recommended for Simplicity): Ollama Embeddings
This approach uses Ollama to serve a dedicated embedding model. nomic-embed-text is a capable open-source model available via Ollama.
'''

from langchain.embeddings import OllamaEmbeddings

def get_ollama_embeddings():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print(f"Initialized Ollama embeddings with model: nomic-embed-text")
    return embeddings


# embedding_function = get_ollama_embeddings() # Call this later


'''
Set Up Local Vector Store (ChromaDB)
ChromaDB is a popular open-source vector database that can be run locally.
ChromaDB provides an efficient way to store and search vector embeddings locally. 
Using a persistent client ensures the indexed data is saved to disk and can be reloaded
without re-processing the documents every time.
'''

from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"

def create_vector_store(chunks, embedding_function, persist_directory=CHROMA_PATH):
    """Initializes or loads the Chroma vector store."""
    vectordb = Chroma(embedding_function=embedding_function,
                      persist_directory=persist_directory)
    
    vectordb.persist()
    print(f"ChromaDB vector store created at: {persist_directory}")
    return vectordb




# vector_store = create_vector_store(embedding_function) # Call this later

'''
Index Documents (Embed and Store)

This is the core indexing step where document chunks are converted to embeddings 
and saved in ChromaDB. The Chroma.from_documents function is convenient for the initial creation
and indexing. If the database already exists, subsequent additions can use vectorstore.add_documents.
'''


def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    """Indexes document chunks into the Chroma vector store."""
    vectordb = Chroma.from_documents(documents=chunks,
                                     embedding=embedding_function,
                                     persist_directory=persist_directory)
    vectordb.persist()      # ensure data is saved to disk
    print(f"Indexed {len(chunks)} document chunks into ChromaDB at: {persist_directory}")
    return vectordb

# vector_store = index_documents(chunks, embedding_function) # Call this for initial indexing



'''
Build the RAG Chain
Now, assemble the components into a LangChain Expression Language (LCEL) chain. 
This involves initializing the Qwen 3 LLM via Ollama, creating a retriever from the vector store, 
defining a suitable prompt, and chaining them together.

A critical parameter when initializing ChatOllama for RAG is num_ctx. This defines the context window size 
(in tokens) that the LLM can handle. Ollama's default (often 2048 or 4096 tokens) might be too small to 
accommodate both the retrieved document context and the user's query/prompt.

Qwen 3 models (8B and larger) support much larger context windows
(for example, 128k tokens), but practical limits depend on your available RAM/VRAM. 
Setting num_ctx to a value like 8192 or higher is often necessary for effective RAG.
'''


from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough   
from langchain_core.output_parsers import StrOutputParser   


def create_rag_chain(vector_store, llm_model="qwen3:4b", num_ctx=4096):
    """Creates a RAG chain using the provided vector store and LLM model."""

    # Initialize the LLM
    llm = ChatOllama(model=llm_model, 
                     num_ctx=num_ctx,
                     temparature=0.1) # lower temperature will make the output more deterministic
     
    print(f"Initialized ChatOllama with model: {llm_model}, num_ctx: {num_ctx}")

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(search_type="similarity", # or "mmr"
                                          search_kwargs={"k": 3})   # number of context documents to retrieve

    # Define the prompt template
    prompt_template = """Answer the question based ONLY on the following context:
                        {context}

                        Question: {question}
                        """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    print("Created prompt template for RAG chain.")

    # Assemble the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain created successfully.")         
    return rag_chain

# rag_chain = create_rag_chain(vector_store) # Call this later

'''
The effectiveness of the RAG system hinges on the proper configuration of each component.
The chunk_size and chunk_overlap in the splitter affect what the retriever finds.

Your choice of embedding_function must be consistent between indexing and querying. 
The num_ctx parameter for the ChatOllama LLM must be large enough to hold the retrieved context and
the prompt itself. A poorly designed prompt template can also lead the LLM astray.
Make sure you carefully tune these elements for optimal performance.
'''

'''
Query Your Documents
Finally, invoke the RAG chain with a question related to the content of the indexed PDF.
'''

def query_rag_chain(rag_chain, question):
    """Queries the RAG chain with a user question."""
    print(f"Querying RAG chain with question: {question}")
    response = rag_chain.invoke({"question": question})
    print("\nResponse:")
    return response



if __name__ == "__main__":
    # Load and process documents
    documents = load_documents(PDF_PATH)
    print(f"\nLoaded {len(documents)} documents from {PDF_PATH}")

    chunks = split_documents(documents)
    print(f"\nSplit documents into {len(chunks)} chunks.")

    # Set up embeddings and vector store
    embedding_function = get_ollama_embeddings()
    print(f"\n Initialized embedding function.")
    
    # Uncomment the next line for initial indexing
    # print("Attempting to index documents...")
    # vector_store = index_documents(chunks, embedding_function) 

    # For subsequent runs, load existing vector store
    vector_store = create_vector_store(chunks, embedding_function)
    print(f"\n Vector store ready.")

    # Create RAG chain
    print("\nCreating RAG chain...")
    rag_chain = create_rag_chain(vector_store, llm_model="qwen3:4b", num_ctx=4092)
    print("\nRAG chain is ready.")

    print("\nYou can now ask questions about the document content.\n")

    # Example query
    question = "What are the key features of LLaMA 2 models?"
    answer = query_rag_chain(rag_chain, question)
    print(answer)

    '''
    I think that's it. Let me write the answer.
    </think>

    Based solely on the provided context, the key features of Llama 2 models are:

    1. Trained on a new mix of publicly available data (updated from Llama 1)
    2. Increased pretraining corpus size by 40%
    3. Doubled context length of the model
    4. Adopted grouped-query attention (referenced from Ainslie et al., 2023)
    5. Released variants with 7B, 13B, and 70B parameters
    6. Includes Llama 2-Chat (a fine-tuned version optimized for dialogue use cases)
    '''


    question2 = "Summarize fine tuning approach of LLaMA 2 models?"
    answer2 = query_rag_chain(rag_chain, question2)
    print(answer2)

    '''
    I'll craft a concise summary based on this information that answers the question about the fine tuning approach of LLaMA 2 models.
</think>

Based on the provided context, here's a summary of the fine tuning approach of LLaMA 2 models:

The LLaMA 2 fine-tuning process involves a multi-stage approach that begins with pretraining of Llama 2 using publicly available online sources.
 The specific fine-tuning methodology for Llama 2-Chat includes:

1. Initial supervised fine-tuning to create the first version of Llama 2-Chat
2. Iterative refinement through Reinforcement Learning with Human Feedback (RLHF) methodologies, specifically using:
   - Rejection sampling
   - Proximal Policy Optimization (PPO)
3. Development of iterative reward modeling that runs in parallel with model enhancements to ensure reward models remain within distribution
4. Implementation of Ghost Attention (GAtt) to help control dialogue flow across multiple turns
5. Safety-focused approaches including safety-specific data annotation, tuning, red-teaming, and iterative evaluations

The paper states that Llama 2-Chat models were developed through "several months of research and iterative applications of 
alignment techniques, including both instruction tuning and RLHF," requiring significant computational resources.
 The fine-tuning process was designed to improve both helpfulness and safety performance while maintaining compatibility with
   the open-source community.
'''