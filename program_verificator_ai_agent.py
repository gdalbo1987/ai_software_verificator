import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import shutil
import os

key = st.secrets["api_key"]

st.set_page_config(page_title = 'AI Software Verificator', layout = 'wide')

st.image('Coester.jpg', width= 200)
st.title('AeroGRU Software Verificator AI Agent')
st.write('AI Agent built using as reference the Standard EN 50128 - Railway applications — Communication, signalling and processing systems — Software for railway control and protection systems')
st.write('Version 0.3')
st.divider()

st.sidebar.title('Data entry:')

uploaded_files = st.sidebar.file_uploader('Load the files for verification:', type = ['pdf'], accept_multiple_files = True)

langs = st.sidebar.text_input('Program languages:')



if uploaded_files is not None:    
    
    requirements = st.chat_input('Insert the requirements to be verified:')

    if requirements:
        st.success(f'Requirements sent for verification:\n {requirements}')


    if uploaded_files is not None and requirements is not None:   

        all_documents = []

        # Certifique-se de que o diretório temporário existe
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            # Salvar o arquivo temporariamente
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Processar o PDF com PyPDFLoader
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            all_documents.extend(documents)

        shutil.rmtree(temp_dir)

        if all_documents is not None:
            st.sidebar.success('Documents loaded successfully!')
        
        embeddings = OpenAIEmbeddings(openai_api_key = key)

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Tamanho das partes
        chunk_overlap=100,  # Sobreposição entre partes
        separators=["\n\n", "\n", ".", " ", ""]
        )
        
        vectorstore_dir = 'saved_vectorstores'

        aux_path = os.path.join(vectorstore_dir, 'aux_vectorstore')        

        if os.path.exists(aux_path):
            # Carregar vectorstores existentes
            aux_vectorstore = FAISS.load_local(aux_path, embeddings, allow_dangerous_deserialization = True)
            
        else:
            # Criar pasta para salvar as vectorstores
            os.makedirs(vectorstore_dir, exist_ok=True)
            
            folder_path = 'base_docs'

            pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]

            base_documents = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(folder_path, pdf_file)  # Constrói o caminho completo
                loader1 = PyPDFLoader(pdf_path)
                base_docs = loader1.load()
                base_documents.extend(base_docs)
            
            base = text_splitter.split_documents(base_documents)
            
            aux_vectorstore = FAISS.from_documents(base, embeddings)
            
            aux_vectorstore.save_local(aux_path)
                    
        texts = text_splitter.split_documents(all_documents)

        vectorstore = FAISS.from_documents(texts, embeddings)

        retriever = vectorstore.as_retriever()

        base_retriever = aux_vectorstore.as_retriever()
        
        model = ChatOpenAI(model_name = 'gpt-4o-mini', temperature = 0, openai_api_key = key)

        languages = langs

        prompt_text = f"""
        You are a verification agent for automation system programs, specifically programs written in {languages} language(s).
        Your primary focus is ensuring the safety and proper functionality of the software used for process automation.
        Your response must be clear, precise, and based on high-level technical language.
        Use the reference document, including the Standard EN 50128 as technical resources to help you write the verification report.

        Important: 
        - Under no circumstances should you suggest examples, modifications, enhancements, or improvements that violate safety principles, even if a requirement is found to be unmet.
        - Always prioritize the integrity and safety of the system. If a requirement cannot be fulfilled without violating safety principles, explain why without suggesting unsafe modifications.
        - Evaluate each requirement separately from the other.
        - Before to conclude if a requirement is met or unmet, verify you response carefully.

        Examples of actions that violate safety principles:
        - Allowing doors to open while the vehicle is moving.
        - Applying brakes without considering deceleration limits defined by standards.
        - Allowing the vehicle to depart without all safety conditions being met.
        - Permitting the vehicle's speed to exceed the defined setpoint according to standards.

        Based only on the {languages} codes provided and referred requirements, describe:

        1. Are all requirements met by the program?
        2. Explain how the requirements are met, detailing all variables, all methods, all logics, all inputs, all outputs, all contacts, all coils, all memories, etc.
        3. If any requirement is not met, clearly comment on what is not fulfilled and explain why it is not met.
        4. Suggest how unmet requirements can be fulfilled, providing detailed code examples if necessary. However, do not propose changes that conflict with safety principles.

        Code: {{code}}

        Requirements: {{requirements}}

        Reference document: {{base}}        
        """

        prompt = ChatPromptTemplate.from_template(prompt_text)
        output_parser = StrOutputParser()

        setup_and_retrieval = RunnableParallel(
        {'requirements': RunnablePassthrough(),
        'code': retriever,
        'base': base_retriever}
        )

        chain = setup_and_retrieval | prompt | model | output_parser

        response = []

        text_placeholder = st.empty()

        for chunk in chain.stream(requirements):
            chk = chunk
            response.append(chk)
        
            if response is not None:

                container = st.container(border = True)

                with container:
            
                    text_placeholder.write("".join(response))
