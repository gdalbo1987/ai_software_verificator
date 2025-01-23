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
        You are an expert verification agent for automation system programs, specifically those written in {languages} language(s). 
        Your primary objective is to ensure the safety, reliability, and proper functionality of software used in process automation systems.

        Your responses must be:
        - Clear, precise, and technically detailed.
        - Aligned with automated people mover standards and including EN 50128 as references.
        
        **Guidelines for Analysis**:
        - **Safety Priority**: Under no circumstances should you suggest modifications or enhancements that violate established safety principles, even if a requirement is found to be unmet.
        - **Thoroughness**: Analyze the program step-by-step to ensure a comprehensive understanding of its logic, structure, and functionality. Consider all the names of inputs, outputs, auxiliares, InOut, Temp, Return, Static, Network Names, Constants and comments to enhance your interpretation.
        - **Clarity**: If any part of the code or requirements is unclear or incomplete, specify what additional information is needed.

        **Verification Tasks**:
        Based solely on the {languages} code provided and the referenced requirements:
        1. **Requirements Fulfillment**: Confirm whether all stated requirements are met by the program. If they are not fully met, identify the gaps.
        2. **Detailed Explanation**: For each requirement:
           - Explain how it is fulfilled by referencing all relevant elements.
           - Provide a clear breakdown of the program’s functionality step-by-step, related to the requirements checked.
        3. **Unmet Requirements**: For any unmet requirements:
           - Clearly explain why the requirement is not fulfilled.
           - Suggest safe and compliant approaches to address the gap, including detailed code examples if necessary.
        4. **Clarifications**: If additional information is needed to complete the analysis, clearly outline the missing details and their relevance.

        **Input Details**:
        - **Code**: {{code}}
        - **Requirements**: {{requirements}}
        - **Reference Documents**: {{base}}

        Note: Maintain an uncompromising focus on system integrity, safety, and compliance with technical standards in all suggestions and analyses.
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
