import streamlit as st
import streamlit_toggle as toggle

from langchain.chains.question_answering import load_qa_chain

from lllm_setup import *
from file_loader import *
from keys import *

load_keys_into_os()

# CoreUI
st.title('🤖🦾 GPT FileBot')
prompt = st.text_input('Input your prompt here')
objectivity_toggle = toggle.st_toggle_switch(label='Objectivity level high', 
                    key="OBJECTIVITY_TOGGLE", 
                    default_value=True, 
                    label_after = False)


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
file_is_ok = uploaded_file is not None and is_Pdf(uploaded_file)

if file_is_ok:
    st.write("Filename: ", uploaded_file.name)

    local_path = 'temp.pdf'
    with open(local_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            f.close()

    with st.spinner('Running...'):
        # AI Logic

        ob_level = ObjectiviTyLevel.FILE_BASED if objectivity_toggle == True else ObjectiviTyLevel.NORMAL
        llm = make_llm(objectivity=ob_level)

        (_, chroma_store) = create_vectorstore(local_path)

        chain = load_qa_chain(llm, chain_type="stuff")

        # If the user hits enter
        if prompt:
            search = chroma_store.similarity_search(prompt)
            
            if len(search) > 0:
                response = chain.run(input_documents=search, question=prompt)
                st.write(response)
            
                with st.expander('Document Similarity Search'):    
                    for page in search:
                        st.write(page.page_content)
                        st.markdown('---------')
            else:
                 st.error('No similarities found in the document')

elif file_is_ok == False and uploaded_file is not None:
    st.error('Please upload a PDF file', icon='🚨')
