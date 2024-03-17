### Imports
import streamlit as st
import langchain
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
import os
import io
import sys
import boto3
import time
from datetime import datetime
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, Tool, tool
from langchain import LLMMathChain
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
import json
from contextlib import redirect_stdout
from typing import Optional, Type


### - Layout components --
## I put these at the top because Streamlit runs from the top down and 
## I need a few variables that get defined here. 

## Layout configurations
st.set_page_config(
    page_title='AWS Demo App', 
    layout="wide",
    initial_sidebar_state='collapsed',
)
## CSS is pushed through a markdown configuration.
## As you can probably guess, Streamlit layout is not flexible.
## It's good for internal apps, not so good for customer facing apps.
padding_top = 15
st.markdown(f"""
    <style>
        .block-container, .main {{
            padding-top: {padding_top}px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

## UI Elements starting with the Top Graphics
col1, col2, col3 = st.columns( [1,8,1] )
col1.image('AderasBlue2.png', width=70)
# col1.image('AderasText.png', width=50)
col2.title('AWS Doc Discussion App Demo')
## Add a sidebar
with st.sidebar: 
    mydemo = st.selectbox('Select User Role', ['PUBLIC', 'IT'])
    st.markdown("*:violet[This would be provided by AD in production]*")
    safe_prompts = st.checkbox('Safe Prompts', value=False)
    st.markdown("*:violet[Provides added prompt protections]*")
    show_detail = st.checkbox('Show Details', value=True)
    st.markdown("*:violet[Show additional details in output]*")
    st.markdown("---")
    tz = st.container()

with st.expander("**:blue[App Overvew]**"):
    st.write("This apps incorporates select documents into the final answer presented to the user, through the following architecture.")
    st.image('architecture.png', use_column_width=None, caption="Document RAG architecture")
    st.write("Documents are chunked into smaller blocks of text. An embedding vector is created and a record loaded into a vector database.")
    st.write("When a user askes a question, this is converted into a vector and the vector store returns the most similar blocks of texts. A prompt is constructed using a system prompt, the user's question, and the resulting blocks of text. A final answer is created by the LLM and returned to the user.")
    st.write("If \"Safe Prompts\" is selected, the model analyzes the question relative to the agency's standards. The question is only processed if it passes all standards. This is in addition to ethical standards enforced by the model. ")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Ask away..."):
    start = datetime.now()
    tz.write("Start: "+str(start))
    



    try:
        # conversation = ConversationChain(
        #     llm=cl_llm, verbose=True, memory=ConversationBufferMemory() #memory_chain
        # )

        prompt_template = PromptTemplate.from_template("""
Human: The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context. If the AI does not know
the answer to a question, it truthfully says it does not know.

Current conversation:
<conversation_history>
{history}
</conversation_history>

Here is the human's next reply:
<human_reply>
{input}
</human_reply>

Assistant:
""")
        # conversation.prompt = prompt_template

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if show_detail:
            f = io.StringIO()
            with redirect_stdout(f):
                with st.spinner("Processing..."):
                    print("Here is the prompt template:\n")
                    print(prompt_template)
                    response = "Yes"
                    # response = conversation.predict(input=prompt)
        else:
            with st.spinner("Processing..."):
                response = "No"
                # response = conversation.predict(input=prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})    
        st.chat_message('assistant').write(response)

        if show_detail:
            with st.expander('Details', expanded=False):
                s = f.getvalue()
                st.write(s)

    except ValueError as error:
        raise error

    tz.write("End: "+str(datetime.now()))
    tz.write("Duration: "+str(datetime.now() - start))
