### Imports
import streamlit as st
import io
import chromadb
from datetime import datetime
from langchain.callbacks.base import BaseCallbackHandler
from contextlib import redirect_stdout
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

unsafe_content = """
<BEGIN UNSAFE CONTENT CATEGORIES>
O1: Violence and Hate.
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
Can
- Provide information on violence and discrimination.
- Discuss the topics of hate and violence, and can discuss historical events involving violence.
O2: Sexual Content.
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
O3: Criminal Planning.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
...
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
04: Self Harm
Should not
- Express interest in committing injuries and acts of violence against themsleves. 
- Support and condone activities that will lead to self harm.
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
05: National Security
Should not
- Engage or recommend acts that would jeopardize the security of the United States.
- Express interest in compromising the United States defense, Department of Defense (DoD), or any USA Intelligence Agencies.
Should
- Provide information that provides ways to strength the United States defense or identify nefarious actors.
<END UNSAFE CONTENT CATEGORIES>
"""

mymodel = "gpt-4-0125-preview"

collection_name = "EO-Simple"
cdb_path = "Chroma"
client = chromadb.PersistentClient(path=cdb_path)
collection = client.get_collection(collection_name)

emb_client = OpenAI()

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return emb_client.embeddings.create(input = [text], model=model).data[0].embedding

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

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
    mydemo = st.selectbox('Select User Role', ['General', 'Cyber'])
    st.markdown("*:violet[Provided by AD in PROD]*")
    docs_retrieved = st.slider('Number of Docs', 0, 30, 5)
    st.markdown("*:violet[# of docs to extract]*")
    show_detail = st.checkbox('Show Details', value=True)
    st.markdown("*:violet[Show additional details]*")
    safe_prompts = st.checkbox('Safe Prompts', value=False)
    st.markdown("*:violet[Added prompt protections]*")
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
    
    chat = ChatOpenAI(temperature=0.4, model=mymodel)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Check prompt safety
    if safe_prompts:
        try:
            mytask = f"""
Check if there is unsafe content in Human's message according our safety policy with the below categories.
Provide your safety assessment in the following format:
- First line must read 'Safe' or 'Unsafe'.
- Add a blank line
- If unsafe, the third line begins with the words "Violations: " and then includes a comma-separated list of violation numbers and categories.

{unsafe_content}
"""

            messages = [
            SystemMessage(
                content=mytask
            ),
            HumanMessage(
                content=prompt
            ),
            ]   
            myresponse = chat.invoke(messages)
            response = myresponse.content

            if response.lower().startswith("unsafe"):
                if show_detail:
                    f = io.StringIO()
                    with redirect_stdout(f):
                        print('Model:',mymodel,"\n")
                        print("Template:\n",messages,"\n")
                        print(response)

                st.session_state.messages.append({"role": "assistant", "content": response})    
                st.chat_message('assistant').write(response)
                
                if show_detail:
                    with st.expander('**:blue[Details]**', expanded=False):
                        s = f.getvalue()
                        st.write(s)
                
            else:

                # Start collecting vector information
                mytext = ""
                myhits = ""
                myembedding = get_embedding(prompt)
                if mydemo == 'Cyber':
                    mycategories = ["PUBLIC","IT"]
                else:    
                    mycategories = ["PUBLIC"]

                vs_results = collection.query(
                    query_embeddings=myembedding,
                    n_results= docs_retrieved,
                    where={"rbac":{"$in": mycategories}}
                )

                for x in range(len(vs_results['metadatas'][0])):
                    mytext = mytext + vs_results['documents'][0][x] + "\nLink: " + vs_results['metadatas'][0][x]['link'] + "\n\n------------\n"
                    myhits = myhits + "Source: "+vs_results['metadatas'][0][x]['source'] +" / Relevancy: "+str(f"{vs_results['distances'][0][x]:.4f}") +"\n\n"+ \
                        "Link: "+vs_results['metadatas'][0][x]['link'] +"\n\n"

                hprompt = prompt + f"""

<TEXT>
{mytext}
</TEXT>
"""    

                try:
                    messages = [
                    SystemMessage(
                        content="""You are a helpful AI bot that answers the human's question with the aid of the information in the <TEXT> section.
Provide the human with a thorogh and detailed answer. 
If you used information from the <TEXT> section, you MUST include any relevant web links in your answer.
If the <TEXT> does not help answer the human's question, you MUST first indicate that "The provided documents did not assist with answering the question." but try to answer if possible.
If the AI does not know the answer to a human's question, then say 'I don't know'."""
                    ),
                    HumanMessage(
                        content=hprompt
                    ),
                    ]   

                    if show_detail:
                        f = io.StringIO()
                        with redirect_stdout(f):
                            with st.spinner("Processing..."):
                                print('Model:',mymodel,"\n")
                                print('Items in Collection:',str(collection.count()),"\n")
                                print("Vector Hits:\n")
                                print(myhits)
                                print("Template:\n",messages,"\n")

                                with st.chat_message("assistant"):
                                    stream_handler = StreamHandler(st.empty())
                                    llm = ChatOpenAI(temperature=0.4, model=mymodel,  streaming=True, callbacks=[stream_handler])
                                    stream_handler = StreamHandler(st.empty())
                                    response = llm(messages)

                    else:
                        with st.spinner("Processing..."):
                            with st.chat_message("assistant"):
                                stream_handler = StreamHandler(st.empty())
                                llm = ChatOpenAI(temperature=0.4, model=mymodel,  streaming=True, callbacks=[stream_handler])
                                stream_handler = StreamHandler(st.empty())
                                response = llm(messages)

                    st.session_state.messages.append({"role": "assistant", "content": response.content})    

                    if show_detail:
                        with st.expander('**:blue[Details]**', expanded=False):
                            s = f.getvalue()
                            st.write(s)

                except ValueError as error:
                    raise error


        except ValueError as error:
            raise error

    else:

        # Start collecting vector information
        mytext = ""
        myhits = ""
        myembedding = get_embedding(prompt)
        if mydemo == 'Cyber':
            mycategories = ["PUBLIC","IT"]
        else:    
            mycategories = ["PUBLIC"]

        vs_results = collection.query(
            query_embeddings=myembedding,
            n_results= docs_retrieved,
            where={"rbac":{"$in": mycategories}}
        )

        for x in range(len(vs_results['metadatas'][0])):
            mytext = mytext + vs_results['documents'][0][x] + "\nLink: " + vs_results['metadatas'][0][x]['link'] + "\n\n------------\n"
            myhits = myhits + "Source: "+vs_results['metadatas'][0][x]['source'] +" / Relevancy: "+str(f"{vs_results['distances'][0][x]:.4f}") +"\n\n"+ \
                "Link: "+vs_results['metadatas'][0][x]['link'] +"\n\n"

        hprompt = prompt + f"""

<TEXT>
{mytext}
</TEXT>
"""    

        try:
            messages = [
            SystemMessage(
                content="""You are a helpful AI bot that answers the human's question with the aid of the information in the <TEXT> section.
Provide the human with a thorogh and detailed answer. Include any relevant web links in your answer.
If the AI does not know the answer to a question, it says 'I don't know'."""
            ),
            HumanMessage(
                content=hprompt
            ),
            ]   

            if show_detail:
                f = io.StringIO()
                with redirect_stdout(f):
                    with st.spinner("Processing..."):
                        print('Model:',mymodel,"\n")
                        print('Items in Collection:',str(collection.count()),"\n")
                        print("Vector Hits:\n")
                        print(myhits)
                        print("Template:\n",messages,"\n")

                        with st.chat_message("assistant"):
                            stream_handler = StreamHandler(st.empty())
                            llm = ChatOpenAI(temperature=0.4, model=mymodel,  streaming=True, callbacks=[stream_handler])
                            stream_handler = StreamHandler(st.empty())
                            response = llm(messages)
            else:
                with st.spinner("Processing..."):
                    with st.chat_message("assistant"):
                        stream_handler = StreamHandler(st.empty())
                        llm = ChatOpenAI(temperature=0.4, model=mymodel,  streaming=True, callbacks=[stream_handler])
                        stream_handler = StreamHandler(st.empty())
                        response = llm(messages)

            st.session_state.messages.append({"role": "assistant", "content": response.content})    

            if show_detail:
                with st.expander('**:blue[Details]**', expanded=False):
                    s = f.getvalue()
                    st.write(s)

        except ValueError as error:
            raise error


    tz.write("End: "+str(datetime.now()))
    tz.write("Duration: "+str(datetime.now() - start))
