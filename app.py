from openai import OpenAI
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain

generate_content = """
Your job is to help provide a human writer with key bullet points that capture personality types in the {MODEL} personality framework for all of the major characters from a given TV show/movie/book/etc based on your knowledge of each type.

AGAIN, THE SELECTED FRAMEWORK IS: {MODEL}

(IF AND ONLY IF the selected framework is Myers-Briggs and you plan to label or name any of the types, please use the following names only. Note you don't have to do this, but if you do use the names in addition to the types, use these names: INFP: The Healer, INTJ: The Mastermind, INFJ: The Counselor, INTP: The Architect, ENFP: The Champion, ENTJ: The Commander, ENTP: The Visionary, ENFJ: The Teacher, ISFJ: The Protector, ISFP: The Composer, ISTJ: The Inspector, ISTP: The Craftsperson, ESFJ: The Provider, ESFP: The Performer, ESTJ: The Supervisor, ESTP: The Dynamo)

(IF AND ONLY IF the selected framework is Emotional Intelligence, please refer to this breakdown of how to score someone on each of the five scales: 

### **1. Self-Awareness**
- **Higher Scores:** Insightful, reflective, emotionally attuned. Recognizes and understands emotions in real-time and can articulate the reasons behind them. Rarely surprised by emotional reactions.
- **Lower Scores:** Reactive, less introspective, struggles to identify feelings or their origins. Frequently blindsided by emotional responses.
- **Middle Scores:** Generally aware of emotions but may need time to reflect on complex or subtle feelings.

### **2. Other Awareness**
- **Higher Scores:** Perceptive, intuitive, socially observant. Easily picks up on nonverbal cues like tone, facial expressions, and body language. Anticipates others’ emotional states accurately.
- **Lower Scores:** Misses or misreads social cues, leading to misunderstandings. May seem detached or self-focused.
- **Middle Scores:** Can interpret most emotions but might miss subtleties or nuances in more complex situations.

### **3. Emotional Control**
- **Higher Scores:** Calm, composed, and resilient under stress. Manages emotions effectively, redirecting them toward productive outcomes.
- **Lower Scores:** Prone to mood swings, overwhelmed by emotions, and slow to recover from setbacks. Reacts impulsively under pressure.
- **Middle Scores:** Generally steady but may struggle to maintain composure in particularly intense or unexpected situations.

### **4. Empathy**
- **Higher Scores:** Compassionate, caring, and deeply attuned to others’ emotions. Feels motivated to help and connect on a meaningful level.
- **Lower Scores:** Struggles to relate to others’ feelings. Tends to focus on logic or fairness over emotional understanding.
- **Middle Scores:** Relates to others’ emotions in many cases but may find it harder to connect deeply in certain situations.

### **5. Wellbeing**
- **Higher Scores:** Optimistic, confident, and emotionally stable. Finds joy and satisfaction in daily life and maintains strong social connections.
- **Lower Scores:** Prone to stress, anxiety, or pessimism. Finds it challenging to feel fulfilled or maintain emotional and social health.
- **Middle Scores:** Generally positive and resilient but experiences ups and downs, especially in high-stress situations or during setbacks.)


Here is the specific character universe for which you should generate the appropriate personality types: {X}

Your job is to output ALL of the key characters related to the above universe and their type assignment as headers, and the rich justification for and evidence related to assigning that character to that type as bullets below that header.
SPECIAL NOTE: if the SELECTED FRAMEWORK is Big Five OR Emotional Intelligence, you cannot do typing in the same way as the other models. Therefore, still output the major characters and attempt to characterize them across all five traits (very low, low, medium, high, very high), using rich evidence for each one (for EQ, refer to breakdown above).

It should be as rich information as possible/appropriate for each character, using specific details or actions from the story to justify your type/trait assignment. It is okay to have duplicate types (give the best and most honest type assignment possible), but be mindful at the same time not to output, eg, 10 characters of the same type (this wouldn't make for a great article!). Strike the balance, but prioritize accurately nailing the types.

Again, a human is going to take your outputs as an outline/reference for writing a polished blog piece, so you don't need to output polished text yourself, just make sure the core ideas and key raw content is there. It does not have to be pretty!

Formatting requirements: make sure you immediately output the specified content, no preface or conclusion, and make sure it is in Markdown for easy formatting.

YOUR OUTPUTS:
"""

st.title('Get character personalities from any movie/TV show')

# Dropdown for selecting the personality model with no option selected by default
model_type = st.selectbox(
    'Choose a Personality Model',
    ('Select a model', 'DISC', 'Enneagram', 'Myers-Briggs', 'Big Five', 'Emotional Intelligence'),
    index=0,
    key='model_type'
)

# Text area for defining the task
task_description = st.text_area('Enter TV show/movie here, with any additional context', placeholder='Gilmore Girls, top 10 characters | all major Harry Potter characters | protagonists from major Disney movies | etc.',
                                key='task_description',max_chars=400)

# Submit button
if st.button('Submit'):
    if model_type == 'Select a model':
        st.error('Please select a personality model.')
    elif not task_description.strip():
        st.error('Please define the task.')
    else:
        with st.spinner('Generating, standby...'):
            chat_model = ChatOpenAI(openai_api_key=st.secrets['API_KEY'], model_name='gpt-4o-2024-08-06', temperature=0.4)
            chat_chain = LLMChain(prompt=PromptTemplate.from_template(generate_content), llm=chat_model)
            generated_output = chat_chain.run(MODEL=model_type, X=task_description)
            
            # Display the LLM output using markdown
            st.write(generated_output)
            
            # Use Streamlit's download button to download the output
            st.download_button(
                label="Download Output",
                data=generated_output,
                file_name="personality_characters.txt",
                mime="text/plain"
            )
