"""
!pip install datasets evaluate transformers[sentencepiece]
!pip install accelerate
!pip install bitsandbytes
!pip install loralib
!pip install peft
!pip install trl

"""
import streamlit as st

st.set_page_config(
    page_title="text simplification app",
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,TrainingArguments,pipeline

"""# Text Simplification using an LLM 
"""

def prompt_instruction_format(sample):
  return f"""### Instruction:
    Use the Task below and the Input given to write the Response:

    ### Task:
    Simplify the Input

    ### Input:
    {sample['original']}

    ### Response:
    {sample['simplifications']}
    """
# setup the pipeline
from transformers import pipeline

model_name = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

#setting padding instructions for tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
temperature = 0.7
pipeline_untrained = pipeline("text2text-generation", model="google/flan-t5-small", tokenizer=tokenizer, max_new_tokens=512, temperature = temperature)

"""




"""
# streamlit user input (text)
text = st.text_input("Insert a text to get simplified by FLAN-T5-Small:")


"""Output:"""


if text:
    simplified_text = pipeline_untrained(text)
    #html_results = ""
    html_results = simplified_text
    st.markdown(html_results, unsafe_allow_html=True)

"""
"""
"""
"""
temperature = st.slider("temperature parameter", min_value=0.0, max_value=1.0, value=0.7, step=0.001, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""

    If you have no idea what to type into the text input field above, you can copy-paste this text: 


"""
"""
It will then dislodge itself and sink back to the river bed in order to digest its food and wait for its next meal.
"""
"""
from ASSET dataset: Abstractive Sentence Simplification Evaluation and Tuning, https://aclanthology.org/2020.acl-main.424
"""
"""
"""

"""
"""
