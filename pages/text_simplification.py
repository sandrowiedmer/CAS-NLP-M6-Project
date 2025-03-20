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
    page_title="text simplification",
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,TrainingArguments,pipeline

"""# Demonstration of Text Simplification using an LLM (FLAN-T5-Small)
"""

"""
Without using the simplification task fine-tuned model (module 3 project), for Streamlit test purpose only
"""
def prompt_instruction_format(sample):
  return f"""### Instruction:
    Use the Task below and the Input given to write the Response:

    ### Task:
    Simplify the Input

    """
# setup the pipeline
from transformers import pipeline
model_name = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

#setting padding instructions for tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
pipeline_untrained = pipeline("text2text-generation", model="google/flan-t5-small", tokenizer=tokenizer, max_new_tokens=512)

"""




"""
# streamlit user input
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
"""
"""
"""
"""
"""
"""
"""
"""
"""

    If you have no idea what to type into the text input field above, you can copy-paste this sentence about the T5 model: 


    "What does Flan mean in Flan T5?
    Flan stands for Fine-tuned Language Net (FLAN). And T5 is a text-to-text transfer transformer (get it, 5 Ts). The FlanT5 model is an encoder-decoder Large Language Model (LLM) from Google that was released in October 2021 and has been specially fine-tuned using instruction fine-tuning."

"""