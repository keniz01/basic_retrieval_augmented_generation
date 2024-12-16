import textwrap
from llama_cpp import LLAMA_POOLING_TYPE_NONE, Llama
import numpy as np
 
chat_model = Llama(
   model_path="./models/Phi-3-mini-128k-instruct.Q6_K.gguf",
   n_ctx=2048,
   verbose=False,
   n_gpu_layers=0,
   logits_all=True,
   chat_format="chatml"    
)
 
embed_model=Llama(
  model_path='./models/bge-small-en-v1.5-q4_k_m.gguf',
  embedding=True,
  verbose=False,
  pooling_type=LLAMA_POOLING_TYPE_NONE
)
 
file = open('document.txt', 'r')
data = file.read()
documents=data.split(".")

doc_embeddings = [embed_model.embed(document)[1] for _, document in enumerate(documents)]    
 
def generate_context(user_query: str) -> str:

  query_embeddings=embed_model.embed(user_query)[1]
  similarities=np.dot(doc_embeddings, query_embeddings)
  
  top_3_idx=np.argsort(similarities, axis=0)[-3:][::1].tolist()
  most_similar_documents=[documents[idx] for idx in top_3_idx]
  
  context = ""
  for _, document in enumerate(most_similar_documents):
      wrapped_text=textwrap.fill(document,width=100)
      context += wrapped_text + "\n"
  
  return context

def generate_user_prompt(user_query: str) -> str:
  '''
  Generates the LLM prompt using the user query query and context.

  :param str user_query: The user query.
  :return: the response string
  :rtype: str
  :raises Error: when an error is encountered
  '''
  def construct_prompt(user_query: str) -> str:   

    context=generate_context(user_query)
    llm_prompt = f"""
    <|system|>
    Use the following context to answer the question at the end.
    If you dont know the answer just say that you dont know, dont try to make up the answer or give advice.
    <|end|>
    <|user|>
    {context}
    {user_query}
    <|end|>
    <|assistant|>
    """

    return llm_prompt
  
  return construct_prompt(user_query)

def generate_query_response(user_prompt: str):
  '''
  Generates the LLM response to the user query.
   
  :param str user_prompt: The user query prompt.
  :return: the response string
  :rtype: str
  :raises Error: when an error is encountered
  '''
  stream = chat_model(
    prompt=user_prompt,
    max_tokens=512,
    temperature=0,
    top_p=0.1,
    stream=True
  )

  for output in stream:
      token = output["choices"][0]["text"]
      print(token, end='')
  
  print()

if __name__ == "__main__":

  print("\nAsk a question or enter 'quit' to exit: ", end='')
  user_query = input()
  
  while user_query.capitalize() != 'quit'.capitalize():
    user_prompt=generate_user_prompt(user_query)
    generate_query_response(user_prompt)    
    print("Question: ", end='')
    user_query = input()

  print()