## Overview
- This repository showcases very basic but important first steps on how to implement a Retrieval Augemented Generation (RAG) solution.
- In this example, we used text extracted from Wikipedia about Charles de Gaulle (https://en.wikipedia.org/wiki/Charles_de_Gaulle).
- The end goal is to the query the text for fairly correct responses.

## Approach
The approach here is:
 - ### Retrieval
  - We take contents in the document.txt file and then embed them.
  - We then embed the user query and then do a similary search between document and user query embedding to retrieve similar documents.
  - The similar documents are then used to create our context for the next stage.
 - ### Augmented
  - Here we construct a prompt (piece of string) with system instruction - instruction to guide the model how to generate the output. 
  - We then inject the user query and context which help to contrain the model to our local data. 
 - ### Generation
  - We get the contructed prompt and pass it to the model. The model then return fairly meaningful responses.
## Useful commands
 - pip freeze > requirements.txt - install dependencies