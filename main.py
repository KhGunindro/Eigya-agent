from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="gemma3")

template = """
    You are an expert in answering questions about food recipes
    
    Here are some relevant recipes: {recipes}
    
    Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# invoke the chain to run our llm
chain = prompt | model

while True:
    print("\n\n---------------------------------------------------------------")
    question = input("Ask your question about recipes: (q to quit)\n\n")
    print("\n\n---------------------------------------------------------------")
    if question == "q":
        break
    recipes = retriever.invoke(question)
    # invoke the chain to run our llm
    result = chain.invoke({"recipes": recipes, "question": question})
    print(result)

