# NanoIdx: RAG from zero to hero

## 1. What is RAG?

RAG, which stands for `Retrieval-Augmented Generation`, is a powerful technique designed to improve the performance of Large Language Models (LLMs). The RAG combines two key abilities: *retrieving factual information* and *generating human-like responses*.

In a typical setup, when an LLM like ChatGPT answers a question, it relies entirely on the knowledge it has learned during training. However, this knowledge is static and may become outdated or incomplete over time. As a result, LLMs can sometimes produce incorrect or made-up responses, a phenomenon commonly referred to as hallucination. This can lead to misleading or unreliable information.

RAG addresses this problem by giving the model access to an external knowledge source before generating a response, such as a document database, a Wikipedia dump, or even real-time search results. Here's how it works in simple terms:

- Retrieval Phase: The system first searches for and retrieves relevant documents or text passages related to the user's query from the external knowledge base.

- Generation Phase: The language model then uses both the retrieved information and its own understanding of language to generate a more accurate, informative, and context-aware response.

By grounding the generation process in real, verifiable data, RAG significantly reduces hallucinations and improves the accuracy, reliability, and trustworthiness of the responses.

Because of these advantages, RAG has become a central focus in modern AI research and development. Many researchers and companies are working on creating new algorithms, tools, and frameworks that implement RAG in practical applications, ranging from question-answering systems to enterprise search tools and digital assistants in specialized domains like healthcare, law, or customer support.

In summary, RAG is a key innovation that helps bridge the gap between general language understanding and access to up-to-date, factual knowledge, making LLMs much more useful and trustworthy in real-world use cases.

## 2. The outline of this tutorial

The aim of this tutorial is to introduce the fundamental concepts behind Retrieval-Augmented Generation (RAG) and provide a from-scratch implementation that anyone can follow, regardless of their experience level. By walking through the core ideas and mechanisms step-by-step, you'll gain a deeper understanding of how RAG works under the hood. This foundation will empower you to build your own RAG system and adapt it to suit your specific use cases.

As the title "From Hero to Zero" suggests, we will be starting from the ground up and implementing the entire RAG workflow manually. That means we won’t rely on high-level frameworks like [LangChain](https://www.langchain.com/) or [LlamaIndex](https://www.llamaindex.ai/), which often abstract away many of the inner workings. This approach helps you learn the actual components and logic involved in building a RAG pipeline. Because of this, you won’t need to install additional packages or set up a specialized vector database. Everything will be built using basic tools and libraries. 

It’s worth noting that this project can be seen as a minimal working implementation of the core ideas behind LlamaIndex, as we take inspiration from its design and functionalities. While we are not using the LlamaIndex framework directly, many of the concepts and coding patterns in this tutorial are informed by how LlamaIndex structures a RAG pipeline. By rebuilding these components from the ground up, and occasionally comparing them with how a full-featured LlamaIndex handles them, you’ll gain a much clearer understanding of how these systems actually work behind the scenes. This knowledge will also give you the flexibility to customize and optimize your own RAG solution without being locked into a specific tool or library.

For the LLM itself, you are free to use either cloud-based models provided by major companies like OpenAI (ChatGPT/GPT-4), Anthropic (Claude), DeepSeek, or Google (Gemini) via remote APIs calling, or run your own LLMs locally using tools such as Ollama, LM Studio, or vLLM. This flexibility allows you to adapt the tutorial to your environment, no matter whether you're working on a personal project with limited resources or experimenting in a more advanced setting with private infrastructure.

There are two main folders that contains files for the NanoIdx

- `nanoidx`: The source code 
- `tutorial`: The tutorial written in Jupyter notebook to show the idea and algorithm in an interactive way.


