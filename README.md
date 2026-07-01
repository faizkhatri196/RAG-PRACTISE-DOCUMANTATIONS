# RAG-PRACTISE-DOCUMANTATIONS

This repository contains Jupyter notebooks and datasets for practicing Retrieval Augmented Generation (RAG) workflows using Python and LangChain.

## Project Structure

*   **`document/`**: Contains Jupyter notebooks for loading documents, text splitting, and embedding generation.
    *   [01.ipynb](file:///c:/Users/Infinity/OneDrive/Desktop/MCP/document/01.ipynb): Demonstrates document loaders (`TextLoader`, `DirectoryLoader`, `PyPDFLoader`, and `PyMuPDFLoader`).
    *   [02.ipynb](file:///c:/Users/Infinity/OneDrive/Desktop/MCP/document/02.ipynb): Demonstrates text splitting using `RecursiveCharacterTextSplitter`.
*   **`data/`**: Source text files and PDF manuals/handbooks used as test data for loading and indexing.

## Dependencies

The project uses the modern LangChain package structure:
*   `langchain`: Core framework.
*   `langchain-core`: Core abstractions.
*   `langchain-community`: Document loaders and community integrations.
*   `langchain-text-splitters`: Splitting utilities (like `RecursiveCharacterTextSplitter`).
*   `pymupdf` & `pypdf`: High-performance PDF parsers.

## Installation & Setup

Ensure you have the `uv` package manager installed, then sync the dependencies:

```bash
# Install dependencies and sync virtual environment
uv sync
```
