from pathlib import Path
from typing import List, Any
import json
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

def load_all_documents(data_dir: str) -> List[Document]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    if not data_path.exists():
        print(f"[WARNING] Data directory {data_path} does not exist.")
        return documents

    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF pages from {pdf_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file.name}: {e}")

    # Text files
    text_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(text_files)} TXT files")
    for txt_file in text_files:
        print(f"[DEBUG] Loading TXT: {txt_file}")
        try:
            loader = TextLoader(str(txt_file), encoding='utf-8')
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file.name}: {e}")

    # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file.name}: {e}")

    # Excel files (xlsx)
    excel_files = list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(excel_files)} Excel files")
    for xlsx_file in excel_files:
        print(f"[DEBUG] Loading Excel: {xlsx_file}")
        try:
            try:
                from langchain_community.document_loaders.excel import UnstructuredExcelLoader
                loader = UnstructuredExcelLoader(str(xlsx_file))
                loaded = loader.load()
            except Exception:
                # Fallback to pandas if unstructured excel loader is not working
                import pandas as pd
                df = pd.read_excel(xlsx_file)
                text = df.to_string()
                loaded = [Document(page_content=text, metadata={"source": str(xlsx_file)})]
            print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {xlsx_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_file.name}: {e}")

    # Word files (docx)
    word_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(word_files)} Word files")
    for docx_file in word_files:
        print(f"[DEBUG] Loading Word: {docx_file}")
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Word docs from {docx_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file.name}: {e}")

    # JSON files
    json_files = list(data_path.glob('**/*.json'))
    print(f"[DEBUG] Found {len(json_files)} JSON files")
    for json_file in json_files:
        print(f"[DEBUG] Loading JSON: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = json.dumps(data, indent=2)
            loaded = [Document(page_content=text, metadata={"source": str(json_file)})]
            print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file.name}: {e}")

    print(f"[DEBUG] Total documents loaded: {len(documents)}")
    return documents
