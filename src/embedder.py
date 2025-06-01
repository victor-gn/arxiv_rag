import os
from typing import List
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()


def get_openai_embeddings(texts: List[str], batch_size: int = 200) -> List[List[float]]:
    """Create embeddings for a list of texts using the OpenAI model.
    Processes in batches to respect API token limits.

    Args:
        texts (List[str]): List of texts to create embeddings for.
        batch_size (int, optional): Size of each processing batch. Defaults to 200.

    Returns:
        List[List[float]]: List of embedding vectors.

    Note:
        If a batch fails, the function will automatically retry with a smaller batch size.
        The minimum batch size is 10 before raising an exception.
    """
    if not texts:
        return []

    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    all_embeddings = []

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            # Create embeddings for current batch
            batch_embeddings = embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {str(e)}")
            # If it fails, try processing with smaller batch
            if batch_size > 10:
                print(f"Trying with smaller batch ({batch_size // 2})...")
                return get_openai_embeddings(texts, batch_size=batch_size // 2)
            else:
                raise e

    return all_embeddings
