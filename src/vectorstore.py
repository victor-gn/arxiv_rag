from qdrant_client import QdrantClient, models
from typing import List, Dict, Any
import time
import uuid
import os
from dotenv import load_dotenv

load_dotenv()


class QdrantVectorStore:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60,
        )

    def upsert_embeddings_qdrant(
        self,
        embeddings: List[list],
        metadata: List[Dict[str, Any]],
        collection_name: str = "arxiv_chunks",
        vector_size: int = 1536,  # text-embedding-3-small
        distance=models.Distance.COSINE,
        batch_size: int = 100,
    ):
        """Insert embeddings and metadata into a Qdrant collection in batches.

        Args:
            embeddings (List[list]): List of embedding vectors.
            metadata (List[Dict[str, Any]]): List of metadata dictionaries (payloads).
            collection_name (str, optional): Name of the Qdrant collection. Defaults to "arxiv_chunks".
            host (str, optional): Qdrant host. Defaults to "localhost".
            port (int, optional): Qdrant port. Defaults to 6333.
            vector_size (int, optional): Dimension of the embeddings. Defaults to 1536.
            distance (models.Distance, optional): Distance metric. Defaults to COSINE.
            batch_size (int, optional): Size of each batch for insertion. Defaults to 100.
            timeout (int, optional): Timeout in seconds for each operation. Defaults to 60.

        Note:
            The function will automatically create the collection if it doesn't exist.
            It uses UUID for generating unique IDs for each point.
            Includes retry mechanism for failed batch insertions.
        """

        # Create collection if it doesn't exist
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name not in collection_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
            )

        # Insert in batches
        total = len(embeddings)
        for i in range(0, total, batch_size):
            batch_embeddings = embeddings[i : i + batch_size]
            batch_metadatas = metadata[i : i + batch_size]

            # Generate unique IDs using UUID for each point
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),  # Generate a unique UUID for each point
                    vector=vec,
                    payload=meta,
                )
                for vec, meta in zip(batch_embeddings, batch_metadatas)
            ]

            # Try to insert batch with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=points,
                    )
                    print(f"Batch {i // batch_size + 1} inserted successfully!")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(
                        f"Error in batch {i // batch_size + 1}, trying again... ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(2)  # Wait 2 seconds before trying again

    def search_similar_chunks(
        self,
        query_embedding: List[float],
        collection_name: str = "arxiv_chunks",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for chunks similar to the query embedding in Qdrant.

        Args:
            query_embedding (List[float]): Query embedding vector.
            collection_name (str, optional): Name of the Qdrant collection. Defaults to "arxiv_chunks".
            limit (int, optional): Maximum number of results to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: List of chunks with their metadata and similarity scores.
                Each chunk contains:
                - text: The chunk's text content
                - arxiv_id: The arXiv ID of the source document
                - chunk_idx: The index of the chunk in the document
                - score: The similarity score
        """

        try:
            # Perform search with score filter
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
            )

            # Format results
            results = []
            for hit in search_result:
                results.append(
                    {
                        "text": hit.payload.get("text", ""),
                        "arxiv_id": hit.payload.get("arxiv_id", ""),
                        "chunk_idx": hit.payload.get("chunk_idx", 0),
                        "score": hit.score,
                    }
                )

            return results

        except Exception as e:
            print(f"Error searching for similar chunks: {str(e)}")
            return []

    def get_collection_stats(
        self, collection_name: str = "arxiv_chunks"
    ) -> Dict[str, Any]:
        """Get statistics for a Qdrant collection.

        Args:
            collection_name (str, optional): Name of the collection. Defaults to "arxiv_chunks".

        Returns:
            Dict[str, Any]: Dictionary containing collection statistics:
                - name: Collection name
                - vectors_count: Number of vectors in the collection
                - status: Collection status
                - config: Collection configuration including vector size and distance metric
        """
        try:
            collection_info = self.client.get_collection(collection_name)
            collection_stats = self.client.get_collection(collection_name).points_count

            return {
                "name": collection_name,
                "vectors_count": collection_stats,
                "status": collection_info.status,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance,
                },
            }
        except Exception as e:
            print(f"Error getting collection statistics: {str(e)}")
            return {}

    def delete_collection(self, collection_name: str = "arxiv_chunks") -> bool:
        """Delete a Qdrant collection.

        Args:
            collection_name (str, optional): Name of the collection to delete. Defaults to "arxiv_chunks".

        Returns:
            bool: True if the collection was successfully deleted, False otherwise.
        """
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
            return False
