from RAG.rag_utils import rag_search

function_map = {
    "rag_search": rag_search
}

{
    "name": "rag_search",
    "description": "Search for bio knowledge from knowledge base",
    "parameters": {
        "type": "object",
        "properties": {
            "collection_name": {
                "type": "string",
                "description": "Name of the Qdrant collection to search"
            },
            "query": {
                "type": "string",
                "description": "Search query text to generate embedding"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top results to return",
                "default": 3
            }
        },
        "required": ["collection_name", "query"]
    },
    "returns": {
        "type": "array",
        "description": "List of search results with text, similarity score, and document ID",
        "items": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Retrieved text content"
                },
                "similarity": {
                    "type": "number",
                    "description": "Similarity score of the result"
                },
                "doc_id": {
                    "type": "string",
                    "description": "Unique identifier of the document"
                }
            }
        }
    }
}
