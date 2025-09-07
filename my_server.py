from fastmcp import FastMCP
import feedparser
import requests
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from urllib.parse import quote_plus

mcp = FastMCP("AI Research Assistant - v0.0.1")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# greet the user cause he need it why not
@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

# search arxiv by query (only fetch the first 5) | max_results = 5 is by default
@mcp.tool
def search_query_arxiv(query: str, max_results: int = 5):
    # this is where the encoding happens  !!!!
    encoded_query = quote_plus(query)
    
    # fixed: Use the actual query parameter  Fixed encoding of the query
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    
    try:
        feed = feedparser.parse(url)
        results = []
        
        # fixed: Handle case where feedparser returns no entries
        if hasattr(feed, 'entries') and feed.entries:
            for entry in feed.entries:
            # au cas ou il y a pas de putain d'auteurs (c'est rare anyway)
                authors = []
                if hasattr(entry, 'authors'):
                    authors = [a.name for a in entry.authors]
                elif hasattr(entry, 'author'):
                    authors = [entry.author]
                
                results.append({
                    "title": getattr(entry, 'title', 'No title'),
                    "author": authors,
                    "summary": getattr(entry, 'summary', 'No summary'),
                    "link": getattr(entry, 'link', 'No link')
                })
        else:
            return {"message": "No papers found for this query", "results": []}
        
        return {"message": f"Found {len(results)} papers", "results": results}
    
    except Exception as e:
        return {"error": f"Failed to search ArXiv: {str(e)}", "results": []}

# function for semantic search on arxiv (only fetch the first top 5 same as the search_query_arxiv)
@mcp.tool
def search_semantic_arxiv(query: str, papers: list, top_k: int = 5):
    """
    Perform a semantic search on arxiv over the given list of papers (dicts with 'summary').
    This will return the top_k most 'relevant' papers based on the query research :-)
    """
    try:
        if isinstance(papers, dict) and 'results' in papers:
            papers = papers['results']
        
        if not papers:
            return {"message": "No papers provided for semantic search", "results": []}

        summaries = []
        for p in papers:
            summary = p.get('summary', '')
            if not summary or summary == 'No summary':
                summary = p.get('title', '')
            summaries.append(summary)
        
        if not any(summaries):
            return {"message": "No text content found in papers for semantic search", "results": []}
        
        query_embedding = model.encode(query, convert_to_tensor=True)
        doc_embeddings = model.encode(summaries, convert_to_tensor=True)
        
        scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        
        actual_top_k = min(top_k, len(papers))
        top_results = scores.topk(actual_top_k)
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            paper = papers[int(idx)]
            results.append({
                "title": paper.get('title', 'No title'),
                "summary": paper.get('summary', 'No summary'),
                "link": paper.get('link', 'No link'),
                "authors": paper.get('author', []),
                "similarity_score": float(score) #  fixed: Added the score of accuracy of semantic search
            })
        
        return {"message": f"Found {len(results)} relevant papers", "results": results}
    
    except Exception as e:
        return {"error": f"Semantic search failed: {str(e)}", "results": []}


if __name__ == "__main__":
    mcp.run()