# LitSynth MCP Server

![alt text](/public/logo-container.png)

A Model Context Protocol (MCP) server for intelligent academic paper discovery and semantic search using ArXiv. This server provides tools for searching academic papers and performing semantic similarity analysis using state-of-the-art sentence transformers.

## Features

- **ArXiv Search**: Query ArXiv database with automatic URL encoding for complex search terms
- **Semantic Search**: Find papers most relevant to your research using AI-powered semantic similarity
- **Robust Error Handling**: Graceful handling of network issues and malformed data
- **Flexible Input**: Support for various query formats including spaces and special characters

## Tools Available

### 1. `greet(name: str)`

Simple greeting function for testing server connectivity.

**Parameters:**

- `name`: String - Name to greet

**Returns:** Greeting message

### 2. `search_query_arxiv(query: str, max_results: int = 5)`

Search ArXiv database for academic papers matching your query.

**Parameters:**

- `query`: String - Search terms (automatically URL encoded)
- `max_results`: Integer - Maximum number of results to return (default: 5)

**Returns:** Structured response with papers including:

- Title
- Authors
- Summary/Abstract
- ArXiv link
- Status message

**Example:**

```python
search_query_arxiv("multimodal agents", 3)
```

### 3. `search_semantic_arxiv(query: str, papers: list, top_k: int = 5)`

Perform semantic search on a list of papers to find the most relevant ones.

**Parameters:**

- `query`: String - Research query for semantic matching
- `papers`: List - Papers to search through (from `search_query_arxiv` or manual list)
- `top_k`: Integer - Number of most relevant papers to return (default: 5)

**Returns:** Ranked papers with similarity scores including:

- Title
- Summary
- Authors
- ArXiv link
- Similarity score (0-1)

**Example:**

```python
papers = search_query_arxiv("machine learning")
relevant = search_semantic_arxiv("deep reinforcement learning", papers, 3)
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone or download the project files**

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

1. **Run the MCP server:**

```bash
python my_server.py
```

## Dependencies

The project requires the following packages (see `requirements.txt`):

- `fastmcp>=0.1.0` - MCP framework
- `feedparser>=6.0.10` - RSS/Atom feed parsing for ArXiv API
- `requests>=2.31.0` - HTTP requests
- `sentence-transformers>=2.2.2` - Semantic search and embeddings
- `torch>=2.0.0` - PyTorch for neural networks
- `transformers>=4.21.0` - Hugging Face transformers
- `numpy>=1.21.0` - Numerical computing

## Project Structure

```bash
ai-research-assistant/
├── my_server.py           # Main MCP server implementation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Usage Examples

### Basic ArXiv Search

Search for papers on a specific topic:

```python
# Search for papers about transformers
results = search_query_arxiv("attention mechanisms transformers", 5)
```

### Semantic Paper Discovery

Find the most relevant papers from a search result:

```python
# First, get papers on a broad topic
papers = search_query_arxiv("artificial intelligence", 20)

# Then find the most relevant ones for your specific research
relevant_papers = search_semantic_arxiv("graph neural networks", papers, 5)
```

### Handling Complex Queries

The server automatically handles special characters and spaces:

```python
# These work automatically without manual encoding
search_query_arxiv("machine learning & deep learning: survey")
search_query_arxiv("reinforcement learning (RL) applications")
```

## Technical Details

### Semantic Search Model

The server uses the `sentence-transformers/all-MiniLM-L6-v2` model for semantic embeddings. This model:

- Provides 384-dimensional sentence embeddings
- Balances speed and accuracy
- Works well for academic text similarity

### Error Handling

The server includes comprehensive error handling:

- **URL Encoding**: Automatic handling of spaces and special characters

- **Network Errors**: Graceful degradation when ArXiv is unavailable
- **Data Validation**: Safe handling of missing or malformed paper data
- **Empty Results**: Informative messages when no papers are found

### Response Format

All functions return structured responses:

```json
{
  "message": "Status or info message",
  "results": [
    {
      "title": "Paper Title",
      "author": ["Author 1", "Author 2"],
      "summary": "Abstract text...",
      "link": "https://arxiv.org/abs/...",
      "similarity_score": 0.85  // Only in semantic search
    }
  ]
}
```

## Troubleshooting

### Common Issues

**"URL can't contain control characters" error:**

- This is fixed in the current version with automatic URL encoding
- Make sure you're using the latest version of the server

**"No papers found" result:**

- Check your query spelling
- Try broader search terms
- Verify ArXiv service availability

**Slow semantic search:**

- First run downloads the transformer model (~90MB)
- Subsequent runs are much faster
- Consider reducing `top_k` for faster results

**Memory issues:**

- The sentence transformer model requires ~500MB RAM
- Reduce batch sizes if experiencing memory problems

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the AI Research Assistant.

## License

This project is open source. Please check individual dependency licenses for commercial use.

## Acknowledgments

- **ArXiv** for providing free access to academic papers
- **Sentence Transformers** for semantic search capabilities
- **FastMCP** for the MCP server framework