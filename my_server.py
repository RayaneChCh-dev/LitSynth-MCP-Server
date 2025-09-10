from fastmcp import FastMCP
import feedparser
import requests
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, list_repo_files, hf_hub_download
from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names
import pandas as pd
from sentence_transformers import util
import time
from urllib.parse import quote_plus

mcp = FastMCP("AI Research Assistant - v0.0.1")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

api = HfApi()

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

@mcp.tool
def search_hf_datasets(query: str, limit: int = 5):
    retries = 3
    
    for attempt in range(retries):
        try:
            datasets = api.list_datasets(search=query, limit=limit)
            results = []
            
            for ds in datasets:
                # Handle potential None values more safely
                description = None
                if hasattr(ds, 'cardData') and ds.cardData:
                    description = ds.cardData.get("description")
                
                results.append({
                    "id": ds.id,
                    "description": description,
                    "url": f"https://huggingface.co/datasets/{ds.id}"
                })
            
            return {"message": f"Found {len(results)} datasets", "results": results}
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for specific retryable errors
            if ("MeasurementsServiceUnavailable" in error_msg or 
                "503" in error_msg or 
                "timeout" in error_msg.lower()):
                
                if attempt < retries - 1:  # Not the last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return {"error": f"Service unavailable after {retries} retries: {error_msg}", "results": []}
            else:
                # Non-retryable error, return immediately
                return {"error": f"Search failed: {error_msg}", "results": []}
    
    # This should never be reached, but just in case
    return {"error": "Unexpected error in retry logic", "results": []}

@mcp.tool
def get_dataset_details(dataset_id: str):
    retries = 3
    
    for attempt in range(retries):
        try:
            # Get basic dataset info
            dataset_info = api.dataset_info(dataset_id)
            
            # Initialize result structure
            details = {
                "dataset_id": dataset_id,
                "url": f"https://huggingface.co/datasets/{dataset_id}",
                "basic_info": {},
                "structure": {},
                "files": [],
                "usage_info": {},
                "research_applications": []
            }
            
            # Basic information
            details["basic_info"] = {
                "description": getattr(dataset_info, 'description', 'No description available'),
                "homepage": getattr(dataset_info, 'homepage', None),
                "license": getattr(dataset_info, 'license', 'Not specified'),
                "citation": getattr(dataset_info, 'citation', None),
                "tags": getattr(dataset_info, 'tags', []),
                "task_categories": getattr(dataset_info, 'task_categories', []),
                "size_categories": getattr(dataset_info, 'size_categories', []),
                "language": getattr(dataset_info, 'language', [])
            }
            
            # Dataset structure and configuration
            if hasattr(dataset_info, 'config_names') and dataset_info.config_names:
                details["structure"]["configurations"] = dataset_info.config_names
            else:
                details["structure"]["configurations"] = ["default"]
            
            # File information
            try:
                files = api.list_repo_files(dataset_id, repo_type="dataset")
                # Filter for data files and limit to most relevant
                data_files = [f for f in files if f.endswith(('.json', '.jsonl', '.csv', '.parquet', '.txt', '.tsv'))]
                details["files"] = data_files[:10]  # Limit to first 10 files
            except Exception:
                details["files"] = ["File information unavailable"]
            
            # Usage information
            details["usage_info"] = {
                "loading_command": f"from datasets import load_dataset\ndataset = load_dataset('{dataset_id}')",
                "streaming_command": f"dataset = load_dataset('{dataset_id}', streaming=True)",
                "python_usage": f"# Load the dataset\nfrom datasets import load_dataset\n\n# Load full dataset\ndataset = load_dataset('{dataset_id}')\n\n# Or stream for large datasets\ndataset = load_dataset('{dataset_id}', streaming=True)\n\n# Access train split\ntrain_data = dataset['train']\nprint(f'Train samples: {{len(train_data)}}')"
            }
            
            # Try to get dataset statistics if available
            try:
                # This tries to load a small sample to understand structure
                from datasets import load_dataset
                sample_dataset = load_dataset(dataset_id, split="train[:5]", trust_remote_code=True)
                
                if len(sample_dataset) > 0:
                    first_example = sample_dataset[0]
                    details["structure"]["sample_fields"] = list(first_example.keys())
                    details["structure"]["sample_data"] = {k: str(v)[:100] + "..." if len(str(v)) > 100 else str(v) 
                                                          for k, v in first_example.items()}
                    details["structure"]["total_samples"] = len(sample_dataset.dataset) if hasattr(sample_dataset, 'dataset') else "Unknown"
                
            except Exception as e:
                details["structure"]["sample_fields"] = ["Could not load sample data"]
                details["structure"]["error"] = str(e)
            
            # Generate research applications based on dataset metadata
            applications = []
            tags = details["basic_info"]["tags"]
            task_categories = details["basic_info"]["task_categories"]
            
            # Infer research applications from metadata
            if "multimodal" in str(tags).lower() or "multimodal" in dataset_id.lower():
                applications.append("Multimodal learning research")
                applications.append("Vision-language model training")
                applications.append("Cross-modal retrieval systems")
            
            if "agent" in str(tags).lower() or "agent" in dataset_id.lower():
                applications.append("Agent behavior analysis")
                applications.append("Reinforcement learning environments")
                applications.append("Multi-agent system research")
            
            if any(task in str(task_categories).lower() for task in ["image", "vision", "visual"]):
                applications.append("Computer vision research")
                applications.append("Image classification and detection")
            
            if any(task in str(task_categories).lower() for task in ["text", "language", "nlp"]):
                applications.append("Natural language processing")
                applications.append("Language model fine-tuning")
            
            if "conversation" in str(tags).lower() or "dialog" in str(tags).lower():
                applications.append("Conversational AI development")
                applications.append("Dialog system training")
            
            # Default applications if none detected
            if not applications:
                applications = [
                    "Machine learning model training",
                    "Benchmark evaluation",
                    "Academic research"
                ]
            
            details["research_applications"] = applications
            
            return {
                "message": f"Successfully retrieved details for {dataset_id}",
                "dataset_details": details
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific errors
            if "does not exist" in error_msg.lower() or "not found" in error_msg.lower():
                return {
                    "error": f"Dataset '{dataset_id}' not found. Please check the dataset ID.",
                    "suggestion": "Make sure you're using the exact dataset ID from the search results (e.g., 'microsoft/COCO', not just 'COCO')"
                }
            
            # Retry for temporary issues
            if ("503" in error_msg or "timeout" in error_msg.lower() or 
                "unavailable" in error_msg.lower()):
                
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return {
                        "error": f"Service temporarily unavailable after {retries} retries: {error_msg}",
                        "suggestion": "Please try again in a few minutes."
                    }
            else:
                # Non-retryable error
                return {
                    "error": f"Failed to get dataset details: {error_msg}",
                    "dataset_id": dataset_id
                }
    
    return {"error": "Unexpected error in retry logic"}

@mcp.tool
def explore_dataset_files(dataset_id: str):
    retries = 3
    
    for attempt in range(retries):
        try:
            result = {
                "dataset_id": dataset_id,
                "url": f"https://huggingface.co/datasets/{dataset_id}",
                "structure": {},
                "available_files": [],
                "sample_preview": {},
                "summary": ""
            }
            
            # Get dataset configurations
            try:
                configs = get_dataset_config_names(dataset_id)
                result["structure"]["configurations"] = configs
            except Exception:
                configs = ["default"]
                result["structure"]["configurations"] = ["default"]
            
            # Explore the first/default configuration
            main_config = configs[0] if configs else None
            
            # Get available splits for the main configuration
            try:
                if main_config and main_config != "default":
                    splits = get_dataset_split_names(dataset_id, config_name=main_config)
                else:
                    splits = get_dataset_split_names(dataset_id)
                
                result["structure"]["splits"] = splits
                result["available_files"] = splits
            except Exception as e:
                # Fallback: try common split names
                common_splits = ["train", "test", "validation", "dev"]
                available_splits = []
                
                for split in common_splits:
                    try:
                        # Try to load just one example to check if split exists
                        test_dataset = load_dataset(
                            dataset_id, 
                            split=f"{split}[:1]", 
                            trust_remote_code=True,
                            streaming=True
                        )
                        list(test_dataset.take(1))  # Test if we can actually access it
                        available_splits.append(split)
                    except Exception:
                        continue
                
                if available_splits:
                    result["structure"]["splits"] = available_splits
                    result["available_files"] = available_splits
                else:
                    result["structure"]["splits"] = ["Unable to determine splits"]
                    result["available_files"] = ["Unable to access dataset structure"]
            
            # Get sample preview from the first available split
            preview_split = None
            if result["available_files"] and result["available_files"][0] != "Unable to access dataset structure":
                preview_split = result["available_files"][0]
                
                try:
                    # Load a small sample (5 examples) from the first split
                    sample_data = load_dataset(
                        dataset_id,
                        split=f"{preview_split}[:5]",
                        trust_remote_code=True
                    )
                    
                    if len(sample_data) > 0:
                        # Get column information
                        columns = sample_data.column_names
                        result["structure"]["columns"] = columns
                        
                        # Create sample preview
                        samples = []
                        for i, example in enumerate(sample_data):
                            sample_dict = {}
                            for col in columns:
                                value = example[col]
                                # Truncate long text for readability
                                if isinstance(value, str) and len(value) > 200:
                                    sample_dict[col] = value[:200] + "..."
                                else:
                                    sample_dict[col] = value
                            samples.append(sample_dict)
                        
                        result["sample_preview"] = {
                            "split": preview_split,
                            "total_columns": len(columns),
                            "sample_size": len(samples),
                            "examples": samples
                        }
                    
                except Exception as e:
                    result["sample_preview"] = {
                        "error": f"Could not load sample data: {str(e)}",
                        "split": preview_split
                    }
            
            # Generate summary
            num_configs = len(result["structure"].get("configurations", []))
            num_splits = len(result["structure"].get("splits", []))
            num_columns = len(result["structure"].get("columns", []))
            
            summary_parts = []
            if num_configs > 1:
                summary_parts.append(f"{num_configs} configurations")
            if num_splits > 0:
                summary_parts.append(f"{num_splits} splits ({', '.join(result['structure'].get('splits', []))})")
            if num_columns > 0:
                summary_parts.append(f"{num_columns} columns")
            
            if summary_parts:
                result["summary"] = f"Dataset '{dataset_id}' contains: " + ", ".join(summary_parts)
            else:
                result["summary"] = f"Dataset '{dataset_id}' structure could not be fully determined"
            
            return {
                "message": "Successfully explored dataset structure",
                "exploration_results": result
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific errors
            if ("does not exist" in error_msg.lower() or 
                "not found" in error_msg.lower() or
                "couldn't find" in error_msg.lower()):
                return {
                    "error": f"Dataset '{dataset_id}' not found on Hugging Face Hub",
                    "suggestion": "Please verify the dataset ID. Format should be 'username/dataset-name' or 'dataset-name' for official datasets."
                }
            
            if ("gated" in error_msg.lower() or "access" in error_msg.lower()):
                return {
                    "error": f"Dataset '{dataset_id}' requires authentication or special access",
                    "suggestion": "This dataset may be gated and require approval or authentication to access."
                }
            
            # Retry for temporary issues
            if ("503" in error_msg or "timeout" in error_msg.lower() or 
                "unavailable" in error_msg.lower() or "connection" in error_msg.lower()):
                
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return {
                        "error": f"Service temporarily unavailable after {retries} retries",
                        "suggestion": "Please try again in a few minutes."
                    }
            else:
                # Non-retryable error
                return {
                    "error": f"Failed to explore dataset: {error_msg}",
                    "dataset_id": dataset_id
                }
    
    return {"error": "Unexpected error in retry logic"}

@mcp.tool
def explore_dataset_structure(dataset_id: str, split_name: str = None):
    retries = 3
    
    for attempt in range(retries):
        try:
            result = {
                "dataset_id": dataset_id,
                "url": f"https://huggingface.co/datasets/{dataset_id}",
                "overview": {},
                "available_splits": [],
                "file_types": [],
                "sample_data": {},
                "description": "",
                "next_steps": ""
            }
            
            # Get dataset configurations
            try:
                configs = get_dataset_config_names(dataset_id)
                main_config = configs[0] if configs else None
                result["overview"]["configurations"] = configs
            except Exception:
                configs = ["default"]
                main_config = None
                result["overview"]["configurations"] = ["default"]
            
            # Get available splits
            try:
                if main_config and main_config != "default":
                    splits = get_dataset_split_names(dataset_id, config_name=main_config)
                else:
                    splits = get_dataset_split_names(dataset_id)
                
                result["available_splits"] = splits
                result["overview"]["total_splits"] = len(splits)
            except Exception:
                # Fallback: try common splits
                common_splits = ["train", "test", "validation", "dev"]
                available_splits = []
                
                for split in common_splits:
                    try:
                        test_load = load_dataset(dataset_id, split=f"{split}[:1]", trust_remote_code=True)
                        available_splits.append(split)
                    except Exception:
                        continue
                
                result["available_splits"] = available_splits if available_splits else ["Unknown"]
                result["overview"]["total_splits"] = len(available_splits) if available_splits else 0
            
            # Detect file types by loading repo files
            try:
                files = api.list_repo_files(dataset_id, repo_type="dataset")
                file_extensions = set()
                for file in files:
                    if '.' in file:
                        ext = file.split('.')[-1].lower()
                        if ext in ['csv', 'json', 'jsonl', 'parquet', 'txt', 'tsv', 'arrow']:
                            file_extensions.add(f".{ext}")
                
                result["file_types"] = list(file_extensions) if file_extensions else ["Unable to determine"]
            except Exception:
                result["file_types"] = ["Unable to access file information"]
            
            # If no specific split requested, show overview and ask user to choose
            if split_name is None:
                result["description"] = f"Dataset '{dataset_id}' overview completed. Available splits: {', '.join(result['available_splits'])}"
                result["next_steps"] = f"To explore a specific split, use: explore_dataset_structure('{dataset_id}', 'SPLIT_NAME')"
                
                # Show basic info about the first split as a preview
                if result["available_splits"] and result["available_splits"][0] != "Unknown":
                    preview_split = result["available_splits"][0]
                    try:
                        sample = load_dataset(dataset_id, split=f"{preview_split}[:1]", trust_remote_code=True)
                        if len(sample) > 0:
                            result["sample_data"]["preview_split"] = preview_split
                            result["sample_data"]["columns"] = sample.column_names
                            result["sample_data"]["total_columns"] = len(sample.column_names)
                            result["sample_data"]["note"] = f"Showing column structure from '{preview_split}' split"
                    except Exception:
                        pass
                
                return {
                    "message": f"Dataset structure overview for '{dataset_id}'",
                    "structure_info": result
                }
            
            # If specific split requested, show detailed preview
            else:
                if split_name not in result["available_splits"]:
                    return {
                        "error": f"Split '{split_name}' not found in dataset '{dataset_id}'",
                        "available_splits": result["available_splits"],
                        "suggestion": f"Available splits are: {', '.join(result['available_splits'])}"
                    }
                
                try:
                    # Load 5 examples from the requested split
                    sample_data = load_dataset(dataset_id, split=f"{split_name}[:5]", trust_remote_code=True)
                    
                    if len(sample_data) > 0:
                        columns = sample_data.column_names
                        
                        # Create table-style preview
                        table_data = []
                        for i, example in enumerate(sample_data):
                            row = {"Row": i + 1}
                            for col in columns:
                                value = example[col]
                                # Handle different data types and truncate long content
                                if isinstance(value, str):
                                    display_value = value[:100] + "..." if len(value) > 100 else value
                                elif isinstance(value, (list, dict)):
                                    display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                                else:
                                    display_value = str(value)
                                
                                row[col] = display_value
                            table_data.append(row)
                        
                        result["sample_data"] = {
                            "split": split_name,
                            "total_columns": len(columns),
                            "column_names": columns,
                            "sample_rows": table_data,
                            "total_examples_shown": len(table_data)
                        }
                        
                        # Try to get total count (may not always work)
                        try:
                            full_split = load_dataset(dataset_id, split=split_name, trust_remote_code=True)
                            result["sample_data"]["total_examples_in_split"] = len(full_split)
                        except Exception:
                            result["sample_data"]["total_examples_in_split"] = "Unable to determine"
                        
                        result["description"] = f"Detailed view of '{split_name}' split from '{dataset_id}'"
                        
                    else:
                        result["sample_data"] = {"error": f"Split '{split_name}' appears to be empty"}
                        
                except Exception as e:
                    result["sample_data"] = {"error": f"Could not load data from split '{split_name}': {str(e)}"}
                
                return {
                    "message": f"Detailed exploration of '{dataset_id}' - '{split_name}' split",
                    "structure_info": result
                }
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific errors
            if ("does not exist" in error_msg.lower() or 
                "not found" in error_msg.lower() or
                "couldn't find" in error_msg.lower()):
                return {
                    "error": f"Dataset '{dataset_id}' not found on Hugging Face Hub",
                    "suggestion": "Please verify the dataset ID format (e.g., 'squad', 'imdb', 'username/dataset-name')"
                }
            
            if ("gated" in error_msg.lower() or "access" in error_msg.lower()):
                return {
                    "error": f"Dataset '{dataset_id}' requires authentication or approval",
                    "suggestion": "This dataset may be gated. You may need to request access or authenticate."
                }
            
            # Retry for temporary issues
            if ("503" in error_msg or "timeout" in error_msg.lower() or 
                "unavailable" in error_msg.lower() or "connection" in error_msg.lower()):
                
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return {
                        "error": f"Service temporarily unavailable after {retries} retries",
                        "suggestion": "Please try again in a few minutes."
                    }
            else:
                return {
                    "error": f"Failed to explore dataset structure: {error_msg}",
                    "dataset_id": dataset_id
                }
    
    return {"error": "Unexpected error in retry logic"}


if __name__ == "__main__":
    mcp.run()