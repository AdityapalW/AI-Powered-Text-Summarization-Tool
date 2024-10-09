from transformers import pipeline
import pandas as pd

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=50, min_length=40): #set the lenght of summarization here
    """
    Summarizes a given text using a pre-trained summarization model.

    Parameters:
        text (str): The text you want to summarize.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        str: Summarized text.
    """
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    #Add text here that you want to summarize
    long_text = """
    Artificial Intelligence (AI) is a broad field of computer science focused on creating systems capable of performing tasks that would typically require human intelligence. 
    These tasks include learning, reasoning, problem-solving, understanding natural language, and sensory perception. AI is an interdisciplinary science with multiple approaches, 
    but advancements in machine learning and deep learning are creating a paradigm shift in virtually every sector of the tech industry.
    """

    summary = summarize_text(long_text)
    print("Original Text:\n", long_text)
    print("\nSummarized Text:\n", summary)

    
