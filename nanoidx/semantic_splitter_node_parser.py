import numpy as np

from tqdm import tqdm
from openai import OpenAI
from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer


class SemanticSplitterNodeParser:
    """
    A semantic text splitter that uses sentence embeddings to intelligently 
    split text into coherent chunks based on semantic similarity.
    """

    def __init__(self, 
                 buffer_size: int = 1,
                 model: str = "model-identifier",
                 base_url: str = "http://localhost:1234/v1",
                 api_key: str = "lm-studio",
                 breakpoint_percentile_threshold: int = 95):
        """
        Initialize the SemanticSplitterNodeParser.
        
        Args:
            buffer_size (int): The size of the buffer for sentence grouping. Defaults to 1.
            model (str): The model identifier for embeddings. Defaults to "model-identifier".
            base_url (str): The base URL for the OpenAI client. Defaults to "http://localhost:1234/v1".
            api_key (str): The API key for the OpenAI client. Defaults to "lm-studio".
            breakpoint_percentile_threshold (int): The threshold for the percentile of the distance between sentences. Defaults to 95.
        """
        
        self.buffer_size = buffer_size
        self.model = model
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = PunktSentenceTokenizer()
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold

    def split_by_sentence_tokenizer(self, text: str) -> List[str]:
        """
        Splits text by a sentence tokenizer.

        Args:
            text (str): The text to split.

        Returns:
           List[str]: A list of split strings.
        """
        spans = list(self.tokenizer.span_tokenize(text))
        sentences = []
        for i, span in enumerate(spans):
            start = span[0]
            if i < len(spans) - 1:
                end = spans[i + 1][0]
            else:
                end = len(text)
            sentences.append(text[start:end])
        return sentences
    
    def build_sentence_groups(self, sentences: List[str]) -> List[str]:
        """ 
        Create a buffer by combining each sentence with its previous and next sentence 
        to provide a wider context. 

        Args:
            sentences (List[str]): The list of sentences to be combined
        
        Returns:
            List[str]: The list of combined sentences
        """
        combined_sentences = []
        for i in range(len(sentences)):
            combined_sentence = ""

            for j in range(i - self.buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]
            
            combined_sentence += sentences[i]

            for j in range(i + 1, i + self.buffer_size + 1):
                if j < len(sentences):
                    combined_sentence += sentences[j]

            combined_sentences.append(combined_sentence)
        return combined_sentences
    
    def get_text_embedding(self, text: str) -> List[float]:
        """ 
        Get the embedding of the text using the configured model.

        Args:
            text (str): The text to be embedded

        Returns:
            List[float]: The embedding of the text
        """
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
    
    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get the text embeddings of the whole text batch

        Args:
            texts (List[str]): The text batch to be embedded

        Returns:
            List[List[float]]: The text embeddings of the whole text batch
        """
        return [self.get_text_embedding(text) for text in tqdm(texts, desc="Text Embedding...")]
    
    def get_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """
        Get the similarity between two embeddings using the cosine similarity

        Args:
            emb1 (List[float]): The first embedding
            emb2 (List[float]): The second embedding

        Returns:
            float: The similarity between the two embeddings
        """
        if isinstance(emb1, list) and isinstance(emb2, list):
            emb1 = np.array(emb1)
            emb2 = np.array(emb2)

        product = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        return product / norm
    
    def calculate_distances_between_embeddings(self, embs: List[List[float]]) -> List[float]:
        """
        Calculate the distances between two consecutive embeddings in the list

        Args:
            embs (List[List[float]]): The list of embeddings to be compared

        Returns:
            List[float]: The list of distances between all pairs of consecutive embeddings
        """
        distances = []
        for i in tqdm(range(len(embs) - 1), desc="Calculating distances between pairs of embeddings..."):
            emb_current = embs[i]
            emb_next = embs[i + 1]

            similarity = self.get_similarity(emb_current, emb_next)
            distance = 1 - similarity  # using the 1 - cosine similarity to measure distance

            distances.append(distance)
        return distances
    
    def build_text_chunk(self, 
                         combined_sentences: List[str], 
                         embs_distances: List[float], 
                         threshold: float) -> List[str]:
        """
        Build text chunks based on the combined sentences and the distances between embeddings.
        If the distance between two consecutive embeddings is greater than a threshold,
        it means that the two sentences are not similar enough to be merged into one chunk.
        Therefore, we need to split the combined sentences into separate chunks.
        
        Args:
            combined_sentences (List[str]): The combined sentences.
            embs_distances (List[float]): The distances between embeddings.
            threshold (float): The threshold for determining whether two sentences are similar enough to be merged into one chunk.

        Returns:
            List[str]: The text chunks.
        """
        chunks = []
        if len(embs_distances) > 0:
            breakpoint_distance_threshold = np.percentile(embs_distances, threshold)

            indices_above_threshold = [
                i for i, x in enumerate(embs_distances) if x > breakpoint_distance_threshold
            ]

            start_idx = 0
            for idx in tqdm(indices_above_threshold, desc="Building Text Chunks..."):
                group = combined_sentences[start_idx : idx + 1]
                combined_text = "".join(group)
                chunks.append(combined_text)

                start_idx = idx + 1

            if start_idx < len(combined_sentences):
                combined_text = "".join(combined_sentences[start_idx:])
                chunks.append(combined_text)
        else:
            chunks.append(" ".join(combined_sentences))

        return chunks
    
    def split_text(self, text: str) -> List[str]:
        """
        Main method to split text into semantic chunks.
        
        Args:
            text (str): The input text to be split.
            threshold (float): The percentile threshold for determining chunk boundaries. 
                              Defaults to 95.0.
        
        Returns:
            List[str]: The list of semantic text chunks.
        """
        # Step 1: Split text into sentences
        sentences = self.split_by_sentence_tokenizer(text)
        
        # Step 2: Build sentence groups with context buffer
        combined_sentences = self.build_sentence_groups(sentences)
        
        # Step 3: Get embeddings for all combined sentences
        embeddings = self.get_text_embedding_batch(combined_sentences)
        
        # Step 4: Calculate distances between consecutive embeddings
        distances = self.calculate_distances_between_embeddings(embeddings)
        
        # Step 5: Build text chunks based on semantic similarity
        chunks = self.build_text_chunk(combined_sentences, distances, self.breakpoint_percentile_threshold)
        
        return chunks