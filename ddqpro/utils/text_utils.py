from typing import List, Dict, Set, Tuple
import re
from collections import Counter
from difflib import SequenceMatcher
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model for better NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess

    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Enhanced stopwords combining spaCy and custom words
STOP_WORDS = set(nlp.Defaults.stop_words) | {
    'please', 'provide', 'describe', 'explain', 'list', 'tell', 'share',
    'discuss', 'detail', 'include', 'excluding', 'including', 'specify',
    'indicate', 'state', 'following', 'information', 'details'
}


class TextAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words=list(STOP_WORDS),
            ngram_range=(1, 2),
            max_features=1000
        )

    def extract_key_terms(self, text: str) -> List[Tuple[str, float]]:
        """Extract key terms with their importance scores"""
        doc = nlp(text)

        # Extract named entities
        entities = {ent.text: 1.5 for ent in doc.ents}  # Give higher weight to entities

        # Extract noun phrases
        noun_phrases = {chunk.text: 1.2 for chunk in doc.noun_chunks}

        # Extract important individual tokens
        tokens = {}
        for token in doc:
            if (not token.is_stop and not token.is_punct and
                    token.text.lower() not in STOP_WORDS):
                # Score based on part of speech
                score = self._get_pos_score(token)
                tokens[token.lemma_] = score

        # Combine all terms and scores
        all_terms = {**entities, **noun_phrases, **tokens}

        # Sort by score
        return sorted(all_terms.items(), key=lambda x: x[1], reverse=True)

    def _get_pos_score(self, token) -> float:
        """Score token based on part of speech"""
        pos_scores = {
            'PROPN': 1.3,  # Proper nouns
            'NOUN': 1.0,  # Common nouns
            'VERB': 0.8,  # Verbs
            'ADJ': 0.7,  # Adjectives
            'NUM': 0.9,  # Numbers
        }
        return pos_scores.get(token.pos_, 0.5)

    def group_similar_questions(self,
                                questions: List[Dict],
                                threshold: float = 0.6) -> Dict[int, List[int]]:
        """Group questions using semantic similarity"""
        # Prepare texts for vectorization
        texts = [q.text if hasattr(q, 'text') else q['text'] for q in
                 questions]  # Handle both Question objects and dicts

        # Get TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Calculate similarity matrix
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

        # Group questions
        groups = {}
        group_id = 0
        processed = set()

        for i in range(len(questions)):
            if i in processed:
                continue

            # Start new group
            current_group = [i]
            processed.add(i)

            # Find similar questions
            for j in range(i + 1, len(questions)):
                if j in processed:
                    continue

                if similarity_matrix[i, j] >= threshold:
                    current_group.append(j)
                    processed.add(j)

            # Store group
            groups[group_id] = {
                'indices': current_group,
                'questions': [questions[idx] for idx in current_group],
                'centroid': np.mean(tfidf_matrix[current_group].toarray(), axis=0)
            }
            group_id += 1

        return groups

    def create_optimized_query(self,
                               terms: List[Tuple[str, float]],
                               max_length: int = 200) -> str:
        """Create optimized search query from key terms"""
        # Sort terms by importance score
        sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)

        # Build query respecting max length
        query_terms = []
        current_length = 0

        for term, score in sorted_terms:
            if current_length + len(term) + 1 <= max_length:
                query_terms.append(term)
                current_length += len(term) + 1
            else:
                break

        return " ".join(query_terms)

    def analyze_question_type(self, text: str) -> Dict[str, float]:
        """Analyze the type and complexity of a question"""
        doc = nlp(text)

        # Detect question characteristics
        characteristics = {
            'requires_calculation': any(token.like_num for token in doc),
            'requires_list': bool(re.search(r'list|enumerate|specify', text.lower())),
            'requires_explanation': bool(re.search(r'explain|describe|discuss|how|why', text.lower())),
            'binary_question': doc[0].tag_ in ['MD', 'VBP', 'VBZ'],  # Modal verbs or aux verbs
            'multiple_parts': text.count('?') + text.count(';')
        }

        # Estimate complexity
        complexity = self._estimate_complexity(doc, characteristics)

        return {
            'type': self._determine_question_type(characteristics),
            'complexity': complexity,
            'characteristics': characteristics
        }

    def _estimate_complexity(self, doc, characteristics: Dict) -> float:
        """Estimate question complexity (0-1)"""
        factors = [
            len(doc) / 50,  # Length factor
            len([t for t in doc if t.dep_ in ['nsubj', 'dobj', 'iobj']]) / 5,  # Syntactic complexity
            characteristics['multiple_parts'] * 0.2,
            characteristics['requires_calculation'] * 0.3,
            characteristics['requires_explanation'] * 0.2
        ]
        return min(1.0, sum(factors) / len(factors))

    def _determine_question_type(self, chars: Dict) -> str:
        """Determine the primary question type"""
        if chars['requires_calculation']:
            return 'quantitative'
        elif chars['requires_list']:
            return 'enumeration'
        elif chars['requires_explanation']:
            return 'explanation'
        elif chars['binary_question']:
            return 'binary'
        else:
            return 'factual'
# Common English words to filter out when extracting key terms
COMMON_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
    'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
    'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
    'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
    'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
    'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
    'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
    'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
    'give', 'day', 'most', 'us', 'has', 'had', 'was', 'were', 'been',
    'being', 'are', 'is', 'am', 'did', 'does', 'doing', 'must', 'should',
    'please', 'provide', 'describe', 'explain', 'list', 'tell', 'share',
    'discuss', 'detail', 'include', 'excluding', 'including', 'specify',
    'indicate', 'state'
}


def extract_key_terms(text: str) -> set:
    """Extract key terms from text, removing common words and punctuation"""
    # Convert to lowercase and split
    words = text.lower().split()

    # Remove punctuation and common words
    clean_words = {
        word.strip('.,?!()[]{}:;"\'')
        for word in words
        if len(word) > 3 and word not in COMMON_WORDS
    }

    return clean_words


def create_search_query(terms: set, max_length: int = 200) -> str:
    """Create a search query from key terms, respecting max length"""
    return " ".join(sorted(terms))[:max_length]


def group_by_similarity(texts: List[str],
                        threshold: float = 0.7) -> Dict[int, List[int]]:
    """Group texts by similarity using key terms overlap"""
    groups = {}
    group_id = 0

    # Extract terms for each text
    text_terms = [extract_key_terms(text) for text in texts]

    # Group by term overlap
    processed = set()
    for i, terms1 in enumerate(text_terms):
        if i in processed:
            continue

        current_group = [i]
        processed.add(i)

        # Compare with all other texts
        for j, terms2 in enumerate(text_terms[i + 1:], i + 1):
            if j in processed:
                continue

            # Calculate similarity using Jaccard index
            similarity = len(terms1 & terms2) / len(terms1 | terms2)
            if similarity >= threshold:
                current_group.append(j)
                processed.add(j)

        groups[group_id] = current_group
        group_id += 1

    return groups


