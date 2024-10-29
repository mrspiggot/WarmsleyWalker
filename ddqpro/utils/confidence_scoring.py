from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceMetrics:
    """Metrics used to calculate confidence score"""
    context_relevance: float  # How relevant is the context to the question
    answer_completeness: float  # How complete is the answer
    source_quality: float  # Quality of sources used
    llm_confidence: float  # LLM's own confidence signals
    consistency: float  # Add this line - it was missing
    confidence_score: float = 0.0  # Add a default overall score


class ConfidenceScorer:
    def __init__(self):
        # Weights for different components
        self.weights = {
            'context_relevance': 0.25,
            'answer_completeness': 0.25,
            'source_quality': 0.20,
            'llm_confidence': 0.15,
            'consistency': 0.15
        }

    def calculate_confidence(self,
                             question: str,
                             answer: str,
                             context: List[Dict],
                             response_metadata: Dict,
                             previous_answers: Dict = None) -> ConfidenceMetrics:
        """Calculate comprehensive confidence score"""
        try:
            # Calculate individual metrics
            metrics = ConfidenceMetrics(
                context_relevance=self._calculate_context_relevance(question, context),
                answer_completeness=self._calculate_completeness(question, answer),
                source_quality=self._evaluate_sources(context),
                llm_confidence=self._get_llm_confidence(response_metadata),
                consistency_score=self._check_consistency(answer, context, previous_answers)
            )

            # Calculate weighted final score
            final_score = sum(
                getattr(metrics, metric) * weight
                for metric, weight in self.weights.items()
            )

            logger.debug(f"Confidence metrics for answer: {metrics}")
            return final_score, metrics

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5, None  # Default moderate confidence if calculation fails

    def _calculate_context_relevance(self, question: str, context: List[Dict]) -> float:
        """Calculate how relevant the context is to the question"""
        try:
            # Calculate term overlap between question and context
            question_terms = set(question.lower().split())

            relevance_scores = []
            for ctx in context:
                ctx_terms = set(ctx['content'].lower().split())

                # Calculate Jaccard similarity
                overlap = len(question_terms & ctx_terms)
                union = len(question_terms | ctx_terms)
                score = overlap / union if union > 0 else 0

                # Weight by position in context
                position_weight = 1.0 / (context.index(ctx) + 1)
                relevance_scores.append(score * position_weight)

            return min(1.0, sum(relevance_scores))

        except Exception as e:
            logger.error(f"Error calculating context relevance: {str(e)}")
            return 0.5

    def _calculate_completeness(self, question: str, answer: str) -> float:
        """Evaluate how complete the answer is"""
        try:
            # Check for key question elements
            question_indicators = {
                'what': 'description',
                'how': 'process',
                'why': 'explanation',
                'when': 'timing',
                'where': 'location',
                'who': 'person/entity',
                'list': 'enumeration'
            }

            completeness_scores = []

            # Check if answer addresses question requirements
            for indicator, requirement in question_indicators.items():
                if indicator in question.lower():
                    # Check if answer contains related content
                    score = self._check_requirement_met(answer, requirement)
                    completeness_scores.append(score)

            # Check answer length adequacy
            expected_length = self._estimate_expected_length(question)
            length_score = min(1.0, len(answer) / expected_length)
            completeness_scores.append(length_score)

            return np.mean(completeness_scores) if completeness_scores else 0.7

        except Exception as e:
            logger.error(f"Error calculating completeness: {str(e)}")
            return 0.5

    def _evaluate_sources(self, context: List[Dict]) -> float:
        """Evaluate the quality and relevance of sources"""
        try:
            source_scores = []
            for source in context:
                score = 0.0

                # Check source type
                if source.get('doc_type') == 'DDQ':
                    score += 0.4  # Direct DDQ sources are highly relevant

                # Check recency if available
                if 'date' in source:
                    age_score = self._calculate_recency_score(source['date'])
                    score += age_score * 0.3

                # Check content length/detail
                content_length = len(source.get('content', ''))
                length_score = min(1.0, content_length / 1000)  # Normalize to 1000 chars
                score += length_score * 0.3

                source_scores.append(score)

            return np.mean(source_scores) if source_scores else 0.5

        except Exception as e:
            logger.error(f"Error evaluating sources: {str(e)}")
            return 0.5

    def _get_llm_confidence(self, response_metadata: Dict) -> float:
        """Extract confidence signals from LLM response"""
        try:
            confidence_signals = []

            # Check token usage (higher usually means more detailed response)
            if 'token_usage' in response_metadata:
                token_score = min(1.0, response_metadata['token_usage'] / 1000)
                confidence_signals.append(token_score)

            # Check response generation time
            if 'generation_time' in response_metadata:
                time_score = self._calculate_time_score(response_metadata['generation_time'])
                confidence_signals.append(time_score)

            # Check model-specific confidence indicators
            if 'model_confidence' in response_metadata:
                confidence_signals.append(response_metadata['model_confidence'])

            return np.mean(confidence_signals) if confidence_signals else 0.7

        except Exception as e:
            logger.error(f"Error getting LLM confidence: {str(e)}")
            return 0.5

    def _check_consistency(self,
                           answer: str,
                           context: List[Dict],
                           previous_answers: Dict = None) -> float:
        """Check consistency with context and previous answers"""
        try:
            consistency_scores = []

            # Check consistency with context
            for ctx in context:
                similarity = SequenceMatcher(None, answer, ctx['content']).ratio()
                consistency_scores.append(similarity)

            # Check consistency with previous answers if available
            if previous_answers:
                for prev_answer in previous_answers.values():
                    similarity = SequenceMatcher(None, answer, prev_answer).ratio()
                    consistency_scores.append(similarity)

            return np.mean(consistency_scores) if consistency_scores else 0.6

        except Exception as e:
            logger.error(f"Error checking consistency: {str(e)}")
            return 0.5

    def _estimate_expected_length(self, question: str) -> int:
        """Estimate expected answer length based on question type"""
        # Basic length estimation based on question type
        if any(w in question.lower() for w in ['explain', 'describe', 'elaborate']):
            return 500
        elif any(w in question.lower() for w in ['list', 'enumerate']):
            return 300
        elif question.lower().startswith(('what', 'how')):
            return 200
        return 100

    def _calculate_recency_score(self, date_str: str) -> float:
        """Calculate score based on content recency"""
        try:
            date = datetime.fromisoformat(date_str)
            age_days = (datetime.now() - date).days
            return max(0.0, min(1.0, 1 - (age_days / 365)))
        except:
            return 0.5

    def _calculate_time_score(self, generation_time: float) -> float:
        """Calculate score based on response generation time"""
        # Assume optimal time is between 2-10 seconds
        if generation_time < 2:
            return 0.5  # Too fast might indicate canned response
        elif generation_time > 10:
            return max(0.3, 1 - (generation_time - 10) / 20)
        else:
            return 1.0


