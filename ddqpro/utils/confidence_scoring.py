from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceMetrics:
    """Metrics used to calculate confidence score"""
    context_relevance: float
    answer_completeness: float
    source_quality: float
    llm_confidence: float
    consistency: float  # Changed from consistency_score
    overall_confidence: float = 0.0


class ConfidenceScorer:
    def __init__(self):
        self.weights = {
            'context_relevance': 0.25,
            'answer_completeness': 0.25,
            'source_quality': 0.20,
            'llm_confidence': 0.15,
            'consistency': 0.15  # Changed from consistency_score
        }

    def calculate_confidence(self,
                             question: str,
                             answer: str,
                             context: List[Dict],
                             response_metadata: Dict,
                             previous_answers: Dict = None) -> tuple[float, ConfidenceMetrics]:
        """Calculate comprehensive confidence score"""
        try:
            # Calculate individual metrics
            context_relevance = self._calculate_context_relevance(question, context)
            answer_completeness = self._calculate_completeness(question, answer)
            source_quality = self._evaluate_sources(context)
            llm_confidence = self._get_llm_confidence(response_metadata)
            consistency = self._check_consistency(answer, context, previous_answers)

            # Calculate overall confidence
            overall_confidence = (
                    context_relevance * self.weights['context_relevance'] +
                    answer_completeness * self.weights['answer_completeness'] +
                    source_quality * self.weights['source_quality'] +
                    llm_confidence * self.weights['llm_confidence'] +
                    consistency * self.weights['consistency']
            )

            # Create metrics object
            metrics = ConfidenceMetrics(
                context_relevance=context_relevance,
                answer_completeness=answer_completeness,
                source_quality=source_quality,
                llm_confidence=llm_confidence,
                consistency=consistency,
                overall_confidence=overall_confidence
            )

            logger.debug(f"Confidence metrics: {metrics}")
            return overall_confidence, metrics

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            # Return moderate confidence with None metrics on error
            return 0.5, None

    def _calculate_context_relevance(self, question: str, context: List[Dict]) -> float:
        try:
            if not context:
                return 0.5

            # Calculate term overlap between question and context
            question_terms = set(question.lower().split())

            relevance_scores = []
            for ctx in context:
                ctx_terms = set(ctx['content'].lower().split())

                # Calculate Jaccard similarity
                overlap = len(question_terms & ctx_terms)
                union = len(question_terms | ctx_terms)
                score = overlap / union if union > 0 else 0

                # Weight by position
                position_weight = 1.0 / (context.index(ctx) + 1)
                relevance_scores.append(score * position_weight)

            return min(1.0, sum(relevance_scores))

        except Exception as e:
            logger.error(f"Error calculating context relevance: {str(e)}")
            return 0.5

    def _calculate_completeness(self, question: str, answer: str) -> float:
        try:
            if not answer:
                return 0.0

            # Basic completeness checks
            checks = {
                'has_content': len(answer.strip()) > 0,
                'reasonable_length': 50 <= len(answer) <= 1000,
                'addresses_question': any(term in answer.lower()
                                          for term in question.lower().split())
            }

            completeness_score = sum(checks.values()) / len(checks)
            return completeness_score

        except Exception as e:
            logger.error(f"Error calculating completeness: {str(e)}")
            return 0.5

    def _evaluate_sources(self, context: List[Dict]) -> float:
        try:
            if not context:
                return 0.5

            source_scores = []
            for source in context:
                score = 0.0

                # Check source type
                if source.get('doc_type') == 'DDQ':
                    score += 0.4

                # Check content length
                content_length = len(source.get('content', ''))
                length_score = min(1.0, content_length / 1000)
                score += length_score * 0.3

                source_scores.append(score)

            return np.mean(source_scores) if source_scores else 0.5

        except Exception as e:
            logger.error(f"Error evaluating sources: {str(e)}")
            return 0.5

    def _get_llm_confidence(self, response_metadata: Dict) -> float:
        try:
            confidence_signals = []

            # Check token usage
            if 'token_usage' in response_metadata:
                token_score = min(1.0, response_metadata['token_usage'] / 1000)
                confidence_signals.append(token_score)

            # Check generation time
            if 'generation_time' in response_metadata:
                time_score = min(1.0, response_metadata['generation_time'] / 10)
                confidence_signals.append(time_score)

            return np.mean(confidence_signals) if confidence_signals else 0.7

        except Exception as e:
            logger.error(f"Error getting LLM confidence: {str(e)}")
            return 0.5

    def _check_consistency(self,
                           answer: str,
                           context: List[Dict],
                           previous_answers: Dict = None) -> float:
        try:
            consistency_scores = []

            # Check consistency with context
            for ctx in context:
                # Simple text overlap check
                ctx_words = set(ctx['content'].lower().split())
                answer_words = set(answer.lower().split())
                overlap = len(ctx_words & answer_words)
                total = len(ctx_words | answer_words)

                if total > 0:
                    score = overlap / total
                    consistency_scores.append(score)

            # Check consistency with previous answers if available
            if previous_answers:
                for prev_answer in previous_answers.values():
                    prev_words = set(prev_answer.lower().split())
                    answer_words = set(answer.lower().split())
                    overlap = len(prev_words & answer_words)
                    total = len(prev_words | answer_words)

                    if total > 0:
                        score = overlap / total
                        consistency_scores.append(score)

            return np.mean(consistency_scores) if consistency_scores else 0.6

        except Exception as e:
            logger.error(f"Error checking consistency: {str(e)}")
            return 0.5