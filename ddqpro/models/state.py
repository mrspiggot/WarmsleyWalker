from typing import Dict, List, Optional, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class QuestionMetadata(BaseModel):
    """Metadata for a DDQ question"""
    category: str = Field(description="Primary category of the question")
    subcategory: Optional[str] = Field(description="Optional subcategory")
    context: Optional[str] = Field(description="Additional context or instructions")


class Question(BaseModel):
    """Represents a single DDQ question"""
    id: str = Field(description="Unique question identifier")
    text: str = Field(description="Full question text")
    type: str = Field(description="Question type (text, multiple_choice, table)")
    required: bool = Field(description="Whether the question is required")
    metadata: QuestionMetadata = Field(description="Question metadata")
    section: str = Field(description="Section title containing this question")


class Section(BaseModel):
    """Represents a section in the DDQ"""
    title: str = Field(description="Section title")
    level: int = Field(description="Hierarchical level of the section")
    questions: List[Question] = Field(description="Questions in this section")


class ExtractionResult(BaseModel):
    """Results from a document extraction attempt"""
    section_count: int = Field(description="Number of sections identified")
    question_count: int = Field(description="Number of questions extracted")
    content: Dict = Field(description="Extracted content including sections and questions")
    confidence: float = Field(description="Confidence in extraction quality")


class DDQReflection(BaseModel):
    """Reflection on extraction quality"""
    missing_sections: str = Field(description="Identified missing sections")
    extraction_issues: str = Field(description="Problems with extraction")
    suggested_improvements: str = Field(description="How to improve extraction")


class ResponseGeneration(BaseModel):
    """Generated response for a DDQ question"""
    question_id: str = Field(description="ID of the question being answered")
    response_text: str = Field(description="Generated response text")
    confidence: float = Field(description="Confidence score of the response")
    sources: List[Dict] = Field(description="Source documents used for response")


class ResponseReflection(BaseModel):
    """Reflection on response quality"""
    accuracy_score: float = Field(description="Score for factual accuracy", ge=0, le=1)
    completeness_score: float = Field(description="Score for completeness", ge=0, le=1)
    consistency_score: float = Field(description="Score for consistency with other responses", ge=0, le=1)
    issues: List[str] = Field(description="Identified issues with the response")
    suggestions: List[str] = Field(description="Suggestions for improvement")


class ResponseState(TypedDict):
    """State for response generation process"""
    question: Question
    context: List[Dict]
    current_response: Optional[ResponseGeneration]
    reflections: List[ResponseReflection]
    attempts: int
class Answer(BaseModel):
    """Answer to a DDQ question"""
    text: str = Field(description="Generated answer text")
    confidence: float = Field(description="Confidence score for the answer", ge=0, le=1)
    sources: List[Dict] = Field(description="Source documents used for the answer")
    metadata: Dict = Field(description="Additional metadata about the answer")

class QuestionAnswerPair(BaseModel):
    """Represents a question and its answer"""
    question: Question
    answer: Optional[Answer] = None

# Update ExtractionResult to include answers
class ExtractionResult(BaseModel):
    """Results from a document extraction attempt"""
    section_count: int = Field(description="Number of sections identified")
    question_count: int = Field(description="Number of questions extracted")
    content: Dict = Field(description="Extracted content including sections and questions")
    confidence: float = Field(description="Confidence in extraction quality")
    answers: Dict[str, Answer] = Field(default_factory=dict, description="Answers for each question")

# Update DDQState to include response information
class DDQState(TypedDict):
    """State management for DDQ processing"""
    input_path: str
    file_type: str
    current_extractor: str
    extraction_results: Optional[ExtractionResult]
    reflections: List[DDQReflection]
    json_output: Optional[Dict]
    response_states: Dict[str, ResponseState]  # question_id -> ResponseState
    completed_responses: Dict[str, ResponseGeneration]  # question_id -> final response
    cost_tracking: Dict[str, Any]  # Cost tracking information