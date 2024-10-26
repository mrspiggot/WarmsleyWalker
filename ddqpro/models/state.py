from typing import Dict, List, Optional
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


class DDQState(TypedDict):
    """State management for DDQ processing"""
    input_path: str
    file_type: str
    current_extractor: str
    extraction_results: Optional[ExtractionResult]
    reflections: List[DDQReflection]
    json_output: Optional[Dict]