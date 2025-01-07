from pydantic import BaseModel, Field
from typing import List, Optional

class QuestionMetadata(BaseModel):
    category: str
    subcategory: Optional[str]
    context: Optional[str]

class Question(BaseModel):
    id: str
    text: str
    type: str
    required: bool
    metadata: QuestionMetadata

class Section(BaseModel):
    title: str
    level: int
    questions: List[Question]

class DDQAnalysis(BaseModel):
    """Expected JSON structure from LLM"""
    section_count: int = Field(ge=0)
    total_question_count: int = Field(ge=0)
    sections: List[Section]
    completeness_score: float = Field(ge=0, le=1)
    potential_issues: List[str]