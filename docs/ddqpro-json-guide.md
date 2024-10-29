# DDQPro JSON Structure User Manual

## Overview
The DDQPro tool generates a structured JSON output containing analyzed questions from Due Diligence Questionnaires (DDQs) and their corresponding AI-generated answers. This document explains how to interpret each section of the JSON output.

## Top-Level Structure

```json
{
  "metadata": { ... },
  "analysis": { ... },
  "potential_issues": [ ... ]
}
```

### Metadata Section
Contains basic information about the processed document:
- `file_name`: Original document filename
- `file_type`: File format (.pdf, .docx)
- `extractor`: Extraction method used (typically "default")

### Analysis Section
Contains the core analysis results:
- `section_count`: Total number of identified sections
- `question_count`: Total number of questions found
- `confidence_score`: Overall document analysis confidence (0-1)
- `sections`: Array of analyzed sections
- `questions`: Flat dictionary of all questions

## Question Fields

### Question Structure
Each question in the JSON contains the following fields:

```json
{
  "id": "1.01",
  "text": "Original question text",
  "type": "text|multiple_choice|table",
  "required": true|false,
  "metadata": {
    "category": "Category classification",
    "subcategory": "Optional subcategory",
    "context": "Additional context"
  },
  "section": "Section name"
}
```

#### Question Field Interpretations:
- `id`: Unique identifier, typically follows section numbering (e.g., "1.01")
- `text`: Complete question text as extracted from document
- `type`: Question format classification
  - `text`: Free-form text response
  - `multiple_choice`: Selection from options
  - `table`: Tabular data required
- `required`: Boolean indicating if question is mandatory
- `metadata`:
  - `category`: Primary classification (e.g., "company", "financial", "regulatory")
  - `subcategory`: Optional further classification
  - `context`: Additional contextual information or instructions
- `section`: Parent section name

## Answer Fields

### Answer Structure
Generated answers contain detailed information about the response and its generation:

```json
{
  "text": "Generated answer content",
  "confidence": 0.85,
  "sources": [
    {
      "file": "Source document name",
      "excerpt": "Relevant excerpt from source",
      "content": "Full context used"
    }
  ],
  "metadata": {
    "question_type": "Type of question answered",
    "generated_at": "ISO timestamp",
    "context_query": "Query used for retrieval",
    "key_terms": ["Important", "terms", "identified"],
    "confidence_metrics": {
      "context_relevance": 0.8,
      "answer_completeness": 0.9,
      "source_quality": 0.85,
      "llm_confidence": 0.9,
      "consistency": 0.8
    }
  }
}
```

#### Answer Field Interpretations:

##### Core Fields
- `text`: The actual generated answer
- `confidence`: Overall confidence score (0-1) representing answer quality
- `sources`: Array of reference materials used to generate the answer

##### Sources
Each source contains:
- `file`: Origin document name
- `excerpt`: Relevant portion used (first 200 characters)
- `content`: Complete context chunk used

##### Metadata
Detailed information about answer generation:
- `question_type`: Matches question type classification
- `generated_at`: Timestamp of answer generation
- `context_query`: Search query used to find relevant context
- `key_terms`: Important terms identified in the question

##### Confidence Metrics
Detailed breakdown of confidence score components:
- `context_relevance` (0-1): How well the source material matches the question
- `answer_completeness` (0-1): Whether all aspects of question are addressed
- `source_quality` (0-1): Reliability and relevance of sources used
- `llm_confidence` (0-1): AI model's internal confidence measure
- `consistency` (0-1): Agreement between different sources and previous answers

## Interpreting Confidence Scores

### Overall Confidence Score
- 0.9-1.0: Highly reliable answer with strong supporting evidence
- 0.7-0.9: Good confidence, suitable for most purposes
- 0.5-0.7: Moderate confidence, may need human verification
- <0.5: Low confidence, should be manually reviewed

### Component Confidence Interpretations
- `context_relevance`: High scores (>0.8) indicate well-matched source material
- `answer_completeness`: Scores >0.9 suggest comprehensive answers
- `source_quality`: High scores indicate reliable, recent sources
- `llm_confidence`: Model's self-assessment of answer quality
- `consistency`: High scores indicate agreement across sources

## Example Usage

To evaluate an answer's reliability:
1. Check overall `confidence` score first
2. Review `sources` to verify information origin
3. Examine `confidence_metrics` for specific weaknesses
4. Check `generated_at` for recency
5. Review `context_query` and `key_terms` to understand context retrieval

## Best Practices

1. Always review answers with confidence scores below 0.7
2. Check source excerpts for answer verification
3. Pay attention to question type matching in metadata
4. Use confidence metrics to identify potential issues
5. Consider timestamp for time-sensitive information

## Potential Issues Array
Contains warnings or problems identified during processing:
- Document formatting issues
- Missing sections
- Ambiguous questions
- Low confidence scores
- Context retrieval problems
