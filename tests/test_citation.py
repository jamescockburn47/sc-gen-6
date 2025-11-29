"""Tests for citation parser and verifier."""

import pytest

from src.generation.citation import CitationParser, CitationVerifier, verify_citations
from src.schema import Citation


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        {
            "chunk_id": "chunk_1",
            "text": "John Smith testified about the events.",
            "score": 0.85,
            "metadata": {
                "file_name": "witness.pdf",
                "page_number": 1,
                "paragraph_number": 1,
            },
        },
        {
            "chunk_id": "chunk_2",
            "text": "The meeting occurred in 2023.",
            "score": 0.75,
            "metadata": {
                "file_name": "witness.pdf",
                "page_number": 1,
                "paragraph_number": 2,
            },
        },
        {
            "chunk_id": "chunk_3",
            "text": "Legal concept of fraud.",
            "score": 0.90,
            "metadata": {
                "file_name": "contract.pdf",
                "page_number": 5,
                "paragraph_number": 3,
                "section_header": "Definitions",
            },
        },
    ]


def test_citation_parser_basic():
    """Test basic citation parsing."""
    parser = CitationParser()
    text = 'John Smith testified [Source: witness.pdf | Page 1, Para 1]. The events occurred [Source: witness.pdf | Page 1, Para 2].'

    citations = parser.parse_citations(text)

    assert len(citations) == 2
    assert citations[0]["file_name"] == "witness.pdf"
    assert citations[0]["page_number"] == 1
    assert citations[0]["paragraph_number"] == 1


def test_citation_parser_with_section():
    """Test citation parsing with section title."""
    parser = CitationParser()
    text = '[Source: contract.pdf | Page 5, Para 3 | "Definitions"]'

    citations = parser.parse_citations(text)

    assert len(citations) == 1
    assert citations[0]["section_title"] == "Definitions"


def test_citation_parser_no_para():
    """Test citation parsing without paragraph number."""
    parser = CitationParser()
    text = '[Source: witness.pdf | Page 1]'

    citations = parser.parse_citations(text)

    assert len(citations) == 1
    assert citations[0]["page_number"] == 1
    assert citations[0]["paragraph_number"] is None


def test_extract_sentences():
    """Test sentence extraction."""
    parser = CitationParser()
    text = "First sentence. Second sentence! Third sentence?"

    sentences = parser.extract_sentences(text)

    assert len(sentences) == 3
    assert "First sentence." in sentences[0]["text"]


def test_citation_verifier_verify_all_cited(sample_chunks):
    """Test verification with all citations valid."""
    verifier = CitationVerifier(sample_chunks, confidence_threshold=0.70)

    response = 'John Smith testified [Source: witness.pdf | Page 1, Para 1]. The meeting occurred [Source: witness.pdf | Page 1, Para 2].'

    result = verifier.verify(response)

    assert result.verified is True
    assert len(result.missing) == 0
    assert len(result.citations) == 2
    assert result.all_cited is True


def test_citation_verifier_missing_citation(sample_chunks):
    """Test verification with missing citation."""
    verifier = CitationVerifier(sample_chunks, confidence_threshold=0.70)

    response = 'John Smith testified [Source: witness.pdf | Page 1, Para 1]. Unknown fact [Source: unknown.pdf | Page 99, Para 99].'

    result = verifier.verify(response)

    assert result.verified is False
    assert len(result.missing) == 1
    assert "unknown.pdf" in result.missing[0]


def test_citation_verifier_wrong_page(sample_chunks):
    """Test verification with wrong page number."""
    verifier = CitationVerifier(sample_chunks, confidence_threshold=0.70)

    response = 'John Smith testified [Source: witness.pdf | Page 999, Para 1].'

    result = verifier.verify(response)

    assert result.verified is False
    assert len(result.missing) == 1


def test_citation_verifier_uncited_sentence(sample_chunks):
    """Test verification flags uncited sentences."""
    verifier = CitationVerifier(sample_chunks, confidence_threshold=0.70)

    response = 'John Smith testified [Source: witness.pdf | Page 1, Para 1]. This is an uncited fact. Another cited fact [Source: witness.pdf | Page 1, Para 2].'

    result = verifier.verify(response, check_all_sentences=True)

    assert result.all_cited is False
    assert len(result.uncited_sentences) > 0
    assert any("uncited" in s.lower() for s in result.uncited_sentences)


def test_citation_verifier_confidence_threshold(sample_chunks):
    """Test verification respects confidence threshold."""
    # Create chunks below threshold
    low_confidence_chunks = [
        {
            "chunk_id": "chunk_low",
            "text": "Low confidence text.",
            "score": 0.50,  # Below 0.70 threshold
            "metadata": {
                "file_name": "low.pdf",
                "page_number": 1,
                "paragraph_number": 1,
            },
        }
    ]

    verifier = CitationVerifier(low_confidence_chunks, confidence_threshold=0.70)

    response = 'Some text [Source: low.pdf | Page 1, Para 1].'

    result = verifier.verify(response)

    # Citation should be missing because chunk doesn't meet threshold
    assert len(result.missing) == 1


def test_should_refuse_before_generation_no_chunks():
    """Test refusal check with no chunks."""
    verifier = CitationVerifier([], confidence_threshold=0.70)

    should_refuse, reason = verifier.should_refuse_before_generation()

    assert should_refuse is True
    assert "No chunks" in reason


def test_should_refuse_before_generation_low_confidence():
    """Test refusal check with low confidence chunks."""
    low_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Low confidence.",
            "score": 0.50,  # Below threshold
            "metadata": {"file_name": "test.pdf", "page_number": 1},
        }
    ]

    verifier = CitationVerifier(low_chunks, confidence_threshold=0.70)

    should_refuse, reason = verifier.should_refuse_before_generation()

    assert should_refuse is True
    assert "threshold" in reason.lower()


def test_should_refuse_before_generation_high_confidence(sample_chunks):
    """Test refusal check with high confidence chunks."""
    verifier = CitationVerifier(sample_chunks, confidence_threshold=0.70)

    should_refuse, reason = verifier.should_refuse_before_generation()

    assert should_refuse is False


def test_format_source_list(sample_chunks):
    """Test formatting source list."""
    verifier = CitationVerifier(sample_chunks, confidence_threshold=0.70)

    source_list = verifier.format_source_list()

    assert "witness.pdf" in source_list
    assert "contract.pdf" in source_list
    assert "Page 1" in source_list


def test_verify_citations_convenience_function(sample_chunks):
    """Test convenience function."""
    response = 'John Smith testified [Source: witness.pdf | Page 1, Para 1].'

    result = verify_citations(response, sample_chunks, confidence_threshold=0.70)

    assert result.verified is True
    assert len(result.citations) == 1


def test_acceptance_criteria_all_cited(sample_chunks):
    """Test acceptance: all citations valid."""
    response = 'John Smith testified [Source: witness.pdf | Page 1, Para 1]. The meeting occurred [Source: witness.pdf | Page 1, Para 2].'

    result = verify_citations(response, sample_chunks)

    assert result.verified is True
    assert result.all_cited is True
    assert len(result.missing) == 0

    print("✓ Acceptance criteria PASSED: all citations verified")


def test_acceptance_criteria_missing_citations(sample_chunks):
    """Test acceptance: missing citations detected."""
    response = 'John Smith testified [Source: witness.pdf | Page 1, Para 1]. Unknown fact [Source: nonexistent.pdf | Page 99, Para 99].'

    result = verify_citations(response, sample_chunks)

    assert result.verified is False
    assert len(result.missing) > 0

    print("✓ Acceptance criteria PASSED: missing citations detected")


def test_acceptance_criteria_wrong_page(sample_chunks):
    """Test acceptance: wrong page detected."""
    response = 'John Smith testified [Source: witness.pdf | Page 999, Para 1].'

    result = verify_citations(response, sample_chunks)

    assert result.verified is False
    assert len(result.missing) > 0

    print("✓ Acceptance criteria PASSED: wrong page detected")




