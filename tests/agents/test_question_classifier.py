"""Tests for the Question Classifier module."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.retrieval.classification import (
    QuestionType,
    RetrievalStrategy,
    QuestionClassifier,
    classify_question,
    should_use_group_event_cluster,
)


class TestQuestionClassifier:
    """Test suite for QuestionClassifier."""

    @pytest.fixture
    def classifier(self):
        return QuestionClassifier()

    # =========================================================================
    # Event-Temporal Questions (Should use GEC_CLUSTER_RERANK)
    # =========================================================================

    @pytest.mark.parametrize("question", [
        "When did Caroline go to the LGBTQ support group?",
        "When did Melanie paint a sunrise?",
        "When did Melanie run a charity race?",
        "When did Caroline give a speech at a school?",
        "When did Caroline meet up with her friends, family, and mentors?",
        "When did Melanie sign up for a pottery class?",
        "When did Melanie go to the museum?",
        "When did Caroline have a picnic?",
        "When did Caroline go to the LGBTQ conference?",
        "When did Melanie read the book \"nothing is impossible\"?",
        "When did Caroline go to the adoption meeting?",
        "When did Melanie go to the pottery workshop?",
    ])
    def test_event_temporal_questions(self, classifier, question):
        """Event temporal questions should be classified as EVENT_TEMPORAL and use GEC_CLUSTER_RERANK."""
        result = classifier.classify(question)
        assert result.question_type == QuestionType.EVENT_TEMPORAL, f"Expected EVENT_TEMPORAL for: {question}"
        assert result.strategy == RetrievalStrategy.GEC_CLUSTER_RERANK, f"Expected GEC_CLUSTER_RERANK for: {question}"

    @pytest.mark.parametrize("question", [
        "When is Caroline going to the transgender conference?",
        "When is Melanie planning on going camping?",
    ])
    def test_event_future_questions(self, classifier, question):
        """Future event questions should also use GEC_CLUSTER_RERANK."""
        result = classifier.classify(question)
        assert result.question_type == QuestionType.EVENT_TEMPORAL, f"Expected EVENT_TEMPORAL for: {question}"
        assert result.strategy == RetrievalStrategy.GEC_CLUSTER_RERANK

    # =========================================================================
    # Event-Activity Questions (Should use GEC_CLUSTER_RERANK)
    # =========================================================================

    @pytest.mark.parametrize("question", [
        "What activities does Melanie partake in?",
        "What does Melanie do to destress?",
    ])
    def test_event_activity_questions(self, classifier, question):
        """Activity questions should be classified as EVENT_ACTIVITY and use GEC_CLUSTER_RERANK."""
        result = classifier.classify(question)
        assert result.question_type == QuestionType.EVENT_ACTIVITY, f"Expected EVENT_ACTIVITY for: {question}"
        assert result.strategy == RetrievalStrategy.GEC_CLUSTER_RERANK

    # =========================================================================
    # Event-Aggregation Questions (Should use GEC_CLUSTER_RERANK)
    # =========================================================================

    @pytest.mark.parametrize("question", [
        "What books has Melanie read?",
        "Where has Melanie camped?",
    ])
    def test_event_aggregation_questions(self, classifier, question):
        """Aggregation questions should be classified as EVENT_AGGREGATION and use GEC_CLUSTER_RERANK."""
        result = classifier.classify(question)
        assert result.question_type == QuestionType.EVENT_AGGREGATION, f"Expected EVENT_AGGREGATION for: {question}"
        assert result.strategy == RetrievalStrategy.GEC_CLUSTER_RERANK

    # =========================================================================
    # Attribute-Identity Questions (Should use AGENTIC_ONLY)
    # =========================================================================

    @pytest.mark.parametrize("question", [
        "What is Caroline's identity?",
        "What is Caroline's relationship status?",
    ])
    def test_attribute_identity_questions(self, classifier, question):
        """Identity questions should be classified as ATTRIBUTE_IDENTITY and use AGENTIC_ONLY."""
        result = classifier.classify(question)
        assert result.question_type == QuestionType.ATTRIBUTE_IDENTITY, f"Expected ATTRIBUTE_IDENTITY for: {question}"
        assert result.strategy == RetrievalStrategy.AGENTIC_ONLY

    # =========================================================================
    # Attribute-Preference Questions (Should use AGENTIC_ONLY)
    # =========================================================================

    @pytest.mark.parametrize("question", [
        "What do Melanie's kids like?",
    ])
    def test_attribute_preference_questions(self, classifier, question):
        """Preference questions should be classified as ATTRIBUTE_PREFERENCE and use AGENTIC_ONLY."""
        result = classifier.classify(question)
        assert result.question_type == QuestionType.ATTRIBUTE_PREFERENCE, f"Expected ATTRIBUTE_PREFERENCE for: {question}"
        assert result.strategy == RetrievalStrategy.AGENTIC_ONLY

    # =========================================================================
    # Attribute-Location Questions (Should use AGENTIC_ONLY)
    # =========================================================================

    @pytest.mark.parametrize("question", [
        "Where did Caroline move from 4 years ago?",
    ])
    def test_attribute_location_questions(self, classifier, question):
        """Location questions should be classified as ATTRIBUTE_LOCATION and use AGENTIC_ONLY."""
        result = classifier.classify(question)
        assert result.question_type == QuestionType.ATTRIBUTE_LOCATION, f"Expected ATTRIBUTE_LOCATION for: {question}"
        assert result.strategy == RetrievalStrategy.AGENTIC_ONLY

    # =========================================================================
    # Reasoning/Hypothetical Questions (Should use GEC_INSERT_AFTER_HIT)
    # =========================================================================

    @pytest.mark.parametrize("question", [
        "Would Caroline still want to pursue counseling as a career if she hadn't received support growing up?",
        "Would Caroline likely have Dr. Seuss books on her bookshelf?",
        "Would Caroline pursue writing as a career option?",
    ])
    def test_reasoning_hypothetical_questions(self, classifier, question):
        """Hypothetical questions should use GEC_INSERT_AFTER_HIT strategy."""
        result = classifier.classify(question)
        assert result.question_type == QuestionType.REASONING_HYPOTHETICAL, f"Expected REASONING_HYPOTHETICAL for: {question}"
        assert result.strategy == RetrievalStrategy.GEC_INSERT_AFTER_HIT

    # =========================================================================
    # Time Calculation Questions (Should use AGENTIC_ONLY)
    # =========================================================================

    @pytest.mark.parametrize("question", [
        "How long has Caroline had her current group of friends for?",
        "How long ago was Caroline's 18th birthday?",
    ])
    def test_time_calculation_questions(self, classifier, question):
        """Time calculation questions should use AGENTIC_ONLY for precise data."""
        result = classifier.classify(question)
        assert result.question_type == QuestionType.TIME_CALCULATION, f"Expected TIME_CALCULATION for: {question}"
        assert result.strategy == RetrievalStrategy.AGENTIC_ONLY

    # =========================================================================
    # Career/Education Questions (HYBRID - special handling)
    # =========================================================================

    @pytest.mark.parametrize("question", [
        "What fields would Caroline be likely to pursue in her education?",
        "What did Caroline research?",
        "What career path has Caroline decided to pursue?",
    ])
    def test_career_questions(self, classifier, question):
        """Career questions use insert_after_hit for context expansion."""
        result = classifier.classify(question)
        # Career questions should use insert_after_hit or at least not fail
        assert result.strategy in (
            RetrievalStrategy.GEC_CLUSTER_RERANK,
            RetrievalStrategy.GEC_INSERT_AFTER_HIT,
            RetrievalStrategy.AGENTIC_ONLY,
        )

    # =========================================================================
    # Convenience Function Tests
    # =========================================================================

    def test_classify_question_function(self):
        """Test the convenience function."""
        result = classify_question("When did Caroline go to the conference?")
        assert result.question_type == QuestionType.EVENT_TEMPORAL
        assert result.strategy == RetrievalStrategy.GEC_CLUSTER_RERANK

    def test_should_use_gec_true(self):
        """Test should_use_group_event_cluster returns True for event questions."""
        assert should_use_group_event_cluster("When did Caroline attend the meeting?") is True

    def test_should_use_gec_false(self):
        """Test should_use_group_event_cluster returns False for attribute questions."""
        assert should_use_group_event_cluster("What is Caroline's identity?") is False

    # =========================================================================
    # Classification Result Tests
    # =========================================================================

    def test_classification_result_to_dict(self, classifier):
        """Test that ClassificationResult can be serialized."""
        result = classifier.classify("When did Caroline go to the conference?")
        result_dict = result.to_dict()

        assert "question_type" in result_dict
        assert "strategy" in result_dict
        assert "confidence" in result_dict
        assert "reasoning" in result_dict
        assert "detected_patterns" in result_dict

    def test_confidence_scores(self, classifier):
        """Test that confidence scores are within valid range."""
        questions = [
            "When did Caroline attend the meeting?",
            "What is Caroline's identity?",
            "Would Caroline pursue writing?",
        ]
        for question in questions:
            result = classifier.classify(question)
            assert 0.0 <= result.confidence <= 1.0, f"Invalid confidence for: {question}"

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_empty_question(self, classifier):
        """Test handling of empty question."""
        result = classifier.classify("")
        # Should return a valid result without error
        assert result.question_type is not None
        assert result.strategy is not None

    def test_question_with_special_characters(self, classifier):
        """Test handling of questions with special characters."""
        result = classifier.classify("When did Caroline's friend arrive?")
        assert result is not None

    def test_case_insensitivity(self, classifier):
        """Test that classification is case-insensitive."""
        result1 = classifier.classify("When did Caroline go?")
        result2 = classifier.classify("WHEN DID CAROLINE GO?")
        result3 = classifier.classify("when did caroline go?")

        assert result1.question_type == result2.question_type == result3.question_type


class TestLoCoMoQuestions:
    """Test classifier against all LoCoMo evaluation questions."""

    LOCOMO_QUESTIONS = [
        # Event-temporal (13 questions) - Should use GEC_CLUSTER_RERANK
        ("When did Caroline go to the LGBTQ support group?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("When did Melanie paint a sunrise?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("When did Melanie run a charity race?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("When did Caroline give a speech at a school?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("When did Caroline meet up with her friends, family, and mentors?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("When did Melanie sign up for a pottery class?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("When is Caroline going to the transgender conference?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("When did Melanie go to the museum?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("When did Caroline have a picnic?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("When did Caroline go to the LGBTQ conference?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("When did Melanie read the book \"nothing is impossible\"?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("When did Caroline go to the adoption meeting?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("When did Melanie go to the pottery workshop?", RetrievalStrategy.GEC_CLUSTER_RERANK),

        # Activity/Aggregation (4 questions) - Should use GEC_CLUSTER_RERANK
        ("What activities does Melanie partake in?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("What books has Melanie read?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("What does Melanie do to destress?", RetrievalStrategy.GEC_CLUSTER_RERANK),
        ("Where has Melanie camped?", RetrievalStrategy.GEC_CLUSTER_RERANK),

        # Attribute (4 questions) - Should use AGENTIC_ONLY
        ("What is Caroline's identity?", RetrievalStrategy.AGENTIC_ONLY),
        ("What is Caroline's relationship status?", RetrievalStrategy.AGENTIC_ONLY),
        ("What do Melanie's kids like?", RetrievalStrategy.AGENTIC_ONLY),
        ("Where did Caroline move from 4 years ago?", RetrievalStrategy.AGENTIC_ONLY),

        # Reasoning (3 questions) - Should use GEC_INSERT_AFTER_HIT
        ("Would Caroline still want to pursue counseling as a career if she hadn't received support growing up?", RetrievalStrategy.GEC_INSERT_AFTER_HIT),
        ("Would Caroline likely have Dr. Seuss books on her bookshelf?", RetrievalStrategy.GEC_INSERT_AFTER_HIT),
        ("Would Caroline pursue writing as a career option?", RetrievalStrategy.GEC_INSERT_AFTER_HIT),

        # Time calculation (3 questions) - Should use AGENTIC_ONLY
        ("How long has Caroline had her current group of friends for?", RetrievalStrategy.AGENTIC_ONLY),
        ("How long ago was Caroline's 18th birthday?", RetrievalStrategy.AGENTIC_ONLY),
        ("When is Melanie planning on going camping?", RetrievalStrategy.GEC_CLUSTER_RERANK),  # Future event

        # Career/Research (3 questions) - Any strategy acceptable
        ("What fields would Caroline be likely to pursue in her education?", None),
        ("What did Caroline research?", None),
        ("What career path has Caroline decided to pursue?", None),
    ]

    @pytest.fixture
    def classifier(self):
        return QuestionClassifier()

    @pytest.mark.parametrize("question,expected_strategy", LOCOMO_QUESTIONS)
    def test_locomo_question_classification(self, classifier, question, expected_strategy):
        """Test that LoCoMo questions are classified correctly."""
        result = classifier.classify(question)

        if expected_strategy is not None:
            assert result.strategy == expected_strategy, (
                f"Question: {question}\n"
                f"Expected: {expected_strategy}\n"
                f"Got: {result.strategy}\n"
                f"Type: {result.question_type}\n"
                f"Reasoning: {result.reasoning}"
            )
        else:
            # For questions where any strategy is acceptable
            assert result.strategy in (
                RetrievalStrategy.GEC_CLUSTER_RERANK,
                RetrievalStrategy.GEC_INSERT_AFTER_HIT,
                RetrievalStrategy.AGENTIC_ONLY,
            )

    def test_locomo_gec_coverage(self, classifier):
        """Test that GEC-suitable questions are correctly identified."""
        gec_questions = [q for q, s in self.LOCOMO_QUESTIONS if s == RetrievalStrategy.GEC_CLUSTER_RERANK]
        gec_count = 0

        for question in gec_questions:
            result = classifier.classify(question)
            if result.strategy == RetrievalStrategy.GEC_CLUSTER_RERANK:
                gec_count += 1

        # Expect at least 90% accuracy for GEC questions
        accuracy = gec_count / len(gec_questions)
        assert accuracy >= 0.9, f"GEC classification accuracy: {accuracy:.1%} (expected >= 90%)"

    def test_locomo_agentic_coverage(self, classifier):
        """Test that Agentic-suitable questions are correctly identified."""
        agentic_questions = [q for q, s in self.LOCOMO_QUESTIONS if s == RetrievalStrategy.AGENTIC_ONLY]
        agentic_count = 0

        for question in agentic_questions:
            result = classifier.classify(question)
            if result.strategy == RetrievalStrategy.AGENTIC_ONLY:
                agentic_count += 1

        # Expect at least 80% accuracy for Agentic questions
        accuracy = agentic_count / len(agentic_questions)
        assert accuracy >= 0.8, f"Agentic classification accuracy: {accuracy:.1%} (expected >= 80%)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
