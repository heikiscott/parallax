"""Question Classifier for routing queries to appropriate retrieval strategies.

This module classifies questions to determine whether they should be handled by:
1. GEC_CLUSTER_RERANK: LLM selects clusters + Agentic fallback (for event-type questions)
2. GEC_INSERT_AFTER_HIT: Original retrieval + cluster expansion (for reasoning/general questions)
3. AGENTIC_ONLY: Pure Agentic retrieval (for attribute/preference questions)

Design Philosophy:
- Event-type questions benefit from cluster-based retrieval (temporal context, event grouping)
- Attribute/preference questions benefit from precise keyword matching (Agentic retrieval)
- Reasoning/general questions use insert_after_hit for context expansion without LLM cluster selection
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Classification of question types for retrieval routing."""

    # Best handled by Group Event Cluster
    EVENT_TEMPORAL = "event_temporal"        # "When did X happen?"
    EVENT_ACTIVITY = "event_activity"        # "What activities does X do?"
    EVENT_AGGREGATION = "event_aggregation"  # "What books has X read?" (multiple events)

    # Best handled by original Agentic retrieval
    ATTRIBUTE_IDENTITY = "attribute_identity"      # "What is X's identity/status?"
    ATTRIBUTE_PREFERENCE = "attribute_preference"  # "What does X like?"
    ATTRIBUTE_LOCATION = "attribute_location"      # "Where is X from?"

    # Requires reasoning or hypotheticals - may benefit from either
    REASONING_HYPOTHETICAL = "reasoning_hypothetical"  # "Would X do Y if...?"
    REASONING_INFERENCE = "reasoning_inference"        # Questions requiring inference

    # Time calculation - special handling
    TIME_CALCULATION = "time_calculation"  # "How long ago...?" "How long has...?"

    # Counting/listing - requires precise enumeration
    COUNTING = "counting"  # "How many...?" "List all..."

    # Default/unknown
    GENERAL = "general"


class RetrievalStrategy(Enum):
    """Retrieval strategy to use for a question.

    Strategies:
    - GEC_CLUSTER_RERANK: LLM selects relevant clusters + Agentic fallback
    - GEC_INSERT_AFTER_HIT: Original retrieval + cluster member expansion
    - AGENTIC_ONLY: Pure Agentic retrieval, no cluster expansion
    """
    GEC_CLUSTER_RERANK = "gec_cluster_rerank"      # LLM cluster selection + Agentic fallback
    GEC_INSERT_AFTER_HIT = "gec_insert_after_hit"  # Original + cluster expansion
    AGENTIC_ONLY = "agentic_only"                  # Pure Agentic, no GEC


@dataclass
class ClassificationResult:
    """Result of question classification."""
    question_type: QuestionType
    strategy: RetrievalStrategy
    confidence: float  # 0.0 - 1.0
    reasoning: str

    # Additional metadata
    detected_patterns: List[str] = None  # Patterns that triggered classification
    entities: List[str] = None           # Detected entities (names, etc.)

    def __post_init__(self):
        if self.detected_patterns is None:
            self.detected_patterns = []
        if self.entities is None:
            self.entities = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_type": self.question_type.value,
            "strategy": self.strategy.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "detected_patterns": self.detected_patterns,
            "entities": self.entities,
        }


class QuestionClassifier:
    """Rule-based question classifier for retrieval routing.

    Uses pattern matching to classify questions into types and determine
    the best retrieval strategy. This is a lightweight, fast classifier
    that can be used without LLM calls.

    For more complex classification, an LLM-based classifier can be used
    as an alternative.
    """

    # =============================================================================
    # Pattern Definitions
    # =============================================================================

    # Event-temporal patterns (When + action verb)
    EVENT_TEMPORAL_PATTERNS = [
        r"^when\s+did\s+\w+\s+(go|attend|visit|have|give|meet|run|sign|paint|read|write|make|do|start|finish|complete)\b",
        r"^when\s+is\s+\w+\s+(going|planning|scheduled)\b",
        r"^when\s+will\s+\w+\s+(go|attend|visit|have)\b",
        r"^what\s+time\s+did\s+\w+",
        r"^on\s+what\s+date\s+did\s+\w+",
    ]

    # Event-activity patterns (What does X do?)
    EVENT_ACTIVITY_PATTERNS = [
        r"^what\s+activities?\s+does?\s+\w+\s+(do|partake|engage|participate)",
        r"^what\s+does?\s+\w+\s+do\s+to\s+(relax|destress|unwind)",
        r"^what\s+hobbies?\s+does?\s+\w+\s+have",
        r"^how\s+does?\s+\w+\s+spend\s+(time|days?|weekends?)",
    ]

    # Event-aggregation patterns (Multiple events/items - experiences/activities)
    EVENT_AGGREGATION_PATTERNS = [
        r"^what\s+(experiences?|activities)\s+(does?|has|did)\s+\w+\s+(have|do|partake)",
        r"^what\s+kinds?\s+of\s+\w+\s+(does?|has|did)\s+\w+",
    ]

    # Counting/listing patterns (Precise enumeration - NOT suitable for clustering)
    COUNTING_PATTERNS = [
        r"^how\s+many\s+(times?|books?|events?|people?|places?|days?)\s+",
        r"^list\s+(all\s+)?(the\s+)?(books?|events?|places?|activities?)",
        r"^what\s+are\s+all\s+(the\s+)?(books?|events?|places?)",
        r"^what\s+books?\s+has\s+\w+\s+read",
        r"^what\s+places?\s+has\s+\w+\s+(visited|been|camped|traveled)",
        r"^where\s+has\s+\w+\s+(been|visited|camped|traveled)",
        r"^what\s+events?\s+has\s+\w+\s+(attended|participated)",
    ]

    # Attribute-identity patterns
    ATTRIBUTE_IDENTITY_PATTERNS = [
        r"^what\s+is\s+\w+['\u2019]?s?\s+(identity|gender|orientation|status)",
        r"^who\s+is\s+\w+",
        r"^what\s+is\s+\w+['\u2019]?s?\s+(relationship|marital)\s+status",
        r"^is\s+\w+\s+(married|single|dating|engaged)",
        r"^what\s+is\s+\w+['\u2019]?s?\s+(job|occupation|profession)",
    ]

    # Attribute-preference patterns
    ATTRIBUTE_PREFERENCE_PATTERNS = [
        r"^what\s+does?\s+\w+\s+(like|prefer|enjoy|love|hate)",
        r"^what\s+is\s+\w+['\u2019]?s?\s+favorite",
        r"^what\s+do\s+\w+['\u2019]?s?\s+(kids?|children|family)\s+like",
        r"^what\s+kind\s+of\s+\w+\s+does?\s+\w+\s+(like|prefer)",
    ]

    # Attribute-location patterns
    ATTRIBUTE_LOCATION_PATTERNS = [
        r"^where\s+(is|does)\s+\w+\s+(from|live|work)",
        r"^where\s+did\s+\w+\s+(move|relocate|come)\s+from",
        r"^what\s+(city|country|state|town)\s+(is|does)\s+\w+",
        r"^where\s+is\s+\w+['\u2019]?s?\s+(hometown|home|birthplace)",
    ]

    # Reasoning/hypothetical patterns
    REASONING_PATTERNS = [
        r"^would\s+\w+\s+(likely|still|ever|probably|pursue|want|choose|prefer)",
        r"^if\s+\w+\s+(had|didn['\u2019]t|hadn['\u2019]t|were|wasn['\u2019]t)",
        r"^could\s+\w+\s+(have|be)\s+",
        r"^might\s+\w+\s+(have|be)\s+",
        r"^do\s+you\s+think\s+\w+\s+would",
    ]

    # Time calculation patterns
    TIME_CALCULATION_PATTERNS = [
        r"^how\s+long\s+(ago|has)\s+",
        r"^how\s+many\s+(years?|months?|days?|weeks?)\s+(ago|since|has)",
        r"^for\s+how\s+long\s+has\s+\w+",
        r"^\w+['\u2019]?s?\s+\w+\s+birthday",  # X's Nth birthday
    ]

    # Career/education patterns (can be either GEC or Agentic depending on context)
    CAREER_PATTERNS = [
        r"^what\s+(career|field|job|profession)\s+(path|would|has|did)\s+\w+",
        r"^what\s+fields?\s+(would|will|did)\s+\w+\s+(pursue|study|choose)",
        r"^what\s+did\s+\w+\s+(research|study|major)",
    ]

    def __init__(self):
        """Initialize the classifier with compiled regex patterns."""
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        self._patterns = {
            # Check COUNTING first - more specific patterns should be checked before general EVENT_AGGREGATION
            QuestionType.COUNTING: [re.compile(p, re.IGNORECASE) for p in self.COUNTING_PATTERNS],
            QuestionType.EVENT_TEMPORAL: [re.compile(p, re.IGNORECASE) for p in self.EVENT_TEMPORAL_PATTERNS],
            QuestionType.EVENT_ACTIVITY: [re.compile(p, re.IGNORECASE) for p in self.EVENT_ACTIVITY_PATTERNS],
            QuestionType.EVENT_AGGREGATION: [re.compile(p, re.IGNORECASE) for p in self.EVENT_AGGREGATION_PATTERNS],
            QuestionType.ATTRIBUTE_IDENTITY: [re.compile(p, re.IGNORECASE) for p in self.ATTRIBUTE_IDENTITY_PATTERNS],
            QuestionType.ATTRIBUTE_PREFERENCE: [re.compile(p, re.IGNORECASE) for p in self.ATTRIBUTE_PREFERENCE_PATTERNS],
            QuestionType.ATTRIBUTE_LOCATION: [re.compile(p, re.IGNORECASE) for p in self.ATTRIBUTE_LOCATION_PATTERNS],
            QuestionType.REASONING_HYPOTHETICAL: [re.compile(p, re.IGNORECASE) for p in self.REASONING_PATTERNS],
            QuestionType.TIME_CALCULATION: [re.compile(p, re.IGNORECASE) for p in self.TIME_CALCULATION_PATTERNS],
        }
        self._career_patterns = [re.compile(p, re.IGNORECASE) for p in self.CAREER_PATTERNS]

    def classify(self, question: str) -> ClassificationResult:
        """Classify a question and determine the best retrieval strategy.

        Args:
            question: The user's question/query

        Returns:
            ClassificationResult with question type, strategy, and metadata
        """
        question = question.strip()
        detected_patterns = []

        # Check each pattern type
        for question_type, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(question):
                    detected_patterns.append(f"{question_type.value}:{pattern.pattern}")
                    return self._create_result(question_type, detected_patterns, question)

        # Check career patterns (special handling - could be either strategy)
        for pattern in self._career_patterns:
            if pattern.search(question):
                detected_patterns.append(f"career:{pattern.pattern}")
                # Career questions benefit from cluster context expansion
                return ClassificationResult(
                    question_type=QuestionType.GENERAL,
                    strategy=RetrievalStrategy.GEC_INSERT_AFTER_HIT,
                    confidence=0.6,
                    reasoning="Career/education question - using insert_after_hit for context expansion",
                    detected_patterns=detected_patterns,
                )

        # Default: use INSERT_AFTER_HIT for unknown patterns (safe fallback with context)
        return ClassificationResult(
            question_type=QuestionType.GENERAL,
            strategy=RetrievalStrategy.GEC_INSERT_AFTER_HIT,
            confidence=0.3,
            reasoning="No specific pattern matched - using insert_after_hit for safe context expansion",
            detected_patterns=detected_patterns,
        )

    def _create_result(
        self,
        question_type: QuestionType,
        detected_patterns: List[str],
        question: str,
    ) -> ClassificationResult:
        """Create a classification result with appropriate strategy."""

        # Map question types to strategies
        type_to_strategy = {
            # EVENT_TEMPORAL -> Pure Agentic (avoid time confusion from multi-timepoint clusters)
            # Based on error analysis: 69% of GEC errors involve temporal confusion
            QuestionType.EVENT_TEMPORAL: RetrievalStrategy.AGENTIC_ONLY,

            # EVENT_ACTIVITY -> Keep GEC cluster_rerank (activity aggregation benefits from clustering)
            QuestionType.EVENT_ACTIVITY: RetrievalStrategy.GEC_CLUSTER_RERANK,

            # EVENT_AGGREGATION -> Downgrade to INSERT_AFTER_HIT (context expansion only)
            QuestionType.EVENT_AGGREGATION: RetrievalStrategy.GEC_INSERT_AFTER_HIT,

            # COUNTING -> Pure Agentic (precise enumeration, clusters introduce noise)
            # Based on error analysis: 23% of GEC errors are counting/multi-hop errors
            QuestionType.COUNTING: RetrievalStrategy.AGENTIC_ONLY,

            # Attribute types -> Pure Agentic (no cluster expansion)
            QuestionType.ATTRIBUTE_IDENTITY: RetrievalStrategy.AGENTIC_ONLY,
            QuestionType.ATTRIBUTE_PREFERENCE: RetrievalStrategy.AGENTIC_ONLY,
            QuestionType.ATTRIBUTE_LOCATION: RetrievalStrategy.AGENTIC_ONLY,

            # Reasoning -> INSERT_AFTER_HIT (context expansion without LLM cluster selection)
            QuestionType.REASONING_HYPOTHETICAL: RetrievalStrategy.GEC_INSERT_AFTER_HIT,
            QuestionType.REASONING_INFERENCE: RetrievalStrategy.GEC_INSERT_AFTER_HIT,

            # Time calculation -> Pure Agentic (precise data needed)
            QuestionType.TIME_CALCULATION: RetrievalStrategy.AGENTIC_ONLY,

            # Default -> INSERT_AFTER_HIT (safe fallback with context)
            QuestionType.GENERAL: RetrievalStrategy.GEC_INSERT_AFTER_HIT,
        }

        # Confidence based on pattern match strength
        confidence_map = {
            QuestionType.COUNTING: 0.95,           # Very clear pattern (how many, list all)
            QuestionType.EVENT_TEMPORAL: 0.9,      # Very clear pattern
            QuestionType.EVENT_ACTIVITY: 0.85,
            QuestionType.EVENT_AGGREGATION: 0.8,   # Lower confidence - narrower patterns now
            QuestionType.ATTRIBUTE_IDENTITY: 0.9,
            QuestionType.ATTRIBUTE_PREFERENCE: 0.8,
            QuestionType.ATTRIBUTE_LOCATION: 0.85,
            QuestionType.REASONING_HYPOTHETICAL: 0.75,
            QuestionType.TIME_CALCULATION: 0.8,
            QuestionType.GENERAL: 0.5,
        }

        # Reasoning for each type
        reasoning_map = {
            QuestionType.COUNTING: "Counting/listing question - Agentic search for precise enumeration",
            QuestionType.EVENT_TEMPORAL: "Temporal event question - Agentic search to avoid time confusion",
            QuestionType.EVENT_ACTIVITY: "Activity question - GEC cluster_rerank for related activities",
            QuestionType.EVENT_AGGREGATION: "Aggregation question - insert_after_hit for context expansion",
            QuestionType.ATTRIBUTE_IDENTITY: "Identity/attribute question - precise Agentic search preferred",
            QuestionType.ATTRIBUTE_PREFERENCE: "Preference question - Agentic search for specific attributes",
            QuestionType.ATTRIBUTE_LOCATION: "Location question - Agentic search for specific facts",
            QuestionType.REASONING_HYPOTHETICAL: "Hypothetical question - insert_after_hit for context expansion",
            QuestionType.TIME_CALCULATION: "Time calculation - needs precise temporal data",
            QuestionType.GENERAL: "General question - insert_after_hit for safe context expansion",
        }

        return ClassificationResult(
            question_type=question_type,
            strategy=type_to_strategy.get(question_type, RetrievalStrategy.GEC_INSERT_AFTER_HIT),
            confidence=confidence_map.get(question_type, 0.5),
            reasoning=reasoning_map.get(question_type, "Unknown pattern"),
            detected_patterns=detected_patterns,
        )


class LLMQuestionClassifier:
    """LLM-based question classifier for more nuanced classification.

    Uses an LLM to classify questions when rule-based classification
    is uncertain or for complex questions.
    """

    CLASSIFICATION_PROMPT = """You are a question classifier for a memory retrieval system.

## Task
Classify the following question to determine the best retrieval strategy.

## Question
{question}

## Question Categories

### 1. EVENT_TEMPORAL (Use: Group Event Cluster)
Questions about when specific events happened.
Examples:
- "When did Caroline go to the LGBTQ support group?"
- "When did Melanie paint a sunrise?"
- "When is Caroline going to the conference?"

### 2. EVENT_ACTIVITY (Use: Group Event Cluster)
Questions about activities someone does/did.
Examples:
- "What activities does Melanie partake in?"
- "What does Melanie do to destress?"

### 3. EVENT_AGGREGATION (Use: Group Event Cluster)
Questions requiring aggregation of multiple events/items.
Examples:
- "What books has Melanie read?"
- "Where has Melanie camped?"

### 4. ATTRIBUTE_IDENTITY (Use: Agentic Retrieval)
Questions about fixed attributes, identity, or status.
Examples:
- "What is Caroline's identity?"
- "What is Caroline's relationship status?"

### 5. ATTRIBUTE_PREFERENCE (Use: Agentic Retrieval)
Questions about preferences or likes/dislikes.
Examples:
- "What do Melanie's kids like?"
- "What is Caroline's favorite book?"

### 6. ATTRIBUTE_LOCATION (Use: Agentic Retrieval)
Questions about locations, origins, or places.
Examples:
- "Where did Caroline move from?"
- "Where does Melanie live?"

### 7. REASONING_HYPOTHETICAL (Use: Insert After Hit)
Hypothetical or conditional questions requiring inference.
Examples:
- "Would Caroline pursue writing as a career option?"
- "Would Caroline likely have Dr. Seuss books?"

### 8. TIME_CALCULATION (Use: Agentic Retrieval)
Questions requiring time calculations or duration.
Examples:
- "How long has Caroline had her friends?"
- "How long ago was Caroline's birthday?"

## Response Format (JSON)
{{"category": "EVENT_TEMPORAL", "strategy": "gec_cluster_rerank", "confidence": 0.9, "reasoning": "Brief explanation"}}

Valid strategies: "gec_cluster_rerank", "gec_insert_after_hit", "agentic_only"
Respond with JSON only."""

    def __init__(self, llm_provider=None, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with LLM provider.

        Args:
            llm_provider: LLM provider instance (e.g., OpenAI client)
            llm_config: Configuration for LLM calls (model, temperature, etc.)
        """
        self.llm_provider = llm_provider
        self.llm_config = llm_config or {}

    async def classify(self, question: str) -> ClassificationResult:
        """Classify a question using LLM.

        Args:
            question: The user's question/query

        Returns:
            ClassificationResult with question type, strategy, and metadata
        """
        if self.llm_provider is None:
            # Fallback to rule-based classifier
            logger.warning("LLM provider not configured, falling back to rule-based classifier")
            return QuestionClassifier().classify(question)

        prompt = self.CLASSIFICATION_PROMPT.format(question=question)

        try:
            response = await self._call_llm(prompt)
            return self._parse_llm_response(response, question)
        except Exception as e:
            logger.error(f"LLM classification failed: {e}, falling back to rule-based")
            return QuestionClassifier().classify(question)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the classification prompt."""
        # Implementation depends on the LLM provider interface
        # This is a placeholder - actual implementation would use the provider's API
        raise NotImplementedError("LLM provider interface not implemented")

    def _parse_llm_response(self, response: str, question: str) -> ClassificationResult:
        """Parse LLM response into ClassificationResult."""
        import json

        try:
            data = json.loads(response)

            # Map category to QuestionType
            category_map = {
                "EVENT_TEMPORAL": QuestionType.EVENT_TEMPORAL,
                "EVENT_ACTIVITY": QuestionType.EVENT_ACTIVITY,
                "EVENT_AGGREGATION": QuestionType.EVENT_AGGREGATION,
                "ATTRIBUTE_IDENTITY": QuestionType.ATTRIBUTE_IDENTITY,
                "ATTRIBUTE_PREFERENCE": QuestionType.ATTRIBUTE_PREFERENCE,
                "ATTRIBUTE_LOCATION": QuestionType.ATTRIBUTE_LOCATION,
                "REASONING_HYPOTHETICAL": QuestionType.REASONING_HYPOTHETICAL,
                "TIME_CALCULATION": QuestionType.TIME_CALCULATION,
            }

            # Map strategy string to enum
            strategy_map = {
                "gec_cluster_rerank": RetrievalStrategy.GEC_CLUSTER_RERANK,
                "gec_insert_after_hit": RetrievalStrategy.GEC_INSERT_AFTER_HIT,
                "agentic_only": RetrievalStrategy.AGENTIC_ONLY,
            }

            return ClassificationResult(
                question_type=category_map.get(data.get("category", ""), QuestionType.GENERAL),
                strategy=strategy_map.get(data.get("strategy", ""), RetrievalStrategy.GEC_INSERT_AFTER_HIT),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "LLM classification"),
                detected_patterns=["llm_classification"],
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return QuestionClassifier().classify(question)


def classify_question(question: str) -> ClassificationResult:
    """Convenience function for quick classification using rule-based classifier.

    Args:
        question: The question to classify

    Returns:
        ClassificationResult
    """
    classifier = QuestionClassifier()
    return classifier.classify(question)


def should_use_group_event_cluster(question: str) -> bool:
    """Quick check if a question should use Group Event Cluster retrieval.

    Args:
        question: The question to check

    Returns:
        True if GEC should be used (cluster_rerank or insert_after_hit),
        False for pure Agentic only
    """
    result = classify_question(question)
    return result.strategy in (
        RetrievalStrategy.GEC_CLUSTER_RERANK,
        RetrievalStrategy.GEC_INSERT_AFTER_HIT,
    )
