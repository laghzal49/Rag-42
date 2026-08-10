class PromptBuilder:
    """Builds prompts for the language model."""

    def __init__(self, system_prompt: str = None):
        """
        Args:
            system_prompt: Optional instruction for the model
        """
        self.system_prompt = (
            system_prompt
            or "You are a helpful assistant that answers questions about codebases."
        )

        self.template = """{system_prompt}

Instructions:
- ONLY answer based on the context provided
- If the context doesn't have the answer, say "I don't have enough information"
- Be concise and accurate

Context:
{context}

Question: {question}

Answer:"""

    def build(self, question: str, context: str) -> str:
        """Build a prompt from question and context.

        Args:
            question: The user's question
            context: Formatted context from sources

        Returns:
            Full prompt string ready to send to model
        """

        if not context or context == "No relevant context available.":
            return f"Question: {question}\n\nAnswer: I don't have enough information."

        return f"""Answer the following question based ONLY on the context.
Be concise and direct. Do not repeat yourself.

Context:
{context}

Question: {question}

Answer:"""
