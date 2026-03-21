import os
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are GitLab Handbook AI — a precise, helpful assistant that answers questions about GitLab's handbook, values, processes, and product direction.

RULES:
1. Answer ONLY using the provided context passages. If the context does not contain enough information, say so clearly.
2. Be direct and concise. Use markdown bold for key terms.
3. Never fabricate facts, numbers, names, or policies.
4. If multiple passages are relevant, synthesise them into a coherent answer.
5. End with a one-sentence summary if the answer is long.
6. Do not mention that you are using "context passages" in your response — just answer naturally.
"""


class LLMClient:
    MODEL = "llama-3.1-8b-instant"   # fast + free on Groq

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Add it to your .env file. Get a free key at https://console.groq.com"
            )
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
        except ImportError as exc:
            raise RuntimeError("groq not installed. Run: pip install groq") from exc

    def answer(self, question: str, context_passages: list[str]) -> str:
        passages_block = "\n\n---\n\n".join(
            f"[Passage {i+1}]\n{p}" for i, p in enumerate(context_passages)
        )
        user_message = (
            f"CONTEXT PASSAGES:\n{passages_block}\n\n"
            f"QUESTION: {question}\n\n"
            "Please answer the question based on the context passages above."
        )

        logger.info("Calling Groq (%s) …", self.MODEL)
        try:
            response = self._client.chat.completions.create(
                model=self.MODEL,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise