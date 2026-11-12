"""The answer track: one reader model answers from the context, and one judge model grades the answer.

Both models are reached through an OpenAI-compatible /chat/completions endpoint, at temperature 0. A run
record keeps the two model names and the SHA-256 of the prompts below, and the board compares only runs
with the same three values. The API key comes from MEMORYBENCH_API_KEY and never goes into a record.
"""

import hashlib
import json
import time
import urllib.error
import urllib.request

READER_SYSTEM = ("You answer a question about a long conversation history. Use only the memory context. "
                 "If the context does not hold the answer, say that you do not know. "
                 "Answer in one short sentence.")
READER_USER = "Memory context:\n{context}\n\nDate of the question: {date}\nQuestion: {question}\nAnswer:"
JUDGE_SYSTEM = "You grade an answer against a gold answer. Reply with one word: yes or no."
JUDGE_USER = ("Question: {question}\nGold answer: {gold}\nAnswer to grade: {answer}\n\n"
              "The answer is correct when it states the same fact as the gold answer. Extra detail is fine. "
              "Dates, numbers and names must agree. When the gold answer says that the information is not "
              "in the conversation, the answer is correct only when it says that it does not know.\n"
              "Is the answer correct? Reply yes or no.")
RETRY = (429, 500, 502, 503, 504)


def prompts_sha256():
    text = "\x00".join((READER_SYSTEM, READER_USER, JUDGE_SYSTEM, JUDGE_USER))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Client:
    def __init__(self, api_base, key=None, timeout=120, tries=4):
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.key, self.timeout, self.tries = key, timeout, tries

    def chat(self, model, system, user, max_tokens):
        """(reply text, input tokens, output tokens) of one call."""
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        for attempt in range(self.tries):
            try:
                request = urllib.request.Request(self.url, data=body, headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as r:
                    data = json.load(r)
                usage = data.get("usage") or {}
                text = data["choices"][0]["message"]["content"] or ""
                return text.strip(), int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
            except urllib.error.HTTPError as e:
                if e.code not in RETRY or attempt == self.tries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.tries - 1:
                    raise
            time.sleep(2 ** attempt)


def grade(client, reader, judge, question, context):
    """(answer, ok, [reader in, reader out, judge in, judge out]) for one question and one context."""
    answer, ri, ro = client.chat(reader, READER_SYSTEM, READER_USER.format(
        context=context, date=question.date or "unknown", question=question.text), 200)
    verdict, ji, jo = client.chat(judge, JUDGE_SYSTEM, JUDGE_USER.format(
        question=question.text, gold=question.answer, answer=answer), 5)
    return answer, verdict.strip().lower().startswith("yes"), [ri, ro, ji, jo]
