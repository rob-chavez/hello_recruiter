import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, InternalServerError

from system_prompt import prompt
from tools import tools as TOOL_SCHEMAS  # the JSON tool schemas for the model
from tools import *  # tool functions

# ---------- setup ----------
load_dotenv(".env", override=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("me-bot")

# Build an explicit registry of callable tools by name
TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    schema["function"]["name"]: globals().get(schema["function"]["name"])
    for schema in TOOL_SCHEMAS
}
# drop any tools that aren't actually defined
TOOL_REGISTRY = {k: v for k, v in TOOL_REGISTRY.items() if callable(v)}

# ---------- helper utilities ----------
def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s) if s else {}
    except json.JSONDecodeError as e:
        # Return a structured error payload—model can decide what to do.
        return {"_error": f"Invalid JSON arguments: {e}", "_raw": s}

def _call_tool(tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Any]:
    fn = TOOL_REGISTRY.get(tool_name)
    if not fn:
        return False, {"_error": f"Unknown tool: '{tool_name}'"}
    try:
        return True, fn(**{k: v for k, v in arguments.items() if k != "_error"})
    except Exception as e:
        logger.exception("Tool '%s' raised an exception", tool_name)
        return False, {"_error": f"Tool '{tool_name}' failed: {e.__class__.__name__}: {e}"}

def _retry_chat_create(
    client: OpenAI,
    **kwargs
):
    """Simple exponential backoff for transient OpenAI errors."""
    delays = [0.5, 1.0, 2.0, 4.0]
    last_exc = None
    for d in delays:
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, InternalServerError) as e:
            last_exc = e
            logger.warning("Transient OpenAI error: %s. Retrying in %.1fs", e, d)
            time.sleep(d)
        except APIError as e:
            # Non-transient API error; don't retry blindly
            raise
    # final attempt without catching so the caller can see the error
    return client.chat.completions.create(**kwargs)

# ---------- main class ----------
class Me:
    def __init__(self, model: str = "gpt-4o-mini", max_tool_turns: int = 8):
        self.client = OpenAI()
        self.model = model
        self.system_prompt = prompt
        self.max_tool_turns = max_tool_turns

    def _handle_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        logger.info("Tool calls received: %s", [tc.function.name for tc in (tool_calls or [])])
        results: List[Dict[str, Any]] = []

        for tc in tool_calls or []:
            tool_name = tc.function.name
            args = _safe_json_loads(tc.function.arguments)
            logger.info("→ Tool called: %s args=%s", tool_name, args)

            ok, result = _call_tool(tool_name, args)
            # content must be a string; pass JSON back to the model
            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result if ok else result)  # preserve error payloads
            })
        return results

    def chat(
        self,
        message: str,
        history: Optional[Sequence[Dict[str, str]]] = None,
        *,
        temperature: float = 0.2
    ) -> str:
        """
        Runs a chat turn with tool-use support.
        `history` should be a list of messages like [{"role":"user"|"assistant","content": "..."}].
        Returns the assistant's final text content.
        """
        history = list(history or [])
        messages: List[Dict[str, str]] = (
            [{"role": "system", "content": self.system_prompt}]
            + history
            + [{"role": "user", "content": message}]
        )

        tool_turns = 0

        while True:
            response = _retry_chat_create(
                self.client,
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=temperature,
            )

            choice = response.choices[0]
            finish = choice.finish_reason
            assistant_msg = choice.message

            # Append the assistant's message either way so the transcript is complete
            messages.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                **({"tool_calls": assistant_msg.tool_calls} if assistant_msg.tool_calls else {})
            })

            if finish == "tool_calls":
                tool_turns += 1
                if tool_turns > self.max_tool_turns:
                    logger.warning("Max tool turns (%d) exceeded; stopping.", self.max_tool_turns)
                    break

                tool_calls = assistant_msg.tool_calls or []
                tool_results = self._handle_tool_calls(tool_calls)
                messages.extend(tool_results)
                # Continue the loop so the model can read tool results
                continue

            # Done: finish_reason is "stop", "length", etc.
            break

        # Return the last assistant content (may be empty if tools-only turn)
        # Find the last assistant message with non-empty content, if possible
        for msg in reversed(messages):
            if msg["role"] == "assistant" and (msg.get("content") or "").strip():
                return msg["content"]  # type: ignore[return-value]
        return ""  # Fallback if no textual content
