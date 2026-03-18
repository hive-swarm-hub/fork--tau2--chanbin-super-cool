"""τ²-bench customer service agent — the artifact agents evolve.

This file is self-contained: all agent logic is here. Modify anything.
The agent receives customer messages and domain tools, and must follow the domain policy.
"""

import json
import os
import threading
import time

from litellm import completion

from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# ── PROMPT (the main lever for improving performance) ─────────────────────────

INSTRUCTIONS = """
You are a customer service agent. You MUST follow the <policy> exactly. The policy is your sole source of truth — never invent rules, procedures, or information not in the policy or provided by the user.

## Critical rules
1. Each turn: EITHER send a message to the user OR make a tool call. NEVER both at the same time.
2. Only make ONE tool call per turn.
3. Before any action that modifies the database (booking, modifying, cancelling), you MUST:
   a. Verify all policy preconditions are met (eligibility, rules, restrictions).
   b. List the exact action details to the user and get explicit confirmation.
   c. Only then make the tool call.
4. The APIs do NOT enforce policy rules — YOU must check them before calling.
5. If a request is against policy, deny it and explain why.
6. Transfer to a human agent ONLY if the request cannot be handled within the scope of your actions. To transfer: first call transfer_to_human_agents, then send "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
7. Do not proactively offer compensation unless the user explicitly asks.

## Key practices
- First identify the user (get user ID). If the user provides something like "firstname_lastname_XXXX", treat that as a user ID and look it up directly.
- Gather all needed information using tools before taking action. Be proactive — use tools to look up information rather than asking the user for details you can retrieve.
- Always look up CURRENT prices/availability — never reuse prices from old reservations.
- Check every policy rule that applies to the situation before calling an API.
- Use exact values from tool results (IDs, dates, amounts). Do not guess or approximate.
- When the user confirms, proceed immediately — do not ask for confirmation again.
- Be action-oriented: once you have the necessary information and user confirmation, execute ALL required changes (flights, passengers, baggage, payment, etc.) — don't stop partway through.
- For technical support: follow the troubleshooting workflow step by step, checking each condition before moving to the next.
- Keep responses concise.

## Tool result verification
After receiving a tool result, carefully verify it against the policy:
- Compare EACH field in the result against what the policy requires. Look for what is MISSING, not just what is present.
- If the policy says a condition must be met, confirm the tool result explicitly shows it is met.
- Do not assume "no news is good news" — if a required field or status is absent from the result, investigate further.
- Match line by line: if the policy lists specific requirements, check them one by one against the actual data returned.

## Technical support
- Follow troubleshooting workflows step by step. Check each condition before moving to the next.
- Run ALL required diagnostics before concluding. Do not skip steps even if early results look normal.
- When checking permissions, settings, or configurations: verify EVERY required item is present. If the policy requires items A, B, and C, confirm all three — not just two.
- IMPORTANT: If there are multiple issues, fix ALL issues you CAN resolve first, then escalate only the remaining unresolvable issues. Do not escalate prematurely — complete all fixable steps before transferring.
- After each fix, re-run diagnostics to verify the fix worked and check for remaining issues.
""".strip()

SYSTEM_TEMPLATE = """
<instructions>
{instructions}
</instructions>
<policy>
{policy}
</policy>
""".strip()

# ── MESSAGE CONVERSION ────────────────────────────────────────────────────────

def to_api_messages(messages):
    """Convert tau2 message objects to OpenAI-style dicts."""
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AssistantMessage):
            d = {"role": "assistant", "content": m.content or ""}
            if m.is_tool_call():
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(d)
        elif isinstance(m, ToolMessage):
            content = m.content if m.content else ""
            out.append({"role": "tool", "content": content, "tool_call_id": m.id})
    return out


def parse_response(choice):
    """Convert an LLM API response choice into a tau2 AssistantMessage."""
    tool_calls = None
    if choice.tool_calls:
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in choice.tool_calls
        ]
    return AssistantMessage(
        role="assistant",
        content=choice.content or "",
        tool_calls=tool_calls or None,
    )


# ── AGENT ─────────────────────────────────────────────────────────────────────

MAX_RETRIES = 8
# Global rate limiter: ensure minimum spacing between API calls
_call_lock = threading.Lock()
_last_call_time = 0.0
_MIN_CALL_INTERVAL = 1.0  # seconds between calls (across all threads)

class CustomAgent(LLMAgent):
    """Self-contained customer service agent."""

    def __init__(self, tools: list[Tool], domain_policy: str, llm=None, llm_args=None):
        LocalAgent.__init__(self, tools=tools, domain_policy=domain_policy)
        self.llm = "openai/gpt-4.1"
        self.llm_args = dict(llm_args or {})

    @property
    def system_prompt(self) -> str:
        return SYSTEM_TEMPLATE.format(instructions=INSTRUCTIONS, policy=self.domain_policy)

    def get_init_state(self, message_history=None) -> LLMAgentState:
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history or []),
        )

    def generate_next_message(self, message: ValidAgentInputMessage, state: LLMAgentState):
        # 1. Append incoming message(s) to conversation history
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # 2. Build API request
        api_messages = to_api_messages(state.system_messages + state.messages)
        api_tools = [t.openai_schema for t in self.tools] if self.tools else None

        # 3. Call LLM with retry logic (handles rate limits)
        global _last_call_time
        for attempt in range(MAX_RETRIES):
            try:
                # Throttle: enforce minimum interval between calls
                with _call_lock:
                    now = time.time()
                    elapsed = now - _last_call_time
                    if elapsed < _MIN_CALL_INTERVAL:
                        time.sleep(_MIN_CALL_INTERVAL - elapsed)
                    _last_call_time = time.time()

                response = completion(
                    model=self.llm,
                    messages=api_messages,
                    tools=api_tools,
                    tool_choice="auto" if api_tools else None,
                    num_retries=0,  # disable litellm's internal retries
                    **self.llm_args,
                )
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    # For rate limits, wait longer
                    if "RateLimit" in type(e).__name__ or "rate" in str(e).lower():
                        wait = max(5, 2 ** (attempt + 1))
                    else:
                        wait = 2 ** (attempt + 1)
                    wait = min(wait, 60)
                    time.sleep(wait)
                    continue
                raise

        # 4. Parse response
        assistant_msg = parse_response(response.choices[0].message)
        state.messages.append(assistant_msg)
        return assistant_msg, state

    def set_seed(self, seed: int):
        self.llm_args["seed"] = seed
