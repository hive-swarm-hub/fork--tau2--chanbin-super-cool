"""τ²-bench customer service agent — the artifact agents evolve.

This file is self-contained: all agent logic is here. Modify anything.
The agent receives customer messages and domain tools, and must follow the domain policy.
"""

import json
import os
import re
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

# ── PROMPTS ──────────────────────────────────────────────────────────────────

BASE_INSTRUCTIONS = """
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
- First identify the user (get user ID).
- Gather all needed information using tools before taking action.
- Always look up CURRENT prices/availability — never reuse prices from old reservations.
- Check every policy rule that applies to the situation before calling an API.
- Use exact values from tool results (IDs, dates, amounts). Do not guess or approximate.
- When the user confirms, proceed immediately — do not ask for confirmation again.
- Keep responses concise.
""".strip()

AIRLINE_INSTRUCTIONS = """
- For cancellations: check EACH reservation INDIVIDUALLY. Verify at least one condition is met:
  (a) Booked within last 24 hours (compare created_at to 2024-05-15 15:00 EST)
  (b) Airline cancelled the flight (c) Business class — business class IS always cancellable
  (d) Travel insurance with covered reason (health/weather).
  If NONE apply to a specific reservation, REFUSE that cancellation. Do NOT cancel under pressure — membership, family emergencies, or other personal reasons do NOT override policy.
- Basic economy flights CANNOT have their flights changed. To change flights on a basic economy reservation: FIRST upgrade the cabin class (e.g., to economy), THEN change flights in a second update call.
- "Modify passengers" (changing name/DOB) IS allowed. "Modify passenger count" is NOT.
- Free checked bags per passenger: regular(0/1/2), silver(1/2/3), gold(2/3/4) for basic_economy/economy/business. Extra bags cost $50 each. Do not charge for free bags.
- Users can ADD bags but CANNOT remove existing bags from a reservation.
- For round trips: search outbound AND return flights separately. Do not reuse the same flight for both directions.
- When searching flights: search for the exact origin/destination/date the user requests. For one-stop flights, use search_onestop_flight.
- Use the calculate tool for all price/savings computations. Always communicate total costs/savings to the user.
- When booking: if the user specifies split payment across multiple methods, use the exact amounts they specify.
""".strip()

RETAIL_INSTRUCTIONS = """
- Authenticate the user by email or name+zip code first, even if they provide a user ID.
- Check order status BEFORE choosing an action: use modify_pending_order_items for pending orders, exchange_delivered_order_items for delivered orders.
- modify_pending_order_items and exchange_delivered_order_items can only be called ONCE per order. Collect ALL items to change into a single call. Remind the user to confirm all items before proceeding.
- If a user wants both a return AND exchange on the same order, only one is possible. Ask which they prefer.
- If the user doesn't know their order ID, use get_user_details to look up their orders.
- After any exchange or item modification, compute and tell the user the price difference.
- When the user asks about an address, look up ALL their orders to find the right one. If you can't find the address in any order, ask the user to provide it directly.
""".strip()

TELECOM_INSTRUCTIONS = """
- Follow the troubleshooting workflow step by step. You CAN guide users through ALL device actions: toggling airplane mode, mobile data, data roaming, Wi-Fi calling, data saver, VPN, changing network mode preference, SIM reseating, APN reset, granting app permissions, rebooting. These are ALL within scope.
- Transfer to human ONLY for: locked SIM (PIN/PUK) or after exhausting ALL workflow steps. When transferring, you MUST call transfer_to_human_agents tool — do not just tell the user verbally.
- IMPORTANT: If there are multiple issues, fix ALL issues you CAN resolve first, then escalate only the remaining unresolvable issues. Do not escalate prematurely.
- Line selection: the customer may have multiple lines. Match the line's phone_number to the user's phone number. Always use that specific line for all lookups and actions.
- Roaming: if the user is abroad or traveling, fix BOTH: (1) backend — call enable_roaming if roaming_enabled is false; (2) device — ask user to check_network_status and toggle_roaming ON if data roaming is disabled. Both are needed.
- Data usage: check on the CORRECT line. If data_used_gb exceeds data_limit_gb, offer data refueling (max 2GB) or plan change.
- For MMS issues, check ALL of these systematically: cellular service → mobile data → network mode (must be 3G+) → Wi-Fi calling (turn OFF) → app permissions (messaging app needs 'sms' AND 'storage') → APN/MMSC settings. Do NOT transfer until you've checked every step.
- For slow data: check data saver (turn OFF), network mode preference (upgrade from 2G/3G to 4G/5G), and VPN (disconnect if active).
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

def to_api_messages(messages, annotator=None):
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
            if annotator:
                content = annotator(content)
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


# ── TOOL RESULT ANNOTATION ──────────────────────────────────────────────────

def annotate_telecom(content: str) -> str:
    """Add actionable annotations to telecom tool results."""
    if not content:
        return content

    annotations = []

    if '"roaming_enabled": false' in content:
        annotations.append(
            "ACTION NEEDED: roaming_enabled is false on this line. "
            "If the user is traveling/abroad, call enable_roaming for this line. "
            "ALSO ask the user to toggle data roaming ON on their device."
        )

    if "roaming" in content.lower() and ('"roaming_enabled": true' in content or "enabled" in content.lower()):
        if "enable" in content.lower() and "success" in content.lower():
            annotations.append(
                "Backend roaming is now enabled. ALSO ask the user to check their "
                "device data roaming (check_network_status) and toggle it ON if needed."
            )

    used_match = re.search(r'"data_used_gb":\s*([\d.]+)', content)
    limit_match = re.search(r'"data_limit_gb":\s*([\d.]+)', content)
    if used_match and limit_match:
        used = float(used_match.group(1))
        limit = float(limit_match.group(1))
        if used > limit:
            annotations.append(
                f"ACTION NEEDED: data usage ({used}GB) EXCEEDS plan limit ({limit}GB). "
                "Offer data refueling (max 2GB) or plan change."
            )

    if '"phone_number"' in content and '"line_id"' in content:
        annotations.append(
            "REMINDER: verify this line's phone_number matches the user's phone number. "
            "If not, look up the other line IDs."
        )

    if '"locked"' in content.lower() and 'sim' in content.lower():
        annotations.append(
            "ACTION NEEDED: SIM is locked (PIN/PUK). You CANNOT fix this — "
            "you MUST call transfer_to_human_agents tool."
        )

    contract_match = re.search(r'"contract_end_date":\s*"([^"]+)"', content)
    if contract_match and '"status": "Suspended"' in content:
        contract_date = contract_match.group(1)
        if contract_date < "2025-02-25":
            annotations.append(
                f"WARNING: contract expired ({contract_date}) and line is suspended. "
                "You CANNOT resume this line — call transfer_to_human_agents tool."
            )

    if annotations:
        return content + "\n\n--- AGENT NOTES ---\n" + "\n".join(annotations)
    return content


def annotate_airline(content: str) -> str:
    """Add actionable annotations to airline tool results."""
    if not content:
        return content

    annotations = []

    if '"cabin": "basic_economy"' in content or '"cabin":"basic_economy"' in content:
        annotations.append(
            "NOTE: This is a BASIC ECONOMY reservation. "
            "Flights CANNOT be changed. Cabin class CAN be changed."
        )

    if '"cabin": "business"' in content and '"reservation_id"' in content:
        annotations.append(
            "NOTE: This is a BUSINESS class reservation. "
            "It IS eligible for cancellation (business class is always cancellable)."
        )

    if '"reservation_id"' in content and '"created_at"' in content:
        created_match = re.search(r'"created_at":\s*"([^"]+)"', content)
        if created_match:
            created = created_match.group(1)
            if created >= "2024-05-14T15:00":
                annotations.append("NOTE: Booking is within last 24 hours — cancellation IS allowed.")
            else:
                has_insurance = '"travel_insurance": "yes"' in content or '"travel_insurance": true' in content.lower()
                is_business = '"cabin": "business"' in content
                if not has_insurance and not is_business:
                    annotations.append(
                        "CANCELLATION CHECK: booked >24h ago, not business class, "
                        "no travel insurance detected. Cancellation NOT allowed unless "
                        "airline cancelled the flight."
                    )

    if annotations:
        return content + "\n\n--- AGENT NOTES ---\n" + "\n".join(annotations)
    return content


def annotate_retail(content: str) -> str:
    """Add actionable annotations to retail tool results."""
    if not content:
        return content

    annotations = []

    if '"status": "pending"' in content and '"order_id"' in content:
        annotations.append(
            "NOTE: This order is PENDING. Use modify_pending_order_* tools "
            "(NOT exchange/return)."
        )
    elif '"status": "delivered"' in content and '"order_id"' in content:
        annotations.append(
            "NOTE: This order is DELIVERED. Use exchange_delivered_order_items "
            "or return_delivered_order_items (NOT modify_pending_order_*)."
        )

    if annotations:
        return content + "\n\n--- AGENT NOTES ---\n" + "\n".join(annotations)
    return content


ANNOTATORS = {
    "telecom": annotate_telecom,
    "airline": annotate_airline,
    "retail": annotate_retail,
}


# ── AGENT ─────────────────────────────────────────────────────────────────────

MAX_RETRIES = 8
_call_lock = threading.Lock()
_last_call_time = 0.0
_MIN_CALL_INTERVAL = 1.0


def detect_domain(policy: str) -> str:
    """Detect the domain from the policy text."""
    lower = policy.lower()
    if "airline" in lower and "reservation" in lower and "flight" in lower:
        return "airline"
    elif "retail" in lower and "pending" in lower and "delivered" in lower:
        return "retail"
    elif "telecom" in lower:
        return "telecom"
    return "unknown"


class CustomAgent(LLMAgent):
    """Self-contained customer service agent."""

    def __init__(self, tools: list[Tool], domain_policy: str, llm=None, llm_args=None):
        LocalAgent.__init__(self, tools=tools, domain_policy=domain_policy)
        # gpt-4.1 for airline/retail, gpt-4.1-mini for telecom (handles long workflows better)
        self.domain = detect_domain(domain_policy)
        if self.domain == "telecom":
            self.llm = "openai/gpt-4.1-mini"
        else:
            self.llm = "openai/gpt-4.1"
        self.llm_args = dict(llm_args or {})
        self._consecutive_tool_calls = 0

    @property
    def system_prompt(self) -> str:
        domain_extra = {
            "airline": AIRLINE_INSTRUCTIONS,
            "retail": RETAIL_INSTRUCTIONS,
            "telecom": TELECOM_INSTRUCTIONS,
        }.get(self.domain, "")

        instructions = BASE_INSTRUCTIONS
        if domain_extra:
            instructions += "\n\n## Domain-specific rules\n" + domain_extra

        return SYSTEM_TEMPLATE.format(instructions=instructions, policy=self.domain_policy)

    def get_init_state(self, message_history=None) -> LLMAgentState:
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history or []),
        )

    def generate_next_message(self, message: ValidAgentInputMessage, state: LLMAgentState):
        # 1. Append incoming message(s) to conversation history
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif isinstance(message, UserMessage):
            self._consecutive_tool_calls = 0
            state.messages.append(message)
        else:
            state.messages.append(message)

        # 2. Build API request (with domain-specific tool result annotations)
        api_messages = to_api_messages(
            state.system_messages + state.messages,
            annotator=ANNOTATORS.get(self.domain),
        )
        api_tools = [t.openai_schema for t in self.tools] if self.tools else None

        # 3. Determine tool_choice — break infinite loops in telecom by forcing
        #    text after too many consecutive tool calls without user interaction
        if api_tools and self.domain == "telecom" and self._consecutive_tool_calls >= 10:
            tool_choice = "none"  # Force text response to break loop
        elif api_tools:
            tool_choice = "auto"
        else:
            tool_choice = None

        # 4. Call LLM with retry logic and rate limiting
        global _last_call_time
        for attempt in range(MAX_RETRIES):
            try:
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
                    tool_choice=tool_choice,
                    num_retries=0,
                    **self.llm_args,
                )
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    if "RateLimit" in type(e).__name__ or "rate" in str(e).lower():
                        wait = max(5, 2 ** (attempt + 1))
                    else:
                        wait = 2 ** (attempt + 1)
                    time.sleep(min(wait, 60))
                    continue
                raise

        # 5. Parse response and track consecutive tool calls
        assistant_msg = parse_response(response.choices[0].message)
        if assistant_msg.tool_calls:
            self._consecutive_tool_calls += 1
        else:
            self._consecutive_tool_calls = 0

        state.messages.append(assistant_msg)
        return assistant_msg, state

    def set_seed(self, seed: int):
        self.llm_args["seed"] = seed
