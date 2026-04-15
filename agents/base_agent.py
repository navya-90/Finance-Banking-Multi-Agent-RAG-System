from abc import ABC
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from state import BankingState
from config import PRIMARY_LLM, FALLBACK_LLM, make_llm
from tools.banking_tools import AGENT_TOOLS

MAX_TOOL_ROUNDS = 6


class BaseAgent(ABC):
    domain: str
    system_prompt: str

    def __init__(self):
        tools = AGENT_TOOLS[self.domain]
        self._primary_llm  = make_llm(PRIMARY_LLM,  temperature=0.2, tools=tools)
        self._fallback_llm = make_llm(FALLBACK_LLM, temperature=0.2, tools=tools)
        self._tool_node    = ToolNode(tools)

    def _build_system(self, state: BankingState) -> str:
        lines = [self.system_prompt]
        if state.customer_id:
            lines.append(f"Customer ID in context: {state.customer_id}")
        if state.account_id:
            lines.append(f"Account ID in context: {state.account_id}")
        return "\n\n".join(lines)

    def __call__(self, state: BankingState) -> BankingState:
        messages = [SystemMessage(content=self._build_system(state))] + list(state.messages)
        use_fallback = False
        rounds = 0

        while rounds < MAX_TOOL_ROUNDS:
            llm = self._fallback_llm if use_fallback else self._primary_llm
            try:
                response: AIMessage = llm.invoke(messages)
            except Exception:
                if not use_fallback:
                    use_fallback = True
                    continue
                raise
            messages.append(response)

            if not response.tool_calls:
                break

            tool_msgs = self._tool_node.invoke({"messages": messages})["messages"]
            messages.extend(tool_msgs)
            rounds += 1

        final = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls),
            None,
        )
        answer = final.content if final else "I was unable to process your request."
        hedged = any(p in answer.lower() for p in ["i'm not sure", "i don't know", "cannot determine"])
        confidence = 0.35 if hedged else 0.9

        return state.model_copy(update={
            "agent_response": answer,
            "routed_to": self.domain,
            "confidence": confidence,
            "escalate": confidence < 0.4,
            "escalation_reason": "Agent uncertain" if hedged else None,
            "messages": list(state.messages) + [AIMessage(content=answer)],
        })
