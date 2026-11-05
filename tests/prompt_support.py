"""离线替身；实际LangMem优化器保持真实，仅替换外部聊天模型与规划响应。"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

IMPROVED = "已有一次阴性观察且仍缺可靠阳性证据时，优先request_review；不得把阴性观察变成药材结论。"


class OfflineOptimizerModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "offline-langmem-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="test"))])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return RunnableLambda(lambda messages: AIMessage(content="", tool_calls=[{
            "name": "OptimizedPromptOutput", "args": {"analysis": "模拟分析：减少重复无增益观察",
            "improved_prompt": IMPROVED}, "id": "offline-optimization"}]))


class OfflinePlanner:
    def __init__(self, instruction):
        self.instruction = instruction
        self.contexts = []

    def choose(self, context):
        self.contexts.append(context)
        decision = "review" if "已有一次阴性观察" in self.instruction and context["previous_observations"] else context["actions"][0]["action_id"]
        return {"action_id": decision, "decision_summary": "显式模拟规划"}, None
