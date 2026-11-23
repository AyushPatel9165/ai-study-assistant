import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agents.intent_detector import IntentDetector
from src.agents.tools.answer_tool import AnswerTool
from src.agents.tools.summary_tool import SummaryTool
from src.agents.tools.quiz_tool import QuizTool
from src.rag.pipeline import RAGPipeline
from src.rag.vector_store import FAISSVectorStore

class AgentOrchestrator:
    """
    Autonomous AI Agent Orchestrator for AI Study Assistant.
    Coordinates intent detection, automatic tool routing, safe execution,
    and structured response synthesis across:
      1. Answer Tool
      2. Summary Tool
      3. Quiz Tool
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        vector_store: FAISSVectorStore
    ):
        self.rag_pipeline = rag_pipeline
        self.vector_store = vector_store
        self.intent_detector = IntentDetector()

        # Instantiate tools
        self.answer_tool = AnswerTool(rag_pipeline=self.rag_pipeline)
        self.summary_tool = SummaryTool(vector_store=self.vector_store)
        self.quiz_tool = QuizTool(vector_store=self.vector_store)

    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Executes the autonomous agent pipeline through sequential state steps:
        [1] Input Ingestion -> [2] Intent Detection -> [3] Tool Selection
        -> [4] Tool Execution -> [5] Response Synthesis
        """
        execution_trace: List[str] = []
        execution_trace.append(f"[START] Received student prompt: '{prompt}'")

        if not prompt or not prompt.strip():
            return {
                "success": False,
                "prompt": prompt,
                "error": "Empty prompt received.",
                "execution_trace": execution_trace
            }

        # Step 1: Detect Intent
        intent_info = self.intent_detector.detect_intent(prompt)
        intent = intent_info["intent"]
        confidence = intent_info["confidence"]
        params = intent_info["parameters"]
        execution_trace.append(
            f"[INTENT_DETECTION] Detected intent '{intent}' with confidence {confidence}. Reason: {intent_info['reasoning']}"
        )

        # Step 2: Route and Execute Tool
        tool_result: Dict[str, Any] = {}
        tool_name = ""

        try:
            if intent == "QUIZ":
                tool_name = self.quiz_tool.name
                num_q = params.get("num_questions", 3)
                topic = params.get("topic")
                execution_trace.append(f"[TOOL_ROUTED] Selected {tool_name} with num_questions={num_q}, topic={topic}")
                tool_result = self.quiz_tool.execute(topic=topic, num_questions=num_q)

            elif intent == "SUMMARY":
                tool_name = self.summary_tool.name
                doc_name = params.get("document_name")
                max_pts = params.get("max_points", 5)
                execution_trace.append(f"[TOOL_ROUTED] Selected {tool_name} with document_name={doc_name}, max_points={max_pts}")
                tool_result = self.summary_tool.execute(document_name=doc_name, max_points=max_pts)

            else:  # ANSWER
                tool_name = self.answer_tool.name
                query = params.get("query", prompt)
                execution_trace.append(f"[TOOL_ROUTED] Selected {tool_name} with query='{query}'")
                tool_result = self.answer_tool.execute(query=query)

            execution_trace.append(f"[EXECUTION_SUCCESS] Tool '{tool_name}' completed execution successfully.")

        except Exception as e:
            execution_trace.append(f"[EXECUTION_ERROR] Tool '{tool_name}' execution encountered an exception: {str(e)}")
            return {
                "success": False,
                "prompt": prompt,
                "intent": intent,
                "tool_used": tool_name,
                "error": f"Tool execution failed: {str(e)}",
                "execution_trace": execution_trace
            }

        # Step 3: Format Response
        formatted_output = self._format_agent_output(intent, tool_result)
        execution_trace.append("[RESPONSE_SYNTHESIS] Formatted structured response for student.")

        return {
            "success": True,
            "prompt": prompt,
            "intent": intent,
            "confidence": confidence,
            "tool_used": tool_name,
            "extracted_parameters": params,
            "tool_result": tool_result,
            "agent_response": formatted_output,
            "execution_trace": execution_trace
        }

    def _format_agent_output(self, intent: str, tool_result: Dict[str, Any]) -> str:
        """Formats clean, human-readable student response."""
        if not tool_result.get("success", False):
            return f"Agent notification: {tool_result.get('error', 'Unable to complete request.')}"

        if intent == "QUIZ":
            questions = tool_result.get("questions", [])
            lines = [f"Generated {len(questions)} Practice Questions on '{tool_result.get('topic')}':\n"]
            for idx, q in enumerate(questions, 1):
                lines.append(f"Q{idx}. {q['question']}")
                for opt in q['options']:
                    lines.append(f"   {opt}")
                lines.append(f"   Correct Answer: Option {q['correct_option']}")
                lines.append(f"   Explanation: {q['explanation']}\n")
            return "\n".join(lines)

        elif intent == "SUMMARY":
            lines = [
                f"Executive Summary for '{tool_result.get('document_name')}':",
                tool_result.get("executive_summary", ""),
                "\nKey Takeaways:"
            ]
            for pt in tool_result.get("key_takeaways", []):
                lines.append(f"- {pt}")
            return "\n".join(lines)

        else:  # ANSWER
            answer_text = tool_result.get("answer", "")
            found = tool_result.get("found_in_notes", False)
            citations = tool_result.get("citations", [])

            lines = [answer_text]
            if found and citations:
                lines.append("\n---\n**Verified References from Study Notes:**")
                for c in citations:
                    score_pct = int(c.get("relevance_score", 0) * 100)
                    lines.append(f"• Document: **{c['document_name']}** (Page {c['page_number']}, Match: {score_pct}%)")
            return "\n".join(lines)
