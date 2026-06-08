from typing import Dict, TypedDict

from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langgraph.graph import END, StateGraph

from backend.settings import get_settings
from backend.agents import RelevanceChecker, ResearchAgent, VerificationAgent
from backend.log_config import logger


class AgentState(TypedDict):
    question: str
    documents: list[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: EnsembleRetriever
    research_attempts: int


class AgentWorkflow:
    def __init__(self) -> None:
        """Initialize the multi-step research and verification workflow."""
        self.settings = get_settings()
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        self.compiled_workflow = self.build_workflow()

    def build_workflow(self):
        """Compile the LangGraph workflow used to answer a question."""
        workflow = StateGraph(AgentState)
        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)

        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {"relevant": "research", "irrelevant": END},
        )
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {"re_research": "research", "end": END},
        )
        return workflow.compile()

    def _check_relevance_step(self, state: AgentState) -> Dict:
        """Decide whether the retrieved documents can answer the question."""
        classification = self.relevance_checker.check(
            question=state["question"],
            retriever=state["retriever"],
            k=20,
        )

        if classification in {"CAN_ANSWER", "PARTIAL"}:
            return {"is_relevant": True}

        return {
            "is_relevant": False,
            "draft_answer": (
                "This question is not covered by the uploaded documents. "
                "Please ask another question related to the uploaded material."
            ),
            "verification_report": "Relevant: NO",
        }

    @staticmethod
    def _decide_after_relevance_check(state: AgentState) -> str:
        """Route the workflow based on the relevance decision."""
        return "relevant" if state["is_relevant"] else "irrelevant"

    def full_pipeline(self, question: str, retriever: EnsembleRetriever) -> Dict[str, str]:
        """Run retrieval, research, and verification for a user question."""
        try:
            documents = retriever.invoke(question)
            logger.info("Retrieved {} relevant documents", len(documents))

            initial_state = AgentState(
                question=question,
                documents=documents,
                draft_answer="",
                verification_report="",
                is_relevant=False,
                retriever=retriever,
                research_attempts=0,
            )
            final_state = self.compiled_workflow.invoke(initial_state)
            return {
                "draft_answer": final_state["draft_answer"],
                "verification_report": final_state["verification_report"],
            }
        except Exception as exc:
            logger.exception("Workflow execution failed: {}", exc)
            raise

    def _research_step(self, state: AgentState) -> Dict:
        """Generate a draft answer from the retrieved document set."""
        result = self.researcher.generate(state["question"], state["documents"])
        return {
            "draft_answer": result["draft_answer"],
            "research_attempts": state["research_attempts"] + 1,
        }

    def _verification_step(self, state: AgentState) -> Dict:
        """Validate the draft answer against the retrieved documents."""
        result = self.verifier.check(state["draft_answer"], state["documents"])
        return {"verification_report": result["verification_report"]}

    def _decide_next_step(self, state: AgentState) -> str:
        """Choose whether to retry research or finish the workflow."""
        verification_report = state["verification_report"]
        needs_retry = (
            "Supported: NO" in verification_report
            or "Relevant: NO" in verification_report
        )

        if needs_retry and state["research_attempts"] < self.settings.max_research_iterations:
            logger.info(
                "Verification requested another research pass (attempt {}/{})",
                state["research_attempts"],
                self.settings.max_research_iterations,
            )
            return "re_research"

        return "end"
