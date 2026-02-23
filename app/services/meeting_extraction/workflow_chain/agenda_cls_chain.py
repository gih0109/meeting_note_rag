import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from typing import List, Optional, Literal, Dict, Any


class AgendaClsOutput(BaseModel):
    """신규 결정과 후보 결정의 관계 판정 결과."""

    is_same_agenda: bool
    relation_strength_score: Optional[float] = Field(default=None, ge=0, le=1)
    directness_score: Optional[float] = Field(default=None, ge=0, le=1)
    change_type: Literal["created", "updated", "changed", "canceled", "unchanged"]
    base_decision_id: Optional[str]
    reason: str


AGENDA_CLS_SYSTEN_PROMPT = """
You compare one new decision and one prior decision candidate.
Return JSON with:
1) is_same_agenda
2) change_type
3) relation_strength_score (0~1, only when is_same_agenda=true)
4) directness_score (0~1, only when is_same_agenda=true)
5) base_decision_id (candidate decision_id when applicable)
6) reason
"""

AGENDA_CLS_HUMAN_PROMPT = """
[new_decision]
{new_decision}

[candidate_decision]
{candidate_decision}
"""

def build_agenda_cls_chain(llm: Any):
    """후보 관계 판정을 수행하는 구조화 출력 체인을 생성"""

    prompt = ChatPromptTemplate([
        ("system", AGENDA_CLS_SYSTEN_PROMPT),
        ("human", AGENDA_CLS_HUMAN_PROMPT),
    ])

    return prompt | llm.with_structured_output(AgendaClsOutput)
