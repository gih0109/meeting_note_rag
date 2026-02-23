import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph


@dataclass
class Speech:
    idx: int
    start_timestamp: str
    end_timestamp: str
    text: str
    speaker: Optional[str]


@dataclass
class Chunk:
    """텍스트 chunk 단위"""

    idx: int
    text: str
    start_timestamp: str
    end_timestamp: str
    speech_start_idx: int
    speech_end_idx: int


class SummaryState(TypedDict, total=False):
    """summary graph state"""

    meeting_text: str
    meeting_date: str
    speech_list: List[Speech]
    chunk_list: List[Chunk]
    info_list: List[Dict[str, Any]]
    normalized_info: Dict[str, Any]


class SummaryInfoNodes:
    """회의록에서 의논사항, 결정사항, 액션아이템을 추출/정규화하는 노드 모음"""

    def __init__(
        self,
        extract_info_chain: Any,
        normalize_info_chain: Any,
        llm_max_worker: int = 1,
        chunk_max_chars: int = 4096,
        chunk_overlap_speech_num: int = 8,
        # postprocess_min_char: int = 400,
    ):
        self.extract_info_chain = extract_info_chain
        self.normalize_info_chain = normalize_info_chain
        self.runnable_config = RunnableConfig(max_concurrency=llm_max_worker)

        self.chunk_max_chars = chunk_max_chars
        self.chunk_overlap_speech_num = chunk_overlap_speech_num
        # self.min_chars = postprocess_min_char


    def make_chunks(self, state: SummaryState) -> Dict[str, Any]:
        """
        speech_list 기반으로 chunk 생성
        """
        speech_list = state["speech_list"]
        if len(speech_list) == 0:
            return {"chunk_list": []}

        chunk_list: List[Chunk] = []
        chunk_idx = 0
        i = 0
        while i < len(speech_list):
            cur_len = 0
            j = i
            line_list = []

            start_ts = speech_list[i].start_timestamp
            end_ts = speech_list[i].end_timestamp

            while j < len(speech_list):
                speech = speech_list[j]
                line = speech.text.strip()
                len_line = len(line) + (1 if line else 0)

                if line_list and cur_len + len_line > self.chunk_max_chars:
                    break

                line_list.append(line)
                cur_len += len_line
                end_ts = speech.end_timestamp
                j += 1

            text = "\n".join(line_list).strip()
            if text:
                chunk_list.append(
                    Chunk(
                        idx=chunk_idx,
                        text=text,
                        start_timestamp=start_ts,
                        end_timestamp=end_ts,
                        speech_start_idx=i,
                        speech_end_idx=j - 1,
                    )
                )
                chunk_idx += 1

            if j >= len(speech_list):
                break

            next_i = j - self.chunk_overlap_speech_num
            if next_i <= i:
                next_i = i + 1
            i = next_i

        return {"chunk_list": chunk_list}


    def extract_info(self, state: SummaryState) -> Dict[str, Any]:
        """각 chunk에서 의논사항/결정사항/액션아이템 추출"""
        chunk_list = state["chunk_list"]
        if len(chunk_list) == 0:
            return {"info_list": []}

        payload = []
        for chunk in chunk_list:
            payload.append({
                "meeting_date": state.get("meeting_date", ""),
                "transcript": chunk.text,
            })

        outputs = self.extract_info_chain.batch(
            payload,
            config=self.runnable_config,
            return_exceptions=True,
        )

        info_list: List[Dict[str, Any]] = []
        for out in outputs:
            if isinstance(out, Exception):
                continue
            if hasattr(out, "model_dump"):
                info_list.append(out.model_dump())
            elif isinstance(out, dict):
                info_list.append(out)

        return {"info_list": info_list}


    def normalize_info(self, state: SummaryState) -> Dict[str, Any]:
        """chunk별 추출 결과를 종합해 중복 제거/정규화"""
        info_list = state["info_list"]
        if len(info_list) == 0:
            return {
                "normalized_info": {
                    "agenda": [],
                    "key_discussion_points": [],
                    "decisions": [],
                    "action_items": [],
                }
            }

        out = self.normalize_info_chain.invoke(
            {
                "meeting_date": state.get("meeting_date", ""),
                "input_json": json.dumps(info_list, ensure_ascii=False),
            },
            config=self.runnable_config,
        )

        if hasattr(out, "model_dump"):
            return {"normalized_info": out.model_dump()}
        if isinstance(out, dict):
            return {"normalized_info": out}
        return {
            "normalized_info": {
                "agenda": [],
                "key_discussion_points": [],
                "decisions": [],
                "action_items": [],
            }
        }


def build_summary_info_graph(
    extract_info_chain: Any,
    normalize_info_chain: Any,
    **node_kwargs,
):
    nodes = SummaryInfoNodes(
        extract_info_chain=extract_info_chain,
        normalize_info_chain=normalize_info_chain,
        **node_kwargs,
    )

    g = StateGraph(SummaryState)

    g.add_node("make_chunks", nodes.make_chunks)
    g.add_node("extract_info", nodes.extract_info)
    g.add_node("normalize_info", nodes.normalize_info)

    g.set_entry_point("make_chunks")
    g.add_edge("make_chunks", "extract_info")
    g.add_edge("extract_info", "normalize_info")
    g.add_edge("normalize_info", END)

    return g.compile()
