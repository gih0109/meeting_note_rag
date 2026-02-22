import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict


_TS_RE = re.compile(r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*$")


@dataclass
class Speech:
    idx: int
    start_timestamp: str
    end_timestamp: str
    text: str
    speaker: Optional[str]


def parse_speech_from_meeting_text(meeting_text: str) -> List[Speech]:
    """
    테스트용 파싱 함수
    """
    lines = meeting_text.replace("\r\n", "\n").split("\n")

    start_ts_list = []
    text_list = []

    cur_timestamp = None
    temp_text_list = []

    def _flush():
        """
        현재 모아둔 speech 를 하나로 만들어 list 에 넣는 내부함수
        """
        nonlocal cur_timestamp, temp_text_list

        if cur_timestamp is not None:
            text = "\n".join(t.strip() for t in temp_text_list if t.strip()).strip()
            if text:
                start_ts_list.append(cur_timestamp)
                text_list.append(text)
        cur_timestamp, temp_text_list = None, []

    for line in lines:
        m = _TS_RE.match(line)
        if m:
            _flush()
            cur_timestamp = m.group(1)
            continue

        if cur_timestamp is not None and line.strip():
            temp_text_list.append(line)

    _flush()

    speech_list = []
    for i, (start_ts, text) in enumerate(zip(start_ts_list, text_list)):
        end_ts = start_ts_list[i+1] if i+1 < len(start_ts_list) else start_ts
        speech_list.append(
            Speech(
                idx=i,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                text=text,
                speaker=None,
            )
        )

    return speech_list
