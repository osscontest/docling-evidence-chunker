# token_counter.py
import tiktoken
from interfaces import EvidenceUnit

ENCODING = tiktoken.get_encoding("cl100k_base")
DEFAULT_TOKEN_LIMIT = 512

def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

def count_eu_tokens(eu: EvidenceUnit) -> int:
    return count_tokens(eu.text)

def exceeds_token_limit(eu: EvidenceUnit, limit: int = DEFAULT_TOKEN_LIMIT) -> bool:
    return count_eu_tokens(eu) > limit

def get_token_status(eu: EvidenceUnit, limit: int = DEFAULT_TOKEN_LIMIT) -> dict:
    token_count = count_eu_tokens(eu)
    return {
        "eu_id": eu.eu_id,
        "token_count": token_count,
        "limit": limit,
        "exceeds_limit": token_count > limit,
        "usage": f"{token_count/limit:.1%}",
    }

if __name__ == "__main__":
    eu_small = EvidenceUnit(
        eu_id="eu-p1-0",
        page_no=1,
        caption_text="Table 1: Runtime characteristics of Docling",
        table_html="<table><tr><td>CPU</td><td>375s</td></tr></table>",
        footnote_text="OCR is disabled.",
    )

    eu_large = EvidenceUnit(
        eu_id="eu-p1-1",
        page_no=1,
        caption_text="Table 2: Very large table",
        table_html="<table><tr><td>" + "데이터 " * 300 + "</td></tr></table>",
        context_before=["이 표는 매우 긴 설명을 가지고 있습니다. " * 20],
    )

    assert not exceeds_token_limit(eu_small)
    assert exceeds_token_limit(eu_large)
    print("assert 통과 ✅")

    for eu in [eu_small, eu_large]:
        status = get_token_status(eu)
        flag = "🔴 초과" if status["exceeds_limit"] else "🟢 정상"
        print(f"{flag} | {status['eu_id']} | {status['token_count']} 토큰 / 한도 {status['limit']} | usage: {status['usage']}")