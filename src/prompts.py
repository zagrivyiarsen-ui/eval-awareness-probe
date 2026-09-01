"""build_prompt / render / load_tokenizer / user_content_span / лексикон / диффы."""
from __future__ import annotations

import difflib
import hashlib
import logging
import re
from typing import Any, Sequence

from src.conditions import PLACEHOLDER, SMOKE_UNPINNED, Condition, Config, Lexicon

log = logging.getLogger(__name__)
Message = dict[str, str]
SEP = "\n\n"
DP_REQUIRED = ("env", "prompt_id", "context", "actions")


def _words(text: str) -> list[str]:
    t = re.sub(r"red[\s\-_]?team", "redteam", text.lower())
    return re.findall(r"[a-z]+", t)


def lexicon_hits(text: str, stems: Sequence[str], allow: Sequence[str] = ()) -> list[tuple[str, str]]:
    hits = []
    for w in _words(text):
        if w in allow:
            continue
        for s in stems:
            if w.startswith(s):
                hits.append((w, s))
                break
    return hits


def condition_added_text(cond: Condition) -> str:
    t = cond.format_template.replace("{context}", "").replace("{actions}", "")
    if cond.insert_text is not None:
        t += "\n" + cond.insert_text
    return t


def instruction_sentences(cond: Condition) -> set[str]:
    """Предложения-инструкции в добавленном тексте (не заголовки/нумерация). Ревью 1.6: у S1/S2/Dstr должны совпадать.
    Эвристика: ≥3 слов, конец на .?!, без цифр (строки с цифрами — нумерация/счётчики вроде 'Item 1 of 1.').
    Это фильтр, не доказательство: prereg всё равно читает шаблоны руками."""
    out = set()
    for line in condition_added_text(cond).splitlines():
        s = re.sub(r"^[\s#\-\*\d\.\)\:]+", "", line).strip()
        if len(s.split()) >= 3 and s[-1] in ".?!" and not re.search(r"\d", s):
            out.add(s)
    return out


def check_forbidden_terms(cond: Condition, lex: Lexicon) -> list[tuple[str, str]]:
    if not cond.forbidden_eval_terms:
        return []
    added = condition_added_text(cond)
    warn = lexicon_hits(added, lex.warn_stems, lex.allow)
    if warn:
        log.warning("%s: WARN-lexicon words (not blocking; prereg line 'read by hand'): %s", cond.id, warn)
    return lexicon_hits(added, lex.stems, lex.allow)


def _actions_to_str(actions: Any) -> str:
    if isinstance(actions, str):
        return actions
    assert isinstance(actions, (list, tuple)) and actions and all(isinstance(a, str) for a in actions)
    return "\n".join(actions)


def format_body(dp: dict[str, Any], cond: Condition) -> str:
    assert PLACEHOLDER not in cond.format_template, f"{cond.id}: format_template is a placeholder"
    for m in ("{context}", "{actions}"):
        assert cond.format_template.count(m) == 1, f"{cond.id}: {m} must occur exactly once"
    context, actions = dp["context"], _actions_to_str(dp["actions"])
    body = cond.format_template.replace("{context}", context).replace("{actions}", actions)
    assert context in body and actions in body
    return body


def _with_insert(text: str | None, cond: Condition) -> str:
    assert cond.insert_text is not None and PLACEHOLDER not in cond.insert_text, f"{cond.id}: insert_text missing/placeholder"
    if text is None:
        return cond.insert_text
    if cond.insert_position == "prefix":
        return cond.insert_text + SEP + text
    assert cond.insert_position == "suffix", cond.insert_position
    return text + SEP + cond.insert_text


def build_prompt(dp: dict[str, Any], cond: Condition, lex: Lexicon) -> list[Message]:
    missing = [k for k in DP_REQUIRED if k not in dp]
    assert not missing, f"decision_point {dp.get('prompt_id')!r}: missing {missing}"
    hits = check_forbidden_terms(cond, lex)
    assert not hits, f"{cond.id}: forbidden eval terms in condition text: {hits}"
    user = format_body(dp, cond)
    system: str | None = dp.get("system")
    if cond.insert_role == "user":
        user = _with_insert(user, cond)
    elif cond.insert_role == "system":
        if system is None:
            log.info("%s/%s: no system in DP; system = insert only", dp["prompt_id"], cond.id)
        system = _with_insert(system, cond)
    for role, txt in (("system", system), ("user", user)):
        if txt is not None:
            assert txt == txt.strip(), (f"{cond.id}: {role} content has leading/trailing whitespace — chat template Qwen3.5 обрезает его, "
                                        f"user_content_span разойдётся; уберите концевой \\n из format_template/insert_text: {txt[-20:]!r}")
    msgs: list[Message] = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs


def template_sha256(tokenizer) -> str:
    return hashlib.sha256((tokenizer.chat_template or "").encode("utf-8")).hexdigest()


def load_tokenizer(cfg: Config, name_override: str | None = None):
    from transformers import AutoTokenizer
    name = name_override or cfg.tokenizer["name"]
    rev = cfg.tokenizer.get("revision")
    kw = {} if (rev is None or name_override) else {"revision": rev}
    tok = AutoTokenizer.from_pretrained(name, **kw)
    assert "enable_thinking" in (tok.chat_template or ""), f"{name}: chat_template has no enable_thinking switch"
    sha = template_sha256(tok)
    pinned = cfg.tokenizer.get("chat_template_sha256")
    if name_override or pinned == SMOKE_UNPINNED:
        log.warning("tokenizer %s loaded UNPINNED (smoke). chat_template sha256=%s", name, sha)
    elif pinned is None:
        log.warning("tokenizer.chat_template_sha256 is null — put in conditions.yaml after prereg: %s", sha)
    else:
        assert pinned == sha, f"chat_template sha256 mismatch: pinned {pinned} vs loaded {sha}"
    for t in ("<|im_start|>", "<|im_end|>"):
        assert tok.convert_tokens_to_ids(t) not in (None, tok.unk_token_id), f"tokenizer lacks {t}"
    return tok


def _thinking(cfg: Config) -> bool:
    t = cfg.generation.get("thinking")
    assert isinstance(t, bool)
    return t


def render(messages: list[Message], tokenizer, cfg: Config, add_generation_prompt: bool = True) -> list[int]:
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt,
                                        enable_thinking=_thinking(cfg))
    ids = out["input_ids"] if hasattr(out, "keys") else out
    ids = [int(i) for i in ids]
    assert ids, "render produced empty token list"
    return ids


def render_text(messages: list[Message], tokenizer, cfg: Config, add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                         enable_thinking=_thinking(cfg))


def template_has_date(tokenizer) -> list[str]:
    return sorted(set(re.findall(r"strftime|date|today|year|now\(\)", tokenizer.chat_template or "", flags=re.I)))


def chat_turns(ids: Sequence[int], tokenizer) -> list[tuple[int, int, int, str]]:
    """Все ЗАКРЫТЫЕ ходы: (start, content_start, end=<|im_end|>, role). Незакрытый генерационный заголовок не входит."""
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    turns = []
    for i, t in enumerate(ids):
        if t != im_start:
            continue
        j = next((jj for jj in range(i + 1, len(ids)) if ids[jj] == im_end), None)
        if j is None:
            continue
        k = role = None
        for kk in range(i + 2, min(i + 7, j + 1)):
            r = tokenizer.decode(ids[i + 1:kk])
            if r in ("user\n", "system\n", "assistant\n"):
                k, role = kk, r[:-1]
                break
        if k is not None:
            turns.append((i, k, j, role))
    return turns


def user_content_span(ids: Sequence[int], tokenizer, user_content: str) -> tuple[int, int]:
    """[start, end) токенов содержимого ПОСЛЕДНЕГО закрытого user-хода. Устойчиво к system и add_generation_prompt."""
    users = [t for t in chat_turns(ids, tokenizer) if t[3] == "user"]
    assert users, f"no closed user turn in prompt: {tokenizer.decode(list(ids))[:120]!r}"
    _, k, j, _ = users[-1]
    got = tokenizer.decode(ids[k:j])
    if got != user_content:
        i0 = next((i for i, (a, b) in enumerate(zip(got, user_content)) if a != b), min(len(got), len(user_content)))
        raise AssertionError(f"user span decode != content at char {i0}: decoded tail {got[max(0, i0 - 15):i0 + 25]!r} vs "
                             f"content tail {user_content[max(0, i0 - 15):i0 + 25]!r} (BPE boundary or template trims whitespace)")
    return k, j


def token_diff(base: Sequence[int], new: Sequence[int]):
    sm = difflib.SequenceMatcher(a=list(base), b=list(new), autojunk=False)
    return [op for op in sm.get_opcodes() if op[0] != "equal"]


def assert_single_insertion(base: Sequence[int], new: Sequence[int], tokenizer=None) -> tuple[int, int]:
    ops = token_diff(base, new)
    if tokenizer is not None:
        for tag, i1, i2, j1, j2 in ops:
            log.info("diff %s: base=%r new=%r", tag, tokenizer.decode(list(base[i1:i2])), tokenizer.decode(list(new[j1:j2])))
    assert len(ops) == 1 and ops[0][0] == "insert", f"expected exactly one 'insert', got {ops}"
    _, _, _, j1, j2 = ops[0]
    return j1, j2
