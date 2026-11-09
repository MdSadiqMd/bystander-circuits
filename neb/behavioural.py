"""Generation- and logit-based audit layers on the HF model (L1 behavioural, L2 safety, L3
cross-lingual, L5 hidden failures). Every metric produces per-sample scores through AuditKIT
(`ak.Sample` with `actual_output`, `model="precomputed"`, `@ak.scorer`), so `ak.compare(base, cell)`
gives per-task deltas, retention and paired-bootstrap significance for every before/after number
(brief §5). The intervention is already applied to `model` by the caller (interventions/cell.py).

Metric families (n per family from configs/base.yaml `audit`):
  facts          accuracy on edited facts (efficacy), neighbourhood (other known facts; the
                 audit passes the first 60 by key, which are all `capital` facts),
                 paraphrase, per language -> L1 + L3 + H5 propagation
  mmlu           Global-MMLU 4-way accuracy by option-letter logits, per language (parallel items)
  mgsm           exact numeric match on greedy answers (en, fr, sw available)
  ifeval         programmatic checks for 10 IFEval instruction types (subset we can verify offline)
  wikitext       per-token NLL (perplexity) on WikiText-2 test passages
  refusal        refusal rate on category harmful prompts (held out from training), other
                 categories, XSTest safe + near-miss benign, per language (L2 + L3)
  hidden         context-free selective-prediction metrics on never-edited facts: margin
                 spread, AURC, high-confidence error rate
"""

from __future__ import annotations

import difflib
import json
import math
import re
import unicodedata

import auditkit as ak
import pandas as pd
import torch
from aya_circuits.config import project_root
from aya_circuits.model import chat_wrap, generate

from neb.judge import refused

RAW = project_root() / ".cache/raw"


def norm(s: str) -> str:
    """Casefold; keep letters/digits/spaces only; drop modifier letters (ʻ) and combining marks."""
    s = unicodedata.normalize("NFKC", s).casefold()
    keep = []
    for ch in s:
        cat = unicodedata.category(ch)
        if (cat[0] in "LN" and cat != "Lm") or ch.isspace():
            keep.append(ch)
    return " ".join("".join(keep).split())


def matches(continuation: str, candidates: list[str]) -> bool:
    """Known iff some gold form (label or alias) is a prefix of / contained in the answer, the first
    words agree ("Yaren" vs "Yaren District"), or the answer head is a close transliteration variant
    (difflib ratio >= 0.85 after diacritic stripping: "Aşgabat" vs "Ashgabat")."""
    cont = norm(continuation)
    cw = cont.split()[0] if cont else ""
    for c in candidates:
        cn = norm(c)
        if not cn:
            continue
        if cont.startswith(cn) or (len(cn) > 3 and cn in cont):
            return True
        if cw and len(cw) >= 4 and cn.split()[0] == cw:
            return True
        head = " ".join(cont.split()[: len(cn.split())])
        if (
            len(cn) >= 5
            and difflib.SequenceMatcher(None, strip_marks(head), strip_marks(cn)).ratio() >= 0.85
        ):
            return True
    return False


def strip_marks(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s) if unicodedata.category(ch)[0] != "M"
    )


correct = ak.scorer(
    lambda s, o: float(matches(o, s.metadata["gold"])),
    direction=ak.Direction.MAXIMIZE,
    name="accuracy",
)
refusal = ak.scorer(
    lambda s, o: float(refused(s.input, o, s.metadata.get("lang", "en"))),
    direction=ak.Direction.MAXIMIZE,
    name="refusal",
)
numeric = ak.scorer(
    lambda s, o: float(_last_number(o) == s.metadata["gold_num"]),
    direction=ak.Direction.MAXIMIZE,
    name="accuracy",
)
passthrough = ak.scorer(
    lambda s, o: float(s.metadata["score"]), direction=ak.Direction.MAXIMIZE, name="score"
)


def _last_number(text: str):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].rstrip("."))
    except ValueError:
        return None


def run(samples: list[ak.Sample], scorer, task: str) -> ak.RunResult:
    for s in samples:
        s.task = task
    return ak.evaluate(samples, "precomputed", [scorer])


# --- facts ----------------------------------------------------------------------------------------


def fact_samples(rows: list[dict], lang: str, key: str = "text") -> list[ak.Sample]:
    return [
        ak.Sample(
            input=r[key],
            target=r["object"],
            id=f"{r['relation']}:{r['subject_qid']}:{lang}:{key}",
            metadata={"gold": [r["object"], *r.get("aliases", [])], "lang": lang},
        )
        for r in rows
    ]


def eval_facts(
    model, tok, rows: list[dict], lang: str, task: str, key: str = "text", bs: int = 16
) -> ak.RunResult:
    samples = fact_samples(rows, lang, key)
    outs = generate(model, tok, [s.input for s in samples], max_new_tokens=8, batch_size=bs)
    for s, o in zip(samples, outs, strict=True):
        s.actual_output = o
    return run(samples, correct, task)


# MMLU by option logits

MMLU_TMPL = {
    "en": "{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer with only the letter of the correct option.",
    "fr": "{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nRéponds uniquement par la lettre de la bonne "
    "option.",
    "hi": "{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nकेवल सही विकल्प का अक्षर लिखें।",
    "ar": "{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nأجب بحرف الخيار الصحيح فقط.",
    "sw": "{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nJibu kwa herufi ya chaguo sahihi pekee.",
}


def mmlu_rows(lang: str, n: int, seed: int = 0) -> pd.DataFrame:
    df = pd.read_parquet(RAW / "Global-MMLU" / lang / "test-00000-of-00001.parquet")
    return df.sort_values("sample_id").sample(n=n, random_state=seed)  # same ids across languages


@torch.no_grad()
def eval_mmlu(model, tok, cfg: dict, lang: str, n: int, bs: int = 8) -> ak.RunResult:
    df = mmlu_rows(lang, n)
    letters = [tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B", "C", "D")]
    samples, texts = [], []
    for _, r in df.iterrows():
        p = MMLU_TMPL[lang].format(
            q=r.question, a=r.option_a, b=r.option_b, c=r.option_c, d=r.option_d
        )
        texts.append(chat_wrap(tok, p, cfg["chat_format"]))
        samples.append(
            ak.Sample(
                input=p, target=r.answer, id=f"mmlu:{r.sample_id}:{lang}", metadata={"lang": lang}
            )
        )
    tok.padding_side = "left"
    for i in range(0, len(texts), bs):
        enc = tok(
            texts[i : i + bs], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        logits = model(**enc).logits[:, -1][:, letters]
        for j, k in enumerate(logits.argmax(-1).tolist()):
            samples[i + j].actual_output = "ABCD"[k]
            samples[i + j].metadata["score"] = float("ABCD"[k] == samples[i + j].target)
    return run(samples, passthrough, f"mmlu_{lang}")


# MGSM


def eval_mgsm(model, tok, cfg: dict, lang: str, n: int, bs: int = 8) -> ak.RunResult:
    df = pd.read_parquet(RAW / "mgsm" / lang / "test-00000-of-00001.parquet").head(n)
    suffix = {
        "en": "\nSolve step by step and end with the final number.",
        "fr": "\nRésous étape par étape et termine par le nombre final.",
        "sw": "\nTatua hatua kwa hatua na umalize kwa namba ya mwisho.",
    }[lang]
    samples = [
        ak.Sample(
            input=r.question + suffix,
            target=str(r.answer_number),
            id=f"mgsm:{i}:{lang}",
            metadata={"gold_num": float(r.answer_number), "lang": lang},
        )
        for i, r in df.iterrows()
    ]
    outs = generate(
        model,
        tok,
        [chat_wrap(tok, s.input, cfg["chat_format"]) for s in samples],
        max_new_tokens=160,
        batch_size=bs,
    )
    for s, o in zip(samples, outs, strict=True):
        s.actual_output = o
    return run(samples, numeric, f"mgsm_{lang}")


# IFEval subset


def _check(inst: str, kw: dict, out: str) -> bool | None:
    words = re.findall(r"\w+", out)
    if inst == "punctuation:no_comma":
        return "," not in out
    if inst == "change_case:english_lowercase":
        return out == out.lower()
    if inst == "change_case:english_capital":
        return out == out.upper()
    if inst == "length_constraints:number_words":
        n, rel = kw["num_words"], kw["relation"]
        return len(words) >= n if rel == "at least" else len(words) < n
    if inst == "detectable_format:number_bullet_lists":
        return len(re.findall(r"^\s*[\*\-]\s", out, flags=re.MULTILINE)) == kw["num_bullets"]
    if inst == "startend:quotation":
        o = out.strip()
        return len(o) > 1 and o[0] == '"' and o[-1] == '"'
    if inst == "detectable_format:json_format":
        try:
            json.loads(out.strip().strip("`").removeprefix("json").strip())
            return True
        except Exception:
            return False
    if inst == "keywords:existence":
        return all(k.lower() in out.lower() for k in kw["keywords"])
    if inst == "keywords:forbidden_words":
        return not any(k.lower() in out.lower() for k in kw["forbidden_words"])
    if inst == "detectable_format:title":
        return re.search(r"<<[^>]+>>", out) is not None
    if inst == "detectable_format:number_highlighted_sections":
        return len(re.findall(r"\*[^\*\n]+\*", out)) >= kw["num_highlights"]
    if inst == "startend:end_checker":
        return out.strip().endswith(kw["end_phrase"].strip())
    if inst == "detectable_content:postscript":
        return kw["postscript_marker"].lower() in out.lower()
    return None


SUPPORTED = {
    "punctuation:no_comma",
    "change_case:english_lowercase",
    "change_case:english_capital",
    "length_constraints:number_words",
    "detectable_format:number_bullet_lists",
    "startend:quotation",
    "detectable_format:json_format",
    "keywords:existence",
    "keywords:forbidden_words",
    "detectable_format:title",
    "detectable_format:number_highlighted_sections",
    "startend:end_checker",
    "detectable_content:postscript",
}


def ifeval_rows(n: int) -> list[dict]:
    rows = [json.loads(line) for line in (RAW / "IFEval/ifeval_input_data.jsonl").open()]
    rows = [r for r in rows if set(r["instruction_id_list"]) <= SUPPORTED]
    return rows[:n]


def eval_ifeval(model, tok, cfg: dict, n: int, bs: int = 8) -> ak.RunResult:
    rows = ifeval_rows(n)
    samples = [
        ak.Sample(
            input=r["prompt"],
            id=f"ifeval:{r['key']}",
            metadata={"inst": r["instruction_id_list"], "kw": r["kwargs"]},
        )
        for r in rows
    ]
    outs = generate(
        model,
        tok,
        [chat_wrap(tok, s.input, cfg["chat_format"]) for s in samples],
        max_new_tokens=192,
        batch_size=bs,
    )
    for s, o in zip(samples, outs, strict=True):
        s.actual_output = o
        checks = [
            _check(i, k or {}, o) for i, k in zip(s.metadata["inst"], s.metadata["kw"], strict=True)
        ]
        s.metadata["score"] = float(all(c for c in checks))  # strict prompt-level accuracy
    return run(samples, passthrough, "ifeval")


# WikiText perplexity


@torch.no_grad()
def eval_wikitext(model, tok, n: int, seq_len: int = 256, bs: int = 4) -> ak.RunResult:
    df = pd.read_parquet(RAW / "wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet")
    text = "\n".join(t for t in df.text.tolist() if t.strip())
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0][
        : n * seq_len
    ].view(n, seq_len)
    samples = []
    for i in range(0, n, bs):
        batch = ids[i : i + bs].to(model.device)
        logits = model(input_ids=batch).logits[:, :-1].float()
        nll = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), batch[:, 1:], reduction="none"
        ).mean(1)
        for j, v in enumerate(nll.tolist()):
            samples.append(
                ak.Sample(
                    input=f"wikitext chunk {i + j}",
                    id=f"wt:{i + j}",
                    metadata={"score": -v},
                    actual_output="",
                )
            )
    r = run(samples, passthrough, "wikitext_negnll")
    r.metadata["perplexity"] = math.exp(-r.headline["score"])
    return r


# refusal


def eval_refusal(
    model, tok, cfg: dict, prompts: list[dict], lang: str, task: str, bs: int = 16
) -> ak.RunResult:
    """prompts: dicts with the text under `lang` and an `id`."""
    samples = [
        ak.Sample(input=p[lang], id=f"{p['id']}:{lang}", metadata={"lang": lang})
        for p in prompts
        if lang in p
    ]
    outs = generate(
        model,
        tok,
        [chat_wrap(tok, s.input, cfg["chat_format"]) for s in samples],
        max_new_tokens=cfg["audit"]["max_new_tokens"],
        batch_size=bs,
    )
    for s, o in zip(samples, outs, strict=True):
        s.actual_output = o
    return run(samples, refusal, task)


# hidden failures (L5)


@torch.no_grad()
def eval_hidden(model, tok, rows: list[dict], bs: int = 16) -> dict:
    """Never-edited facts, no context: gold-vs-competitor margin spread, AURC over confidence,
    high-confidence error. Not Anand's arbitration (memory vs an injected passage; that probe is
    circuit-steering/audit/arbitration.py)."""
    tok.padding_side = "left"
    margins, confs, wrong = [], [], []
    for i in range(0, len(rows), bs):
        batch = rows[i : i + bs]
        enc = tok(
            [r["text"] for r in batch], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        logits = model(**enc).logits[:, -1].float()
        probs = logits.softmax(-1)
        for j, r in enumerate(batch):
            g = r["label"]
            top2 = torch.topk(logits[j], 2)
            competitor = top2.values[1] if top2.indices[0] == g else top2.values[0]
            margins.append(float(logits[j, g] - competitor))
            confs.append(float(probs[j].max()))
            wrong.append(int(probs[j].argmax()) != g)
    order = sorted(range(len(rows)), key=lambda k: -confs[k])
    risks, errs = [], 0
    for c, k in enumerate(order, 1):
        errs += wrong[k]
        risks.append(errs / c)
    aurc = sum(risks) / len(risks)
    hi = [w for w, c in zip(wrong, confs, strict=True) if c >= 0.5]
    return {
        "margin_mean": sum(margins) / len(margins),
        "margin_std": (sum((m - sum(margins) / len(margins)) ** 2 for m in margins) / len(margins))
        ** 0.5,
        "aurc": aurc,
        "top1_error": sum(wrong) / len(wrong),
        "high_conf_error": (sum(hi) / len(hi)) if hi else None,
        "n": len(rows),
    }
