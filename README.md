# Operator RAG Assistant

**An assistant that answers questions from operating documents and shows the passage each answer came from. It refuses when the documents don't cover the question, or when someone asks for live operating advice.**

![Two recorded examples: a cited answer drawn from two documents, and a refused request to bypass an interlock](docs/examples.png)

## Why I built it

I'm Nathan Ramos. I ran chemical process units for more than ten years at NOVA Chemicals, CF Industries, INEOS Styrolution and GreenMantra, and I hold a TSSA 3rd Class Operating Engineer certificate. On shift, nobody acts on an answer they can't trace back to a procedure, and the worst tool is one that makes something up with confidence. So this assistant is built around two rules:

1. Every answer cites its source, and if the model cites a passage it wasn't given, the tool refuses rather than answering.
2. It refuses live operating requests ("which valve should I open right now", "how do I bypass the interlock") and questions the documents don't cover.

## How well it works

I tested it against a baseline with no AI in it, which just returns the best-matching passage, so the table shows what the language model actually adds.

| Question set | Questions | Cited the right source | Refused correctly | Answer had the key facts* | Baseline, no AI |
|---|---:|---:|---:|---:|---:|
| Easy | 20 | 93.3% | 100% (5 of 5) | 100% | 100% |
| Hard | 12 | 100% | n/a | 41.7% | 75% |

\*This counts whether the expected keywords appear in the answer. It doesn't judge whether the answer is right.

The first question set turned out to be too easy, because the baseline already found the right document every time. So I wrote a harder set with reworded questions, questions that need two documents, and near-miss distractors. That's where the model and the baseline pull apart. Every number here comes from a single run of one local model (27B, 4-bit, temperature 0) over three simulated documents. A replay script recomputes the figures from the saved raw outputs. The full method is in [projects/operator_rag/README.md](projects/operator_rag/README.md).

## How it works

```
question -> TF-IDF search over document chunks -> top 3 passages -> local LLM -> answer + cited passages
                                                                         |
                                     cites a passage it wasn't given, or a live operating request -> refuse
```

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,pdf]"
python -m pytest -q                                   # 36 tests; the LLM is mocked
python -m streamlit run projects/operator_rag/app.py
```

To score a live model, point `OPERATOR_RAG_LLM_URL` at any OpenAI-compatible chat endpoint and run `python projects/operator_rag/evals/run_eval.py`.

## How this was built

I wrote the code with AI coding agents working under my direction. I chose the problem and decided what counts as a pass. I designed the evals, including the baseline and the hard set, and every change had to pass its tests and an independent review before it landed. The domain judgement is mine: what an operator would trust, what the tool must refuse, and which limits have to sit beside each result.

## Limits

- The three documents are simulated study material, not real plant manuals.
- The question sets are small (20 and 12) and were scored on a single run.
- Retrieval is keyword-based (TF-IDF), so a question worded very differently from the document can miss.

There's no employer data, no confidential procedures and no live operating advice in this repo.

## Also by me

- [Plant fault detection on the Tennessee Eastman benchmark](https://github.com/nrvmos/plant-fault-detection): alarms judged by nuisance alarms per shift and minutes to detection.
