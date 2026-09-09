# RYTH Kaggle Smoke-Test Report

**Notebook:** `notebook/ryth_kaggle_start.ipynb`  
**Run type:** Kaggle smoke test  
**Status:** **PASSED ✅**

## 1. Executive Result

The RYTH Kaggle starter notebook successfully completed an end-to-end smoke run:

- Corpus loading/reuse
- Tokenizer setup
- RDS dataset preparation
- Model construction
- Training
- Validation/evaluation
- Training-curve generation
- Run summary generation

The smoke run completed all **12 training steps** without a crash.

## 2. Runtime

| Item | Result |
|---|---|
| Platform | Kaggle |
| Device | CUDA / Tesla T4 |
| Precision | FP16 |
| Model preset | `ryth_30m` |
| Reported parameters | **24.4M** |
| Vocabulary size | **482** |

## 3. Corpus

The corpus stage reported:

- **113 files/records**
- **16 repositories**
- Reported dataset size: **0.017 MB**
- Languages:
  - Python: 96
  - Markdown: 16
  - Unknown: 1
- Splits:
  - Train: 92
  - Validation: 14
  - Test: 7

The task distribution included next-token prediction, FIM, code-to-explanation, completion, and docstring-to-code examples.

## 4. Tokenizer

The notebook uses a **scratch byte-level BPE tokenizer**.

The smoke configuration targeted a vocabulary size of 1000, while the resulting tokenizer reported:

**Actual vocabulary size: 482**

The same 482-token vocabulary was passed into the model.

## 5. RDS Dataset

The dataset was exported into the training-ready RDS format.

Observed output:

- Train chunks: **102**
- Sequence length: **64**
- Validation split: **present**
- Train/validation/test splits were created successfully.

This confirms that the data reached the training-ready binary stage successfully.

## 6. Model

The selected model preset was:

```text
ryth_30m
```

Reported configuration:

- Parameters: **24.4M**
- Vocabulary: **482 tokens**
- Precision: **FP16**
- Micro-batch: **4**
- Gradient accumulation: **1**
- Initial learning rate: **3e-4**
- Smoke training budget: **12 steps**

## 7. Training Results

Training completed successfully through step 12.

Important reported metrics:

| Metric | Result |
|---|---:|
| Best validation loss | **4.370891451835632** |
| Best perplexity | **79.11412670692148** |
| Training time | **12.8 seconds** |
| Training steps | **12** |

The notebook also successfully saved the training curve.

### Loss behavior

The validation loss generally moved downward during the smoke run, ending at approximately **4.37**.

This is useful evidence that the training and evaluation path is functioning.

## 8. What This Proves

The smoke test demonstrates that:

1. The Kaggle environment works.
2. CUDA/T4 execution works.
3. The corpus pipeline works.
4. Tokenization works.
5. RDS dataset generation/loading works.
6. The RYTH model can be instantiated.
7. The trainer can execute optimization steps.
8. Validation can run.
9. Metrics are produced.
10. Training artifacts such as the curve and summary are generated.

**Therefore, the RYTH training pipeline is operational under the smoke configuration.**

## 9. What This Does NOT Prove

The smoke test should **not** be interpreted as proof that RYTH is already a capable LLM.

Only 12 optimization steps were performed. That is enough to validate the engineering pipeline, but nowhere near enough to judge:

- General language quality
- Coding ability
- Instruction following
- Long-context behavior
- Generalization
- Production readiness
- Final model quality

The reported perplexity of ~79 is therefore a **smoke-run metric**, not a serious benchmark result.

## 10. Important Caveats

### Small training corpus

The reported corpus is extremely small by modern LLM-pretraining standards. A real training run needs substantially more useful training data.

### Vocabulary size

The configuration targeted 1000 tokens in smoke mode, but the tokenizer actually reported 482 tokens. This should be reviewed before committing to a serious training run.

### Smoke configuration

The notebook explicitly uses:

```text
SMOKE = True
MAX_STEPS = 12
```

The real run should therefore use the full intended dataset and a substantially larger training budget.

## 11. Next Validation Stage

The smoke test validates the training pipeline. The next validation stage is a real-data training run with a substantially larger training budget, followed by quantitative and qualitative evaluation.

## 12. Final Status

> **RYTH Kaggle Smoke Test: PASSED ✅**

The RYTH Kaggle starter notebook has successfully demonstrated a working end-to-end training pipeline under the smoke configuration.

The next milestone is **real-data training with a substantially larger step budget**, followed by proper quantitative and qualitative evaluation.

---

**Bottom line:**  
The pipeline works. The model is **not yet validated as a good LLM**. The smoke test has done its job: it proved the machinery runs. Now the boring-but-important part begins, namely feeding it real data and training it for long enough to actually learn something.
