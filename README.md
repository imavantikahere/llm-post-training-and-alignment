# LLM Post-Training: Fine-Tuning, Parameter-Efficient Adaptation & Preference Alignment

### Overview

Building a transformer architecture from scratch provides an understanding of how language models process information. However, pretraining a model is only one part of the LLM development lifecycle.

This repository explores what happens beyond pretraining: how pretrained language models can be adapted to specialized domains, trained to follow instructions, and optimized using human preference data.

The objective is to examine different LLM post-training methodologies through practical implementations using Hugging Face Transformers, PyTorch, PEFT, and TRL.

The repository focuses on the underlying training objectives, data preparation pipelines, parameter-update strategies, and trade-offs associated with each approach.

## 1. Objectives

The primary objectives of this repository are to:

- Understand the distinction between continued pretraining, supervised fine-tuning, and preference alignment.
- Explore full-parameter and parameter-efficient model adaptation.
- Examine the mathematical intuition behind LoRA, QLoRA, and DPO.
- Understand how different dataset structures influence training objectives.
- Analyze the computational requirements and limitations of different approaches.
- Document the complete workflow from data preparation to inference and evaluation.

## 2. Understanding the LLM Post-Training Lifecycle

A pretrained language model learns statistical patterns from large collections of text through a language-modeling objective.

Post-training methods allow pretrained models to be adapted for more specialized purposes.

These methods address different aspects of model development:

| Category | Techniques | Purpose |
|---|---|---|
| Domain adaptation | Continued pretraining | Adapt to specialized text distributions |
| Supervised adaptation | Instruction SFT | Improve responses to instructions |
| Parameter efficiency | LoRA, QLoRA | Reduce trainable parameters and memory requirements |
| Preference alignment | RLHF, DPO | Optimize behavior using preference signals |

These categories are not mutually exclusive. For example, LoRA can be used during supervised fine-tuning or preference optimization.

**[Insert original LLM post-training lifecycle flowchart here]**

## 3. Methodologies Explored

### 3.1 Continued Pretraining / Non-Instruction Fine-Tuning

**Objective:** Adapt a pretrained language model to domain-specific text using causal language modeling.

Topics explored:
- Raw text extraction and preprocessing
- Tokenization and sequence packing
- Next-token prediction
- Causal language-modeling loss
- Domain adaptation versus instruction tuning

### 3.2 Supervised Fine-Tuning (SFT)

**Objective:** Train a model on instruction-response examples to improve its ability to follow instructions.

Topics explored:
- Instruction-response dataset construction
- Prompt formatting
- Supervised learning objectives
- Token-level loss computation
- Continued adaptation of a domain-specific model

### 3.3 LoRA and QLoRA

**Objective:** Explore parameter-efficient fine-tuning methods that reduce the computational requirements of adapting pretrained models.

Topics explored:
- Frozen pretrained weights
- Low-rank weight updates
- Rank and scaling parameters
- Quantization of base-model weights
- Trainable parameter comparisons
- Memory and computational trade-offs

### 3.4 Reward Modeling and RLHF

**Objective:** Investigate reward-model-based preference alignment.

Topics to explore:
- Human preference datasets
- Reward model training
- Policy optimization using reward signals
- Reference models and KL regularization
- Computational costs and stability challenges

### 3.5 Direct Preference Optimization (DPO)

**Objective:** Explore direct optimization from preferred and rejected model responses.

Topics explored:
- Preference dataset structure
- Chosen and rejected responses
- Policy and reference model log probabilities
- DPO loss formulation
- Comparison with reward-model-based RLHF

## 4. Experimental Design

The experiments aim to investigate the behavior and computational characteristics of different training methodologies.

Where applicable, experiments will use consistent datasets, base models, evaluation prompts, and hardware configurations to support meaningful comparisons.

Evaluation dimensions may include:

- Training and validation loss
- Trainable parameter count
- Peak GPU memory consumption
- Training duration
- Task-specific evaluation metrics
- Instruction-following performance
- Preference satisfaction

**[Insert experimental architecture and comparison diagrams here]**

## 5. Results and Observations

Results will be documented after running and evaluating the experiments.

This section will include quantitative comparisons, qualitative response examples, observed trade-offs, and limitations.

## 6. Key Technical Questions

This repository explores several questions:

1. How does continued pretraining differ from instruction fine-tuning in its training objective and resulting model behavior?
2. What are the computational advantages and potential performance trade-offs of LoRA and QLoRA?
3. How does preference optimization differ from supervised instruction tuning?
4. How does DPO optimize a model without explicitly training a separate reward model?
5. Under which circumstances might one adaptation strategy be preferable to another?

## 7. Tools and Frameworks

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- TRL
- BitsAndBytes

## 8. References and Acknowledgments

The initial fine-tuning implementations were developed as part of my LLM bootcamp learning. This repository extends that foundation through independently prepared documentation, diagrams, additional experiments, and comparative analysis.

Relevant research papers, official documentation, and source materials will be credited here.

## 9. Scope and Limitations

This is an educational and experimental repository, not a novel fine-tuning framework or production-ready training system.

Experiments conducted on relatively small models or datasets are intended to illustrate training mechanisms and trade-offs. Their results should not automatically be generalized to larger language models or production environments.
