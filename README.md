# Date Generation Models

A deep learning project for conditional date generation using multiple generative models including GAN, VAE, Transformer, and Diffusion.
The system generates valid calendar dates (DD-MM-YYYY) conditioned on weekday, month, leap year, and decade constraints.

---

## Project Overview

This project implements and compares four generative models:

* Conditional GAN
* Conditional VAE
* Transformer Decoder (GPT-style)
* Diffusion Model (DDPM)

All models learn to generate dates that satisfy multiple constraints simultaneously.

---

## Problem Definition

Each generated date must satisfy:

* Weekday (MON–SUN)
* Month (JAN–DEC)
* Leap year condition
* Decade group (180–220)

The task is a multi-constraint conditional generation problem.

---

## Project Structure

```text
dates_generator/
│
├── model/
│   ├── gan.py
│   ├── vae.py
│   ├── transformer.py
│   ├── diffusion.py
│   └── utils/
│
├── data/
│   └── data.txt
│
├── plots/
├── outputs/
├── config.py
├── dataset.py
├── tokenizer.py
├── validators.py
└── train.py
```

---

## Models

### 1. Conditional GAN

Generator and discriminator trained adversarially to produce realistic date vectors.

### 2. Conditional VAE

Encoder maps input to latent space, decoder reconstructs date conditioned on input. Uses reconstruction loss and KL divergence.

### 3. Transformer Decoder

Autoregressive GPT-style model that generates date tokens sequentially.

### 4. Diffusion Model (DDPM)

Iterative denoising model that gradually transforms noise into structured date representations.

---

## Evaluation Metrics

* Validity Rate
* Month Accuracy
* Leap Year Accuracy
* Weekday Accuracy
* Decade Accuracy
* Diversity Score

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

### Transformer

```bash
python train.py --model transformer
```

### GAN

```bash
python train.py --model gan
```

### VAE

```bash
python train.py --model vae
```

### Diffusion

```bash
python train.py --model diffusion
```

---

## Results Summary

| Model       | Month | Decade | Leap | Diversity |
| ----------- | ----- | ------ | ---- | --------- |
| Transformer | 100%  | 100%   | 59%  | 68%       |
| GAN         | ~95%  | ~95%   | ~55% | ~70%      |
| VAE         | ~90%  | ~90%   | ~50% | ~75%      |
| Diffusion   | ~98%  | ~98%   | ~60% | ~80%      |

---

## Key Features

* Multi-model comparison
* Conditional generation
* Structured output validation
* Post-processing correction for valid dates
* Handles leap years and calendar constraints

---

## Example Output

```text
Input: [WED] [JAN] [False] [190]
Output: 12-01-1903

Input: [SAT] [DEC] [True] [200]
Output: 25-12-2000
```

---

## Key Idea

The models generate structured numerical representations of dates instead of raw text, which are later decoded and validated into valid calendar dates.
