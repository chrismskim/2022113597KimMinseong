## 2025-NLP Project

### 2022113597 김민성

### 2022113594 홍승민

# 자연어처리 2025-1 지정주제 기말 프로젝트: GPT-2 구축

## 보고서도 해당 레포지토리에 올려놓았습니다.

## 1. 실행 방법 (Google Colab 기준)

1. `!git clone https://github.com/chrismskim/2022113597KimMinseong.git`
2. `%cd 2022113597KimMinseong`
3. `!pip install -r requirements.txt`

---

## 2. 폴더 구조

```
2022113597KimMinseong/
├── classifier.py
├── config.py
├── datasets.py
├── env.yml
├── evaluation.py
├── LICENSE
├── optimizer.py
├── optimizer_test.npy
├── optimizer_test.py
├── paraphrase_detection.py
├── prepare_submit.py
├── README.md
├── requirements.txt
├── sanity_check.py
├── sonnet_generation_basic.py
├── sonnet_generation_Lora_Peft.py
├── utils.py
├── data/
│   ├── ids-cfimdb-dev.csv
│   ├── ids-cfimdb-test-student.csv
│   ├── ids-cfimdb-train.csv
│   ├── ids-sst-dev.csv
│   ├── ids-sst-test-student.csv
│   ├── ids-sst-train.csv
│   ├── imdb_test.csv
│   ├── imdb_train.csv
│   ├── quora-dev.csv
│   ├── quora-test-student.csv
│   ├── quora-train.csv
│   ├── sonnets.txt
│   ├── sonnets_held_out.txt
│   ├── sonnets_held_out_dev.txt
│   └── TRUE_sonnets_held_out_dev.txt
├── models/
│   ├── base_gpt.py
│   └── gpt2.py
├── modules/
│   ├── attention.py
│   └── gpt2_layer.py
├── predictions/
│   └── README
├── sentiment analysis/
│   ├── Hybrid model.py
│   ├── RoBERTa.py
│   └── XLNet.py
```

---

## 3. Part 1

1. optimizer.py 구현을 테스트

   - 실행 방법: `!python optimizer_test.py`
   - 출력 예시:
     ```
     tensor([...])  # (ref 값)
     tensor([...])  # (actual 값)
     Optimizer test passed!
     ```

2. GPT 모델 구현을 테스트

   - 실행 방법: `!python sanity_check.py`
   - 출력 예시:
     ```
     Your GPT2 implementation is correct!
     ```

3. 모델을 사용한 감정 분류 수행
   - 실행 방법: `!python classifier.py --use_gpu`
   - 출력 예시:
     ```
     eval: 100% 31/31 [00:01<00:00, 17.07it/s]
     DONE DEV
     eval: 100% 61/61 [00:03<00:00, 17.70it/s]
     DONE Test
     dev acc :: 0.~~
     ```

---

## 4. Part 2

- paraphrase_detection.py: 패러프레이즈 탐지 수행

  - 실행 방법: `!python paraphrase_detection.py --use_gpu`
  - 출력 예시"

    ```
    Epoch 7: train loss :: 0.084, dev acc :: 0.896 (patience 2/4)
    train-8: 100% 35376/35376 [32:26<00:00, 18.18it/s]
    eval: 100% 5054/5054 [01:11<00:00, 70.25it/s]
    Epoch 8: train loss :: 0.070, dev acc :: 0.894 (patience 3/4)
    train-9: 100% 35376/35376 [32:46<00:00, 17.99it/s]
    eval: 100% 5054/5054 [01:14<00:00, 67.90it/s]
    Epoch 9: train loss :: 0.059, dev acc :: 0.895 (patience 4/4)
    Early stopping triggered at epoch 9!

    Loaded 40429 train examples from data/quora-dev.csv
    eval: 100%|██████████| 5054/5054 [01:20<00:00, 63.00it/s]
    Dev accuracy: 0.8966, F1 score: 0.8896
    ```

- sonnet_generation_basic.py: 소네트 생성 수행
  - 실행 방법: `!python sonnet_generation_basic.py --use_gpu`
  - 출력 예시:
    ```
    CHRF score: 26.2012
    ```

---

## 5. 기능 확장 TEST

- Sonnet_Lora:

  - 실행 방법: `!python sonnet_generation_Lora_Peft.py --use_gpu`
  - 출력 예시:
    ```
    CHRF score: 38.9614
    ```

- 감성분석 하이브리드/단일모델:
  폴더 이동: ` %cd sentiment_analysis`
  - RoBERTa.py (단일 RoBERTa)
    - 실행 방법: `!python RoBERTa.py --use_gpu`
    - 출력 예시:
      ```
      Using device: cuda
      [Epoch 1/5] Train Loss: ..., Test Loss: ..., Accuracy: ...
      ...
      classification_report
      ...
      ```
  - XLNet.py (단일 XLNet)
    - 실행 방법: `!python XLNet.py --use_gpu`
    - 출력 예시:
      ```
      Using device: cuda
      [Epoch 1/5] Train Loss: ..., Test Loss: ..., Accuracy: ...
      ...
      classification_report
      ...
      ```
  - Hybrid_model.py (RoBERTa + XLNet 앙상블)
    - 실행 방법: `!Hybrid_model.py --use_gpu`
    - 출력 예시:
      ```
      Device: cuda
      Training ROBERTA
      ROBERTA Epoch 1: ...
      ...
      Training XLNET
      XLNET Epoch 1: ...
      ...
      Best Weights: {'roberta': 0.6, 'xlnet': 0.4}
      Final Train (5k)  -> Loss ... | Acc ...
      Final Test        -> Loss ... | Acc ...
      Classification Report:
       ...
      ```
