# Simpsons Characters Recognition—從零到 Private Score **0.93473**

## 專案目標與成果

我們旨在打造一套能夠辨識 50 位常見辛普森家族角色的端到端系統。透過以 **Xception** 為基礎的雙階段 **Transfer learning** 流程，最終在公開排行榜上取得了 **0.93473** 的準確率。

![image](https://github.com/user-attachments/assets/2243ee73-e212-4335-bfd7-b69f6acf7f7b)

---

## 專案動機

* **Transfer Learning 快速驗證：** 想在限時內用最少 code 取得最好 baseline。
* **兩階段訓練：** 真實世界資料往往混雜雜訊，先在乾淨集學 general features，再在雜訊集 fine-tune，可驗證 **robustness** 的提升。

---

## 開發環境與資料結構

```text
SimpsonRecognition/
├── train/
│   ├── character_1/ (≈2 000 imgs)
│   ├── ...
├── test-final/            # 10791 imgs
└── notebooks/             # Colab scripts
```

* **Google Drive**：掛載 `/content/gdrive` 方便儲存 checkpoints 與最終模型
* **主要依賴**：`tensorflow==2.15`, `keras==3`, `scikit-learn`, `seaborn`

---

## 方法總覽

| 階段                              | 核心設定                                                                                                                  | 重點說明                                                  |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Phase 1 – Clean Training**    | `image_size=299×299` <br> freeze 前 100 層 <br> `lr=1e-4`, `batch=32`, `epochs=20`                                      | 只用基礎增強（水平翻轉、平移、旋轉），快速收斂至 \~**78 %** Val Acc           |
| **Phase 2 – Noisy Fine-Tuning** | 解除凍結至 layer 50 之後 <br> `lr=1e-5` with `ReduceLROnPlateau` <br> 自訂 16 種雜訊增強 (Gaussian, Poisson, Speckle, Salt-Pepper…) | 在乾淨+雜訊 Val set 雙重監控，Early Stopping（patience = 3）防止過擬合 |

## 資料策略——從乾淨到帶雜訊 & GAN 啟發

### 兩階段課程式訓練（Curriculum Training）

| 階段                | 資料集      | 目的      | 訓練回合(Epoch)       | 凍結層 |
| ----------------- | -------- | ------- | ----------------- | --- |
| ① 乾淨分割 (22 萬張圖)   | 基線特徵學習   | 20      | freeze 深度模組 0-9   |  100   |
| ② 加雜訊 (多 7.5 萬張圖) | 提升韌性、學細節 | 5 (+ES) | unfreeze 最上 4 個模組 |   50  |

### 隨機雜訊區塊

為應對測試集各種污染的情況，我們把主辦方使用的 **PyTorch augmentation 方法**移植到 Keras 的 functional API。每個 augmentation 包裝如下：

```python
def random_apply(fn, p): return lambda x: tf.cond(
        tf.random.uniform([]) < p, lambda: fn(x), lambda: x)
```

### GAN 啟發的資料擴增

參考文獻（**“Using GAN-based augmentation to improve weather-type recognition”**）顯示，*synthetic* 多樣性常常比傳統 geometric augmentation 更有效。
雖然我們沒有因算力限制訓練完整 GAN，但借用了觀念：
* **情境感知型雜訊** —— 我們加入了 *Poisson*、*Speckle* 及 *Salt-&-Pepper* 層，對應論文裡的「氣象失真」，讓模型對低畫質辛普森畫面壓縮失真免疫。

---

## Try & Error 心路歷程

| 嘗試                                       | 結果                                        | 反思                                                    |
| ---------------------------------------- | ----------------------------------------- | ----------------------------------------------------- |
| **將影像縮至 128×128** 直接訓練                   | 訓練快，但 Public LB < 0.85                    | 特徵資訊流失；轉而保持 Xception 原生 299×299                       |
| **一次性強烈雜訊增強**                            | 造成 early training collapse，Val Acc < 10 % | 先 clean 再 fine-tune，並降低增強 `p=0.05`–`0.1`              |
| **凍結所有 ImageNet layers**                 | Val Acc 卡在 \~60 %                         | Simpsons 角色色彩/線條獨特，需要解凍中高階層做 domain-specific learning |
| **Adam lr=1e-3**                         | Loss 振盪，無法收斂                              | 調降至 1e-4，並加入 `ReduceLROnPlateau`                      |

* `ReduceLROnPlateau`：當模型的某個評估指標（通常是驗證損失 val_loss）連續多個 epoch 都沒有進步時，這個 callback 會自動減小學習率 (learning rate)，讓模型進行更細緻的學習，有助於模型收斂。

---

## 模型效能與結果
### 加入噪聲後的訓練情況

![image](https://github.com/user-attachments/assets/a2ae8814-ed73-427f-b9cc-45d714dfe998)
* Validation Loss：驗證損失出現了波動，並未呈現持續下降的趨勢，特別是在中後期有反覆上升和下降的現象。
        * 模型開始過擬合。
        * 驗證資料中包含較多的噪聲，導致Validation Loss不穩定。
* Validation Accuracy：驗證準確度大致隨著訓練提升，但存在波動，在某些 epoch 下降，可能是由於資料增強或噪聲的影響。



| 指標                   | Val (Clean) | Val (Noisy) | Private Test |
| -------------------- | ----------- | ----------- | ------------ |
| Accuracy             | **0.882**   | **0.845**   | —            |
| Loss                 | 0.41        | 0.60        | —            |
| Kaggle Private Score | —           | —           | **0.93473**  |

### 模型看到什麼？卷積權重可視化分析
為了理解模型在學習初期「看見了什麼」，我們對 `block1_conv1`（Xception 的第一個卷積層）所學習的 32 個濾波器進行視覺化，如下圖所示：

![image](https://github.com/user-attachments/assets/f7c16b5f-9550-458f-8dc7-16f645ecc2c5)

* 視覺化對訓練的幫助：
1. 第一層卷積權重是模型感知世界的「第一眼」，若許多濾波器是灰白均勻->代表幾乎沒學到東西，代表訓練過程出了問題（可能是 learning rate 太小、梯度消失、權重沒解凍等）。
2. 這張圖中可見明確的色彩對比與邊緣方向性，代表模型已順利進入收斂。

![image](https://github.com/user-attachments/assets/d0322aa6-293d-4e7c-b121-38b12ed4bbe5)


### Confusion Matrix

![image](https://github.com/user-attachments/assets/5b9072eb-5885-4502-ae2f-951a1cc99886)

---

> **總結：**
> 本專案結合 Xception 架構的設計理念，及 GAN 增強論文的資料思維，打造了一套輕量但高效的辛普森家族角色辨識器。透過此專案，我不僅掌握了端到端的影像分類實務流程，更累積了在真實 noisy data 上調參、監控與部署的經驗，也充分培養我 **實作能力、系統化思考** 與 **持續改進** 的態度。
>
> 完整程式碼與逐步 Colab 筆記本皆已上傳到 repository。

---

<p align="center"><i>Made with ❤️ in Colab</i></p>
