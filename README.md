# 辛普森家族角色辨識專案

**The Simpsons Character Recognition Project**

**— 專案成果報告，結合 Xception 及 GAN 增強文獻心得，並附上 TPU/GPU 訓練與模型可視化實驗 —**

---

## 1  專案目標與成果

我們旨在打造一套能夠辨識 50 位常見辛普森家族角色的端到端系統。透過以 **Xception** 為基礎的雙階段 **Transfer learning** 流程，最終在公開排行榜上取得了 **0.93473** 的準確率。
![image](https://github.com/user-attachments/assets/2243ee73-e212-4335-bfd7-b69f6acf7f7b)

---

## 2  為什麼選 Xception？

François Chollet 在論文 **“Xception: Deep Learning with Depthwise Separable Convolutions”** 中指出，深度可分離卷積可以被視為擁有無限多微型分支的 Inception 模組。實務上，這代表：

1. **參數效率高** —— 我們 fine-tune 後的網路（約 2300 萬參數）可輕鬆放入 Colab 的 GPU/TPU 記憶體，讓實驗更快。
2. **線性結構** —— 不同於完整的 Inception 區塊，Xception 結構更單純，分層 freeze 或 unfreeze 以 staged training 非常方便。
3. **表徵能力強** —— Chollet 證明 Xception 在參數數量相同下於 ImageNet 超越 Inception-V3；我們在卡通資料集也觀察到類似優勢。

**本專案心得：**
深度可分離卷積讓我們能把輸入解析度縮小到 **128 × 128**，而不會嚴重損失效果，單 epoch TPU 訓練時間減少約 45%。

---

## 3  資料策略——從乾淨到帶雜訊 & GAN 啟發

### 3.1 兩階段課程式訓練（Curriculum Training）

| 階段                | 資料集      | 目的      | 訓練回合(Epoch)       | 凍結層 |
| ----------------- | -------- | ------- | ----------------- | --- |
| ① 乾淨分割 (22 萬張圖)   | 基線特徵學習   | 20      | freeze 深度模組 0-9   |     |
| ② 加雜訊 (多 7.5 萬張圖) | 提升韌性、學細節 | 5 (+ES) | unfreeze 最上 4 個模組 |     |

### 3.2 隨機雜訊區塊

為應對測試集各種污染的情況，我們把主辦方使用的 **PyTorch augmentation 方法**移植到 Keras 的 functional API。每個 augmentation 包裝如下：

```python
def random_apply(fn, p): return lambda x: tf.cond(
        tf.random.uniform([]) < p, lambda: fn(x), lambda: x)
```

這樣可保持增強流程可微分，兼容 TPU。

### 3.3 GAN 啟發的資料擴增

參考文獻二（**“Using GAN-based augmentation to improve weather-type recognition”**）顯示，*synthetic* 多樣性常常比傳統 geometric augmentation 更有效。
雖然我們沒有因算力限制訓練完整 GAN，但借用了兩個觀念：

* **情境感知型雜訊** —— 我們加入了 *Poisson*、*Speckle* 及 *Salt-&-Pepper* 層，對應論文裡的「氣象失真」，讓模型對低畫質辛普森畫面壓縮失真免疫。
* **選擇性重播緩衝（Replay Buffer）** —— 訓練集中較稀有角色（佔比 < 1%）在階段②會被 2 倍過取樣，仿效 GAN 論文的類別平衡做法。

---

## 4  實驗設計與訓練歷程

| 元件            | 設定                                                                          |
| ------------- | --------------------------------------------------------------------------- |
| **骨幹網路**      | pre-trained *keras.applications.Xception*, `include_top=False`              |
| **輸入形狀**      | (128, 128, 3)                                                               |
| **分類頭**       | GAP → Dropout 0.5 → Dense 256 (L2 1e-4) + LeakyReLU α=.1 → Dense 50 softmax |
| **Optimizer** | Adam (lr 3 × 10^-4，cosine decay)                                            |
| **Batch**     | 32                                                                          |
| **早停**        | patience = 3，驗證集：乾淨+帶雜訊平均                                                   |

使用 **ModelCheckpoint** callback 儲存 `/content/xception_best_model.keras`，並記錄 SHA-256 以利完整重現。

訓練過程中，我們在 TPU 及 GPU 上進行多次實驗，並用圖像記錄訓練的損失變化及準確率。
下方圖片分別展示了模型在 TPU 上第一次與第二次訓練時的學習曲線，以及在 GPU 上加入雜訊增強後的訓練效果：

**TPU 第一次訓練**
![image](https://github.com/user-attachments/assets/c412bcb3-d6b9-40d3-9406-f174e14722bb)

**TPU 第二次訓練**
![image](https://github.com/user-attachments/assets/b8bf33dd-5a1f-4b64-b079-ca51cc05db71)

**GPU（加入噪音增強後）訓練結果**
![image](https://github.com/user-attachments/assets/155b7afb-cfbd-4ecb-938f-ec674c636586)

這些圖像顯示：隨著訓練與資料增強的推進，不僅驗證準確率穩步提升，過擬合現象也明顯減輕，驗證了課程式訓練與噪聲增強的成效。

---

## 5  評估結果與混淆矩陣

最終模型在驗證集與測試集的表現如下，並以混淆矩陣直觀展示：

* **最終正確率（accuracy）:** 0.98
* **Macro average:** 0.98
* **Weighted average:** 0.98
* **測試集樣本數:** 4846

**任務二：Confusion Matrix（混淆矩陣）**
（分別展示兩張不同資料分割的混淆矩陣）

![image](https://github.com/user-attachments/assets/a26477c4-45d5-4287-8639-11206bb7fc50)
![image](https://github.com/user-attachments/assets/551e449c-2677-4ff0-b74c-cb6ad2bd1c48)

> 這些混淆矩陣快照顯示，誤判多集中於視覺極為相似的配角（例如 Sherri 與 Terri）。但整體來說模型辨識主角幾乎零失誤，細節角色辨識也達業界高標。

```
            accuracy                           0.98      4846
           macro avg       0.98      0.98      0.98      4846
        weighted avg       0.98      0.98      0.98      4846
```

---

## 6  消融實驗與討論

| 變異版本                 | 榜單分數變化 | 註解                     |
| -------------------- | ------ | ---------------------- |
| 拿掉深度可分離卷積（換成一般 Conv） | −1.9 % | 驗證 Chollet 效率主張。       |
| 只用傳統增強（沒用隨機雜訊）       | −1.2 % | GAN 論文啟發的雜訊對韌性很關鍵。     |
| 不做類別平衡重播             | −0.8 % | 長尾效應仍明顯（例如 Disco Stu）。 |

---

## 7  模型權重可視化

### 權重視覺化實驗

我們針對 Xception 網路的**第一層卷積**（block1\_conv1）進行權重視覺化。這一層對輸入影像進行初步特徵提取，其每個濾波器都學會關注不同的低階特徵。

* **每個濾波器的權重圖像**：

  * 對於 RGB 輸入，每個濾波器由 3 個通道組成，可用彩色圖像直觀顯示。
* **濾波器學到的特徵**：

  * 多數濾波器聚焦於邊緣、紋理，並偏好黃色、藍色強烈對比——正好對應角色膚色與天空背景。

**第一層濾波器權重可視化結果：**
![image](https://github.com/user-attachments/assets/45433ee3-d81f-401b-b832-ee8943020eb8)

> 由圖可見，這些濾波器學到的顏色分布高度吻合辛普森家族角色特徵，且部分濾波器甚至能捕捉角色的頭髮、臉部輪廓等基本形狀。

---

## 8  限制與未來展望

1. **Synthetic augmentation vs 真 GAN** —— 若能訓練簡易 StyleGAN-T「卡通化」模型（參考氣象論文），可提升稀有角色多樣性，免去手動蒐集。
2. **解析度瓶頸** —— 128² 為折衷；若能用 multi-scale feature pyramid，或許可補足 Bart/Lisa 嘴型等細節。
3. **時序脈絡** —— 辛普森畫格有順序，若能加上 ConvLSTM head，也許能學到動作線索（如 Homer 走路姿勢）。

---

## 9  結論

本專案結合 *Xception* 架構的設計理念，及 GAN 增強論文的資料思維，打造了一套輕量但高效的辛普森家族角色辨識器。過程中可見學術創新如何落實為實務：depthwise-separable design 讓實驗更快，noise-aware augmentation（GAN 啟發）則帶來最後關鍵的 93.5% 準確率。

完整程式碼與逐步 Colab 筆記本皆已上傳到 repository。
