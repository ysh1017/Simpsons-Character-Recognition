# the-simpsons-characters-recognition-challenge-iii-ML111701049
the-simpsons-characters-recognition-challenge-iii-ML111701049 created by GitHub Classroom

TPU第一次
![image](https://github.com/user-attachments/assets/334c31c3-3142-4787-b734-c12ebb26d4ac)

TPU第二次
![image](https://github.com/user-attachments/assets/ef10460a-5f1e-4499-9c71-3a0c6f239074)

GPU 加入噪音後
![image](https://github.com/user-attachments/assets/636053a9-6621-424a-a129-790e3a275002)

### 任務二：Confusion Matrix
![image](https://github.com/user-attachments/assets/eca78491-533b-4a9c-8261-e85928599039)
![image](https://github.com/user-attachments/assets/65781f50-f5a1-4272-a364-1096e010f571)

                accuracy                           0.98      4846
               macro avg       0.98      0.98      0.98      4846
            weighted avg       0.98      0.98      0.98      4846


### 任務三：權重可視化

從模型的層結構可以看到，第 1 層是第一個卷積層，名稱是 `block1_conv1`，類型為 `Conv2D`。我們可以使用它的權重來進行可視化。

1. **每個濾波器的權重圖像**：
   - 對於 RGB 輸入，每個濾波器由 3 個通道組成，會以彩色圖像顯示。
   - 對於灰度輸入，濾波器會以單一灰度圖像顯示。

2. **濾波器學到的特徵**：
   - 濾波器的圖像可以直觀地展示模型在第一層中學到的特徵，例如邊緣、紋理等。

3. **輸出說明**：
   - 如果模型的第一層是高層特徵（例如卷積），權重圖像會顯示模型初步學到的特徵。

![image](https://github.com/user-attachments/assets/9d8764bb-cb25-41a9-8c41-d3977316653d)
