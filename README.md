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
Classification Report:
                          precision    recall  f1-score   support

  abraham_grampa_simpson       0.98      0.95      0.96       100
           agnes_skinner       0.99      0.99      0.99        92
  apu_nahasapeemapetilon       0.97      0.96      0.97       104
           barney_gumble       0.99      1.00      1.00       100
            bart_simpson       0.97      0.92      0.94       100
       brandine_spuckler       1.00      0.99      0.99        89
            carl_carlson       0.99      0.99      0.99        87
charles_montgomery_burns       0.98      0.98      0.98       125
            chief_wiggum       0.97      0.96      0.97       112
         cletus_spuckler       0.98      0.99      0.99       107
          comic_book_guy       0.93      0.97      0.95        90
               disco_stu       0.99      0.96      0.98        83
          dolph_starbeam       0.99      1.00      1.00       101
                duff_man       0.96      0.99      0.97        90
          edna_krabappel       0.98      0.93      0.95        86
                fat_tony       0.97      0.99      0.98       103
           gary_chalmers       1.00      0.99      1.00       111
                     gil       0.99      0.98      0.98        90
    groundskeeper_willie       0.96      0.98      0.97       106
           homer_simpson       0.94      0.98      0.96        93
             jimbo_jones       1.00      0.99      1.00       104
        kearney_zzyzwicz       0.99      0.98      0.98        85
           kent_brockman       1.00      0.98      0.99       103
        krusty_the_clown       0.99      0.95      0.97        98
           lenny_leonard       0.99      0.97      0.98        95
             lionel_hutz       1.00      0.96      0.98        93
            lisa_simpson       0.98      0.96      0.97        99
         lunchlady_doris       0.99      0.99      0.99        76
          maggie_simpson       0.98      0.98      0.98        88
           marge_simpson       0.98      0.98      0.98        85
           martin_prince       0.96      0.99      0.97       110
            mayor_quimby       0.96      0.94      0.95        97
     milhouse_van_houten       0.98      0.99      0.99       102
             miss_hoover       0.99      0.99      0.99        92
             moe_szyslak       0.98      0.99      0.98        90
            ned_flanders       0.98      0.97      0.98       111
            nelson_muntz       0.97      0.98      0.97        86
               otto_mann       0.95      1.00      0.98       103
           patty_bouvier       0.99      1.00      0.99        96
       principal_skinner       0.97      0.99      0.98       100
    professor_john_frink       0.98      0.96      0.97        95
      rainier_wolfcastle       0.94      1.00      0.97        97
            ralph_wiggum       0.99      0.99      0.99        91
           selma_bouvier       1.00      0.97      0.99       106
            sideshow_bob       0.98      0.95      0.97        88
            sideshow_mel       0.98      1.00      0.99       111
          snake_jailbird       0.94      1.00      0.97        94
         timothy_lovejoy       1.00      0.96      0.98        90
            troy_mcclure       0.97      0.99      0.98        84
         waylon_smithers       0.99      0.98      0.99       108

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
