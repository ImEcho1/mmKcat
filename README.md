# A Multimodal Deep Learning Framework for Enzyme Turnover Prediction with Missing Modality
Incomplete multimodal learning method for turnover number prediction
![alt text](pic/export.png)

Accurate prediction of the turnover number ($k_{\rm cat}$), which quantifies the maximum rate of substrate conversion at an enzyme's active site, is essential for assessing catalytic efficiency and understanding biochemical reaction mechanisms. Traditional wet-lab measurements of $k_{\rm cat}$ are time-consuming and resource-intensive, making deep learning (DL) methods an appealing alternative. However, existing DL models often overlook the impact of reaction products on $k_{\rm cat}$ due to feedback inhibition, resulting in suboptimal performance. The multimodal nature of this $k_{\rm cat}$ prediction task, involving enzymes, substrates, and products as inputs, presents additional challenges when certain modalities are unavailable during inference due to incomplete data or experimental constraints, leading to the inapplicability of existing DL models. To address these limitations, we introduce __mmKcat__, a novel framework employing a prior-knowledge-guided missing modality training mechanism, which treats substrates and enzyme sequences as essential inputs while considering other modalities as maskable terms. Moreover, an innovative auxiliary regularizer is incorporated to encourage the learning of informative features from various modal combinations, enabling robust predictions even with incomplete multimodal inputs. We demonstrate the superior performance of __mmKcat__ compared to state-of-the-art methods, including DLKcat, TurNuP, and UniKP, using BRENDA and SABIO-RK. Our results show significant improvements under both complete and missing modality scenarios in RMSE, $R^2$, and SRCC metrics, with average improvements of 6.89\%, 21.04\%, and 8.15\%, respectively.

## Performing prediction with mmKcat
### 1. Install necessary dependices
```python
pip install -r requirements.txt
```

### 2. Prepare data
You can download the splitted data from (this link.)[www.example.com], and then put them into folder 'data'.

### 3. Perform $k_{\rm cat}$ prediction
```python
cd model
python test_model.py
```
