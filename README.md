## python_style_transfer
- 建立virtual env與安裝package:
```
conda create -n style_env python=3.7.9
conda activate style_env
pip install -r function_ref/style_requirements.txt
```

## 使用方式
- demo code(可參照test_script.py):
```python
import neural_style_transfer as ns
content_path = 'function_ref/content.jpg'
style_path = 'function_ref/transfer_style1.jpg'
save_dir = 'transfer_result'
ns.main_transfer(content_path, style_path, save_dir, style_weight=0.1, content_weight=100, epochs=5, steps_per_epoch=30)
```

## 參數說明
1. 目前版本為0.1.0
2. neural_style_transfer中main_transfer各項參數說明
	- **content_path**: 內容圖片的路徑
	- **style_path**: 欲提取繪圖風格的圖片路徑，可以參考function_ref中提供的style
	- **save_dir**: 風格轉移完成後的圖片存檔位置
	- **style_weight**: 風格圖片的權重，預設為1
	- **content_weight**: 內容圖片的權重，預設為100 
	- **epochs**: 訓練的epoch總數，預設為1
	- **steps_per_epoch**: 每個epoch中訓練的步長，預設為20
3. main_transfer流程說明:
	- *Extract style and content:*
		- 在圖像上調用此模型，可以返回style_layers的gram矩陣(風格)和content_layers的內容
	- *Run gradient descent:*
		- 使用extractor提取style和content的內容
		- 定義一個`tf.Variable`來表示要迭代的圖像(size需與content圖像相同)
	- *Create an optimizer:*
		- 建立optimizers，程式內使用Adam，其參數可以自己修改或是使用其他optimizer
	- *start iteration:*
		- 針對設定的epochs與steps_per_epoch進行訓練，根據梯度下降的方式來優化目標並透過style_content_loss計算風格的總體損失
		- 將每個epoch的圖片存於save_dir中

## 參考資料
- [style_transfer in tensorflow](https://www.tensorflow.org/tutorials/generative/style_transfer)
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- [gram matrix](https://www.cnblogs.com/yifanrensheng/p/12862174.html)
