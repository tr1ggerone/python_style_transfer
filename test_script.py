# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:30:13 2023

@author: HuangAlan
"""
import neural_style_transfer as ns

# %%
content_path = 'function_ref/content.jpg'
style_path = 'function_ref/transfer_style1.jpg'
save_dir = 'transfer_result'
ns.main_transfer(content_path, style_path, save_dir, 
                style_weight=0.1, content_weight=100,
                epochs=5, steps_per_epoch=30)
