# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:57:28 2023

@author: HuangAlan
"""
__version__ = '0.1.0'
import tensorflow as tf

# %% img procress unit
def load_img(path_to_img, max_dim=512):
    
    # ----- pre process -----
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # ----- zoom in/out accroding to the max_dim -----
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape) 
    scale = max_dim/long_dim
    new_shape = tf.cast(shape*scale, tf.int32) # shape and scale in same unit
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :] # turn into 4 dim
    
    return img

def tensor_to_image(tensor):
    import numpy as np
    import PIL.Image
    
    # ----- transfer tensor to image -----
    tensor = tensor*255 # Make the pixel values from [0 , 1] to [0, 255]
    tensor = np.array(tensor, dtype=np.uint8) # Convert the pixels from float type to unit8 type
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1 # the first dim must be 1 in tensor
        tensor = tensor[0] # down to 3 dimension

    return PIL.Image.fromarray(tensor)

# %% transfer unit
def vgg_layers(layer_names):
    "Creates a VGG model that returns a list of intermediate output values."
    
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False # set as false to retrain
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    
    return model

def gram_matrix(input_tensor):
    "Calculate style"
    
    # ----- 4-D matrix inner product to 3-D -----
    gram_up = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) 
    input_shape = tf.shape(input_tensor)
    
    # ----- the denominator of gram -----
    gram_down = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    
    return gram_up/(gram_down)

class StyleContentModel(tf.keras.models.Model):
    "Build a model that returns the style and content tensors."
    
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs*255.0 # Expects float input in [0,1]
        
        # ----- transfer input to vgg19 format -----
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        
        # ----- style_outputs calculated the gram matrix as new style_outputs -----
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]
        
        # ----- write into dict -----
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}
        
        return {'content': content_dict, 'style': style_dict}

def clip_0_1(image):
    "this is a float image, define a function to keep the pixel values between 0 and 1"
    
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs, style_targets, style_weight, num_style_layers, 
                       content_targets, content_weight, num_content_layers):
    "To optimize this, use a weighted combination of the two losses to get the total loss"
    
    # ----- part of style -----
    style_outputs = outputs['style']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    
    # ----- part of content -----
    content_outputs = outputs['content']
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    
    # ----- get loss -----
    loss = style_loss + content_loss
    
    return loss

@tf.function()
def train_step(image, extractor, style_targets, style_weight, num_style_layers, 
               content_targets, content_weight, num_content_layers):
    "Use tf.GradientTape to update the image."
    
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, style_weight,
                                  num_style_layers, content_targets, 
                                  content_weight, num_content_layers)
    grad = tape.gradient(loss, image)
    
    return grad, loss

# %% main
def main_transfer(content_path, style_path, save_dir, 
                  style_weight=1, content_weight=100,
                  epochs=1, steps_per_epoch=20):
    "get the style from style_path and fusion into content_path"
    import IPython.display as display
    import os
    import time
    
    # ----- initial -----
    content_img = load_img(content_path)
    style_img = load_img(style_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = (content_path.split('/')[-1].split('.')[0] + '_' + 
                 style_path.split('/')[-1].split('.')[0])
    
    # ----- generate layer parameter form vgg -----
    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    # ----- Extract style and content -----
    extractor = StyleContentModel(style_layers, content_layers)

    # ----- Run gradient descent -----
    style_targets = extractor(style_img)['style'] 
    content_targets = extractor(content_img)['content']
    synthesis_img = tf.Variable(content_img)

    # ----- Create an optimizer -----
    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # ----- start iteration -----
    kwargs = dict(style_targets=style_targets,
                  style_weight=style_weight,
                  num_style_layers=num_style_layers,
                  content_targets=content_targets,
                  content_weight=content_weight,
                  num_content_layers=num_content_layers)

    _steps_cnt = 0
    t_start = time.time()
    for i_e in range(epochs):
        for i_s in range(steps_per_epoch):
            _steps_cnt += 1
            grad, loss = train_step(synthesis_img, extractor, **kwargs)
            opt.apply_gradients([(grad, synthesis_img)])
            synthesis_img.assign(clip_0_1(synthesis_img))
            print(f"Iteration {i_s}: loss={loss:.2f}")
        
        display.clear_output(wait=True)
        display.display(tensor_to_image(synthesis_img))
        image_name = os.path.join(save_dir, file_name+f'_at_epoch_{i_e}.png')
        tf.keras.preprocessing.image.save_img(image_name, tensor_to_image(synthesis_img))
        print(f"Train step: {_steps_cnt}")

    t_end = time.time()
    print(f"Time consumed: {t_end-t_start:.1f}")
