import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_lab_categorical = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='uint8')
test_lab_categorical = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='uint8')

from sklearn.model_selection import train_test_split

train_im, valid_im, train_lab, valid_lab = train_test_split(x_train,
                                                            train_lab_categorical,
                                                            test_size=0.20,
                                                            stratify=train_lab_categorical,
                                                            random_state=42,
                                                            shuffle=True)

training_data = tf.data.Dataset.from_tensor_slices((train_im, train_lab))
validation_data = tf.data.Dataset.from_tensor_slices((valid_im, valid_lab))
test_data = tf.data.Dataset.from_tensor_slices((x_test, test_lab_categorical))

autotune = tf.data.AUTOTUNE

train_data_batches = training_data.shuffle(buffer_size=40000).batch(128).prefetch(buffer_size=autotune)
valid_data_batches = validation_data.shuffle(buffer_size=10000).batch(32).prefetch(buffer_size=autotune)
test_data_batches = test_data.shuffle(buffer_size=10000).batch(32).prefetch(buffer_size=autotune)

##### generate patches
class generate_patch(layers.Layer):
    def __init__(self, patch_size):
        super(generate_patch, self).__init__()
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding="VALID")
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

#############
# visualize
#############
from itertools import islice, count

train_iter_7im, train_iter_7label = next(islice(training_data, 7, None))
train_iter_7im = tf.expand_dims(train_iter_7im, 0)
train_iter_7label = train_iter_7label.numpy()
patch_size=4

generate_patch_layer = generate_patch(patch_size=patch_size)
patches = generate_patch_layer(train_iter_7im)

print('patch per image and patches shape: ', patches.shape[1], '\n', patches.shape)

from matplotlib import pyplot as plt
class_types = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def render_image_and_patches(image, patches):
    plt.figure(figsize=(6, 6))
    plt.imshow(tf.cast(image[0], tf.uint8))
    plt.xlabel(class_types [np.argmax(train_iter_7label)], fontsize=13)
    plt.savefig('figure01.png')
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(6, 6))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i+1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        ax.imshow(patch_img.numpy().astype("uint8"))
        ax.axis("off")
    plt.savefig('figure02.png')
    plt.close()

render_image_and_patches(train_iter_7im, patches)

### Positional Encoding Layer
class PatchEncode_Embed(layers.Layer):
    '''
    1. flatten the patches
    2. Map to dim D
    '''
    def __init__(self, num_patches, projection_dim):
        super(PatchEncode_Embed, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def generate_patch_conv_orgPaper_f(patch_size, hidden_size, inputs):
    patches = layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    row_axis, col_axis = (1, 2)
    seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
    x = tf.reshape(patches, [-1, seq_len, hidden_size])
    return x

### Positional Encoding Layer
class AddPositionEmbs(layers.Layer):
    """inputs are image patches
    Custom layer to add positional embeddings to the inputs."""
    def __init__(self, posemb_init=None, **kwargs):
        super().__init__(**kwargs)
        self.posemb_init = posemb_init
    def build(self, inputs_shape):
        pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
        self.pos_embedding = self.add_weight('pos_embedding', pos_emb_shape, initializer=self.posemb_init)
    def call(self, inputs, inputs_positions=None):
        pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)
        return inputs + pos_embedding
pos_embed_layer = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02))

'''
ViT implementation
'''
def ml_block_f(mlp_dim, inputs):
    x = layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
    x = layers.Dropout(rate=0.1)(x)
    x = layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x)
    x = layers.Dropout(rate=0.1)(x)
    return x

def Encoder1Dblock_f(num_heads, mlp_dim, inputs):
    x = layers.LayerNormalization(dtype=inputs.dtype)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(x, x)
    x = layers.Add()([x, inputs]) # 1st residual part

    y = layers.LayerNormalization(dtype=x.dtype)(x)
    y = ml_block_f(mlp_dim, y)
    y_1 = layers.Add()([y, x]) # 2nd residual part
    return y_1

def Encoder_f(num_layers, mlp_dim, num_heads, inputs):
    x = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input')(inputs)
    x = layers.Dropout(rate=0.2)(x)
    for _ in range(num_layers):
        x = Encoder1Dblock_f(num_heads, mlp_dim, x)

    encoded = layers.LayerNormalization(name='encoder_norm')(x)
    return encoded

'''
Building blocks of ViT
Transformer Encoder Block (Encoder_f)
Final Classification
'''

##########################
# hyperparameter section #
##########################
transformer_layers = 6
patch_size = 4
hidden_size = 64
num_heads = 4
mlp_dim = 128
##########################
rescale_layer = tf.keras.Sequential([layers.experimental.preprocessing.Rescaling(1./255)])
def build_ViT():
    inputs = layers.Input(shape=train_im.shape[1:])
    rescale = rescale_layer(inputs)
    patches = generate_patch_conv_orgPaper_f(patch_size, hidden_size, rescale)

    # transformer blocks
    encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, patches)

    # classification
    im_representation = tf.reduce_mean(encoder_out, axis=1)

    logits = layers.Dense(units=len(class_types), name='head', kernel_initializer=tf.keras.initializers.zeros)(im_representation)
    final_model = tf.keras.Model(inputs = inputs, outputs = logits)
    return final_model

ViT_model = build_ViT()
ViT_model.summary()