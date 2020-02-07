# TODO: Please move to /autoencoders
#from keras.models import Model, Sequential
#from keras.layers import Dense, Input
#from keras.datasets import mnist
from audio_playground import load_snippets_as_mel_matrices
import numpy as np
import tensorflow as tf

#b = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

train_data= load_snippets_as_mel_matrices("data",2)
test_data=load_snippets_as_mel_matrices("data",2)
#print(train_data)
x_train =np.array(train_data)
x_test = np.array(test_data)
new_x_train=np.expand_dims(x_train,0)
x=tf.convert_to_tensor(new_x_train)

#converts  objects of various types to Tensor objects


learning_param=0.001
epochs=300
batch_size=32
# Network_parameters

data_dimension=64*87
neural_network_dimension=512
latent_variable_dimension=2

def xavier_normalisation(in_shape):
    val=tf.random_normal(shape=in_shape,stddev=1./tf.sqrt(in_shape[0]/2.))
    return val
#Weight and Bias Dictionaries

Weight={"weight_matrix_encoder_hidden":tf.Variable(xavier_normalisation([data_dimension,neural_network_dimension])),
         "weight_mean_hidden": tf.Variable(xavier_normalisation([neural_network_dimension, latent_variable_dimension])),
         "weight_std_hidden": tf.Variable(xavier_normalisation([neural_network_dimension,latent_variable_dimension])),
         "weight_matrix_decoder_hidden": tf.Variable(xavier_normalisation([latent_variable_dimension,neural_network_dimension])),
         "weight_decoder": tf.Variable(xavier_normalisation([neural_network_dimension,data_dimension])),

         }
Bias={"bias_matrix_encoder_hidden":tf.Variable(xavier_normalisation([neural_network_dimension])),
      "bias_mean_hidden":tf.Variable(xavier_normalisation([latent_variable_dimension])),
      "bias_std_hidden":tf.Variable(xavier_normalisation([latent_variable_dimension])),
      "bias_matrix_decoder_hidden":tf.Variable(xavier_normalisation([neural_network_dimension])),
      "bias_decoder":tf.Variable(xavier_normalisation([data_dimension]))}


#Encoder
data_x=tf.placeholder(tf.float32,shape=[None, data_dimension])
Encoder_layer=tf.add(tf.matmul(data_x,Weight["weight_matrix_encoder_hidden"]),Bias["bias_matrix_encoder_hidden"])
Encoder_layer=tf.nn.tanh(Encoder_layer)
#print(Encoder_layer)

Mean_layer=tf.add(tf.matmul(Encoder_layer,Weight["weight_mean_hidden"]),Bias["bias_mean_hidden"])
Standart_deviation_layer=tf.add(tf.matmul(Encoder_layer,Weight["weight_std_hidden"]),Bias["bias_std_hidden"])
#print(Standart_deviation_layer)

epsilon=tf.random_normal(tf.shape(Standart_deviation_layer),dtype=tf.float32,mean=0.0,stddev=1.0)
print(epsilon)
latent_layer=Mean_layer + tf.exp(0.5*Standart_deviation_layer)*epsilon
#print(latent_layer)

#Decoder
Decoder_hidden=tf.add(tf.matmul(latent_layer,Weight["weight_matrix_decoder_hidden"]),Bias["bias_matrix_decoder_hidden"])
Decoder_hidden=tf.nn.tanh(Decoder_hidden)
#print(Decoder_hidden)
Decoder_output_layer=tf.add(tf.matmul(Decoder_hidden,Weight["weight_decoder"]),Bias["bias_decoder"])
Decoder_output_layer=tf.nn.sigmoid(Decoder_output_layer)
#Defining the Varational Autoencoder Loss

def loss_function(original_data, reconstructed_data):
    data_loss=original_data*tf.log(reconstructed_data)+(1-original_data)*tf.log( 1- reconstructed_data)
    data_loss=-tf.reduce_sum(data_loss, 1)


    #KL Divergence Loss
    KL_div_loss=1+ Standart_deviation_layer - tf.square(Mean_layer)-tf.exp(Standart_deviation_layer)
    KL_div_loss=-0.5 * tf.reduce_sum(KL_div_loss,1)

    alpha=1
    beta=1
    network_loss=tf.reduce_mean(alpha*data_loss + beta*KL_div_loss)
    return network_loss
loss_value=loss_function(data_x,Decoder_output_layer)
optimizer=tf.train.RMSPropOptimizer(learning_param).minimize(loss_value)

#Initialize all the variables
init=tf.global_variables_initializer()

#Start the session

sess=tf.Session()
sess.run(init)




