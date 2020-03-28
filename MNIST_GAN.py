import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#Importing the MNIST DATASET to work with
(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()

#Dividing the dataset into Test and Trainset
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 100

#with the previous Declared batchsize and Buffer size we do a random slice and shuffle of the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#MAKING THE GENERATOR MODEL
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(7,(3,3),padding="same",input_shape=(28,28,1),activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(50,activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    return model

model_discriminator = make_discriminator_model()

model_discriminator(np.random.rand(1,28,28,1).astype("float32"))

#MAKING THE DISCRIMINATOR LOSS FUNCTION
def get_discriminator_loss(real_predictions,fake_predictions):
    real_predictions = tf.sigmoid(real_predictions)
    fake_predictions = tf.sigmoid(fake_predictions)
    real_loss = tf.losses.binary_crossentropy(tf.ones_like(real_predictions),real_predictions)
    fake_loss = tf.losses.binary_crossentropy(tf.zeros_like(fake_predictions),fake_predictions)
    return real_loss+ fake_loss

#MAKING GENERATOR MODEL
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256,input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape((7,7,256)))
    model.add(tf.keras.layers.Conv2DTranspose(128,(3,3),padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(64,(3,3),strides=(2,2),padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(1,(3,3),strides=(2,2),padding="same",activation="tanh"))
    
    return model

generator = make_generator_model()

#MAKING GENERATOR LOSS FUNCTION
def get_generator_loss(fake_predictions):
    fake_predictions = tf.sigmoid(fake_predictions)
    fake_loss = tf.losses.binary_crossentropy(tf.ones_like(fake_predictions),fake_predictions)
    return fake_loss

#GIVING BOTH THE NETWORKS An optimizer
discriminator_optimizer= tf.optimizers.Adam(1e-3)
generator_optimizer=tf.optimizers.Adam(1e-4)

#Trainining Step function that calculates and shows individuals losses
def train_step(images):
    noise = np.random.randn(BATCH_SIZE, 100).astype("float32")
      
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = model_discriminator(images, training=True)
        generated_output = model_discriminator(generated_images, training=True)

        gen_loss = get_generator_loss(generated_output)
        disc_loss = get_discriminator_loss(real_output, generated_output)


    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        
    gradients_of_discriminator = disc_tape.gradient(disc_loss, model_discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, model_discriminator.trainable_variables))

    print ("generator loss: ", np.mean(gen_loss))
    print ("discriminator loss: ", np.mean(disc_loss))

#Train function to pass the dataset and no.f epochs
def train(dataset, epochs):
  for _ in range (epochs):
    for images in dataset:
      images= tf.cast(images, tf.float32)
      train_step(images)

#Calling train function
train(train_dataset,20)