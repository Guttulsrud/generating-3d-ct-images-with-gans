# import time
#
# from tensorflow import keras
# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# from src.data_loader import DataLoader
# from src.model.discriminator import build_discriminator
# from src.model.generator import build_generator
# import os
#
# image_shape = (256, 256, 256, 3)
#
# generator = build_generator(image_shape)
# discriminator = build_discriminator(image_shape)
#
# # This method returns a helper function to compute cross entropy loss
# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#
#
# def generate_and_save_images(model, epoch, test_input):
#     # Notice `training` is set to False.
#     # This is so all layers run in inference mode (batchnorm).
#     predictions = model(test_input, training=False)
#
#     fig = plt.figure(figsize=(4, 4))
#
#     for i in range(predictions.shape[0]):
#         plt.subplot(4, 4, i + 1)
#         plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
#         plt.axis('off')
#
#     plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
#     plt.show()
#
#
# def discriminator_loss(real_output, fake_output):
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss
#
#
# def generator_loss(fake_output):
#     return cross_entropy(tf.ones_like(fake_output), fake_output)
#
#
# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
# BUFFER_SIZE = 60000
# BATCH_SIZE = 256
# EPOCHS = 50
# noise_dim = 100  # What exactly is this?
# num_examples_to_generate = 16
#
# # You will reuse this seed overtime (so it's easier)
# # to visualize progress in the animated GIF
# seed = tf.random.normal([num_examples_to_generate, noise_dim])
#
#
# # The training loop begins with generator receiving a random seed as input. That seed is used to produce an image.
# # The discriminator is then used to classify real images (drawn from the training set) and
# # fakes images (produced by the generator).
# # The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
#
# # Notice the use of `tf.function`
# # This annotation causes the function to be "compiled".
# @tf.function
# def train_step(images):
#     noise = tf.random.normal([BATCH_SIZE, noise_dim])
#
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_images = generator(noise, training=True)
#
#         real_output = discriminator(images, training=True)
#         fake_output = discriminator(generated_images, training=True)
#
#         gen_loss = generator_loss(fake_output)
#         disc_loss = discriminator_loss(real_output, fake_output)
#
#     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
#
#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
#
#
# def train(dataset, epochs):
#     print("Training")
#     for epoch in range(epochs):
#         print('Epoch: ', epoch)
#         start = time.time()
#         print(dataset)
#         for image_batch in dataset:
#             print(image_batch.shape)
#             train_step(image_batch)
#         exit()
#         # Produce images for the GIF as you go
#         # display.clear_output(wait=True)
#         generate_and_save_images(generator,
#                                  epoch + 1,
#                                  seed)
#
#         # Save the model every 15 epochs
#         if (epoch + 1) % 15 == 0:
#             checkpoint.save(file_prefix=checkpoint_prefix)
#
#         print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
#
#     generate_and_save_images(generator,
#                              epochs,
#                              seed)
#
#
# training_data = DataLoader('training').get_dataset()
# for x in training_data:
#     print(x.shape)
#     exit()
# # train(training_data, EPOCHS)
