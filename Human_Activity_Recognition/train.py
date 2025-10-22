import gin
import tensorflow as tf
import logging
import os
import wandb


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval,
                 wandb_project="default_project", wandb_run_name="default_run"):
        # Initialize WandB
        # wandb.init(project=wandb_project, name=wandb_run_name)

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(model=model)

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam()

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        self.model = model
        self.ds_train = ds_train.repeat()
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    def train(self):
        try:
            for idx, (images, labels) in enumerate(self.ds_train):
                step = idx + 1
                self.train_step(images, labels)

                if step % self.log_interval == 0:
                    # Reset validation metrics
                    self.val_loss.reset_states()
                    self.val_accuracy.reset_states()

                    for val_images, val_labels in self.ds_val:
                        self.val_step(val_images, val_labels)

                    template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                    logging.info(template.format(step,
                                                 self.train_loss.result(),
                                                 self.train_accuracy.result() * 100,
                                                 self.val_loss.result(),
                                                 self.val_accuracy.result() * 100))

                    # Reset train metrics
                    self.train_loss.reset_states()
                    self.train_accuracy.reset_states()

                    # Yield the validation accuracy for external tracking
                    yield self.val_accuracy.result().numpy()

                if step % self.ckpt_interval == 0:
                    logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                    self.ckpt.save(os.path.join(self.run_paths['path_ckpts_train'], 'ckpt'))

                if step >= self.total_steps:
                    logging.info(f'Finished training after {step} steps.')
                    break

        except StopIteration:
            logging.info('Dataset exhausted before reaching total steps.')

        # Final validation after training
        logging.info('Performing final validation after training completion.')
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()

        for val_images, val_labels in self.ds_val:
            self.val_step(val_images, val_labels)

        final_val_accuracy = self.val_accuracy.result().numpy()
        logging.info(f'Final Validation Accuracy: {final_val_accuracy:.4f}')

        # Ensure the final checkpoint is saved
        logging.info(f'Saving final checkpoint to {self.run_paths["path_ckpts_train"]}.')
        self.ckpt.save(os.path.join(self.run_paths['path_ckpts_train'], 'ckpt_final'))

        yield final_val_accuracy  # Yield final accuracy for comparison
