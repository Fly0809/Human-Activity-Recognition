import tensorflow as tf

class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.matrix = self.add_weight("matrix", shape=(num_classes, num_classes), initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=1)
        new_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes)
        self.matrix.assign_add(new_matrix)

    def result(self):
        return self.matrix

    def reset_states(self):
        self.matrix.assign(tf.zeros_like(self.matrix))