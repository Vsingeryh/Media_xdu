import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(x_test.shape)
# print(x_train.shape)
# print(y_test.shape)
# print(y_train.shape)

# 数据预处理
def train_preprocess(x_train, y_train):
    x_train = tf.cast(x = x_train, dtype = tf.float32) / 255.
    y_train = tf.cast(x = y_train, dtype = tf.int32)
    y_train = tf.one_hot(indices = y_train, depth = 10)

    return x_train, y_train


def test_preprocess(x_test, y_test):
    x_test = tf.cast(x = x_test, dtype = tf.float32) / 255.
    y_test = tf.cast(x = y_test, dtype = tf.int32)

    return x_test, y_test

train_db = tf.data.Dataset.from_tensor_slices(tensors=(x_train, y_train))
train_db = train_db.map(map_func=train_preprocess).shuffle(buffer_size=1000).batch(batch_size=128)

test_db = tf.data.Dataset.from_tensor_slices(tensors=(x_test, y_test))
test_db = test_db.map(map_func=test_preprocess).batch(batch_size=128)

# 建立网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=32, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10),

])
model.build(input_shape=[None, 28 * 28])
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)

if __name__ == '__main__':
    for epoch in range(80):
        for step, (x_train, y_train) in enumerate(train_db):
            x_train = tf.reshape(tensor=x_train, shape=[-1, 28 * 28])
            with tf.GradientTape() as tape:
                logits = model(x_train)
                loss = tf.losses.categorical_crossentropy(y_true=y_train, y_pred=logits, from_logits=True)
                loss = tf.reduce_mean(input_tensor=loss)

            gradient = tape.gradient(target=loss, sources=model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            if step % 100 == 0:
                print("第 %s epoch， 第 %s step， loss：%s" % (epoch, step, float(loss)))

        total_correct = 0
        total_num = 0
        for step, (x_test, y_test) in enumerate(test_db):
            x_test = tf.reshape(tensor=x_test, shape=[-1, 28 * 28])
            logits = model(x_test)
            probability = tf.nn.softmax(logits=logits, axis=1)
            prediction = tf.argmax(input=probability, axis=1)
            prediction = tf.cast(x=prediction, dtype=tf.int32)
            correct = tf.equal(x=prediction, y=y_test)
            correct = tf.cast(x=correct, dtype=tf.int32)
            correct = tf.reduce_sum(input_tensor=correct)

            total_correct += int(correct)
            total_num += x_test.shape[0]

        accuracy = total_correct / total_num
        print('accuracy:', accuracy)

    model.save('mnist.h5')