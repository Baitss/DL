keras.backend.clear_session()

aug = keras.models.Sequential([
    keras.layers.Rescaling(1./255),
    keras.layers.RandomFlip(),
    keras.layers.RandomRotation(0.3),
    keras.layers.RandomZoom(0.2),
    # keras.layers.RandomTranslation(0.1, 0.1, 'nearest'),
])

root = keras.models.Sequential([
    keras.layers.Conv2D(32, (7,7), (2,2), activation='relu'),
    keras.layers.Conv2D(64, (5,5), (2,2), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    keras.layers.BatchNormalization()
])

left = keras.models.Sequential([
    keras.layers.Conv2D(96, (1,1), activation='relu'),
    keras.layers.Conv2D(64, (1,7), padding='same', activation='relu'),
    keras.layers.Conv2D(64, (7,1), padding='same', activation='relu'),
    keras.layers.BatchNormalization()
])

right = keras.models.Sequential([
    keras.layers.Conv2D(64, (1,1), activation='relu'),
    keras.layers.Conv2D(96, (3,3), padding='same', activation='relu'),
    keras.layers.Conv2D(96, (3,3), padding='same', activation='relu'),
    keras.layers.BatchNormalization()
])

fold1 = keras.models.Sequential([
    keras.layers.Conv2D(128, (3,3), (2,2), activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(48, (1,1), activation='relu'),
    keras.layers.BatchNormalization()
])

fold2 = keras.models.Sequential([
    keras.layers.Conv2D(128, (3,3), (2,2), activation='relu'),
    keras.layers.AvgPool2D(),
    keras.layers.Conv2D(48, (1,1), padding='same', activation='relu'),
    keras.layers.BatchNormalization()
])

fold3 = keras.models.Sequential([
    keras.layers.Conv2D(48, (3,3), (2,2), activation='relu'),
    keras.layers.Conv2D(48, (2,2), (2,2), activation='relu'),
    keras.layers.Conv2D(128, (1,3), padding='same', activation='relu'),
    keras.layers.Conv2D(128, (3,1), padding='same', activation='relu'),
    keras.layers.Conv2D(64, (1,1), activation='relu'),
    keras.layers.BatchNormalization()
])

il = keras.layers.Input(shape=(512, 512, 3))

il = aug(il)

ilr, ilg, ilb = tf.split(il, 3, axis=3, num=None, name='split')      # tensorflow의 split을 이용해서 채널을 분리.

ilr = root(ilr)
ilg = root(ilg)
ilb = root(ilb)
hl = keras.layers.Concatenate()((ilr, ilg, ilb))
hl = keras.layers.Conv2D(160, (1,1), activation='relu')(hl)
hl = keras.layers.Dropout(0.10)(hl)
hll = left(hl)
hlr = right(hl)
hl = keras.layers.Concatenate()((hll, hlr))

hl1 = fold1(hl)
hl2 = fold2(hl)
hl3 = fold3(hl)
hl = keras.layers.Concatenate()((hl1, hl2, hl3))
hl = keras.layers.Dropout(0.10)(hl)
hll = left(hl)
hlr = right(hl)
hl = keras.layers.Concatenate()((hll, hlr))

hl1 = fold1(hl)
hl2 = fold2(hl)
hl3 = fold3(hl)
hl = keras.layers.Concatenate()((hl1, hl2, hl3))
hl = keras.layers.Dropout(0.10)(hl)
hll = left(hl)
hlr = right(hl)
hl = keras.layers.Concatenate()((hll, hlr))

hl = keras.layers.Conv2D(32, (1,1), activation='relu')(hl)
hl = keras.layers.Flatten()(hl)
hl = keras.layers.Dense(64, activation='relu')(hl)
hl = keras.layers.Dense(32, activation='relu')(hl)
hl = keras.layers.BatchNormalization()(hl)
hl = keras.layers.PReLU()(hl)
hl = keras.layers.Dropout(0.20)(hl)
ol = keras.layers.Dense(1, 'sigmoid')(hl)

model = keras.models.Model(il, ol)

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

model.summary()
