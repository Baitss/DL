keras.backend.clear_session()

il = keras.layers.Input(shape=(32, 32, 3))

aug = keras.layers.RandomFlip()(il)             # 이미지 augmentation
aug = keras.layers.RandomRotation(0.2)(aug)     # layer를 사용하는 방법.
aug = keras.layers.RandomZoom(0.2)(aug)         # fit 시에만 활성화됨. predict 같은 경우는 아무것도 하지 않고 바로 넘겨주지 않을까?

ilr, ilg, ilb = tf.split(
    aug, 3, axis=3, num=None, name='split'      # tensorflow의 split을 이용해서 채널을 분리.
)

clr1 = keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(ilr)
clg1 = keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(ilg)
clb1 = keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(ilb)

clr2 = keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(clr1)
clg2 = keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(clg1)
clb2 = keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(clb1)

hl = keras.layers.Concatenate()((clr2, clg2, clb2))
hl = keras.layers.BatchNormalization()(hl)
hl = keras.layers.MaxPool2D()(hl)
hl = keras.layers.Dropout(0.15)(hl)

hl = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(hl)
hl = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(hl)
hl = keras.layers.BatchNormalization()(hl)
hl = keras.layers.MaxPool2D()(hl)
hl = keras.layers.Dropout(0.15)(hl)

hl = keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(hl)
hl = keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(hl)
hl = keras.layers.BatchNormalization()(hl)

hl2 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(aug)
hl2 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(hl2)
hl2 = keras.layers.BatchNormalization()(hl2)
hl2 = keras.layers.MaxPool2D()(hl2)
hl2 = keras.layers.Dropout(0.15)(hl2)

hl2 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(hl2)
hl2 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(hl2)
hl2 = keras.layers.BatchNormalization()(hl2)
hl2 = keras.layers.MaxPool2D()(hl2)
hl2 = keras.layers.Dropout(0.15)(hl2)

hl2 = keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(hl2)
hl2 = keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(hl2)
hl2 = keras.layers.BatchNormalization()(hl2)

hl = keras.layers.Concatenate()((hl, hl2))
hl = keras.layers.Conv2D(512, (2,2), strides=(2, 2), activation='relu')(hl)     # 연산량 압축
hl = keras.layers.Conv2D(1024, (2,2), strides=(2, 2), activation='relu')(hl)

hl = keras.layers.Flatten()(hl)
hl = keras.layers.Dense(2048, activation='relu')(hl)
hl = keras.layers.Dense(4096, activation='relu')(hl)
hl = keras.layers.BatchNormalization()(hl)
hl = keras.layers.Dropout(0.25)(hl)
ol = keras.layers.Dense(ynum, 'softmax')(hl)                                     # 앞에서 구했던 총 클래스 수

model = keras.models.Model(il, ol)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

model.summary()
