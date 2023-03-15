keras.backend.clear_session()

aug = keras.models.Sequential([                                      # keras의 layer를 이용한 데이터 증강.
    keras.layers.RandomFlip(),                                       # 레이어를 바로 집어넣으면 tf 2.9 이상의 버그로 fit 시간이 매우 느려짐
    keras.layers.RandomRotation(0.1),                                # 별도 모델로 분리하여 사용하니 적용되는지는 모르겠지만 fit 시간이 느려지지 않았음
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomTranslation(0.1, 0.1, 'nearest'),
])

il = keras.layers.Input(shape=(32, 32, 3))

il = aug(il)

ilr, ilg, ilb = tf.split(il, 3, axis=3, num=None, name='split')      # tensorflow의 split을 이용해서 채널을 분리.

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

hl2 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(il)
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
