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

hlf1 = fold1(hl)
hlf2 = fold2(hl)
hlf3 = fold3(hl)
hl = keras.layers.Concatenate()((hlf1, hlf2, hlf3))
hl = keras.layers.Dropout(0.10)(hl)
hll = left(hl)
hlr = right(hl)
hl = keras.layers.Concatenate()((hll, hlr))

# hl2 = root2(il)
hl2 = keras.layers.Conv2D(32, (7,7), (2,2), activation='relu')(il)
hl2 = keras.layers.Conv2D(64, (5,5), (2,2), activation='relu')(hl2)
hl2 = keras.layers.BatchNormalization()(hl2)
hl2 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(hl2)
hl2 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(hl2)
hl2 = keras.layers.BatchNormalization()(hl2)
hl2 = keras.layers.Conv2D(160, (1,1), activation='relu')(hl2)
hll2 = left(hl2)
hlr2 = right(hl2)
hl2 = keras.layers.Concatenate()((hll2, hlr2))
hl2 = keras.layers.Conv2D(160, (1,1), activation='relu')(hl2)
hl2 = keras.layers.Dropout(0.10)(hl2)

hl2f1 = fold1(hl2)
hl2f2 = fold2(hl2)
hl2f3 = fold3(hl2)
hl2 = keras.layers.Concatenate()((hl2f1, hl2f2, hl2f3))
hl2 = keras.layers.Dropout(0.10)(hl2)
hll2 = left(hl2)
hlr2 = right(hl2)
hl2 = keras.layers.Concatenate()((hll2, hlr2))
hl2 = keras.layers.Conv2D(480, (1,1), activation='relu')(hl2)
hl = keras.layers.Concatenate()((hl, hl2))
hl = keras.layers.Conv2D(160, (1,1), activation='relu')(hl)

hlf1 = fold1(hl)
hlf2 = fold2(hl)
hlf3 = fold3(hl)
hl = keras.layers.Concatenate()((hlf1, hlf2, hlf3))
hl = keras.layers.Dropout(0.10)(hl)
hll = left(hl)
hlr = right(hl)
hl = keras.layers.Concatenate()((hll, hlr))
hl = keras.layers.PReLU()(hl)
hl = keras.layers.Conv2D(32, (1,1), activation='relu')(hl)
hl = keras.layers.Flatten()(hl)
hl = keras.layers.Dense(128, activation='relu')(hl)
hl = keras.layers.Dense(96, activation='relu')(hl)
hl = keras.layers.BatchNormalization()(hl)
hl = keras.layers.PReLU()(hl)
hl = keras.layers.Dropout(0.20)(hl)
ol = keras.layers.Dense(1, 'sigmoid')(hl)                                     # 앞에서 구했던 총 클래스 수

model = keras.models.Model(il, ol)

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

model.summary()




# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to                     
# ==================================================================================================
#  input_2 (InputLayer)           [(None, 512, 512, 3  0           []                               
#                                 )]                                                                
                                                                                                  
#  tf.split (TFOpLambda)          [(None, 512, 512, 1  0           ['input_2[0][0]']                
#                                 ),                                                                
#                                  (None, 512, 512, 1                                               
#                                 ),                                                                
#                                  (None, 512, 512, 1                                               
#                                 )]                                                                
                                                                                                  
#  sequential_1 (Sequential)      (None, 125, 125, 12  275072      ['tf.split[1][0]',               
#                                 8)                                'tf.split[1][1]',               
#                                                                   'tf.split[1][2]']               
                                                                                                  
#  concatenate (Concatenate)      (None, 125, 125, 38  0           ['sequential_1[3][0]',           
#                                 4)                                'sequential_1[4][0]',           
#                                                                   'sequential_1[5][0]']           
                                                                                                  
#  conv2d_19 (Conv2D)             (None, 125, 125, 16  61600       ['concatenate[1][0]']            
#                                 0)                                                                
                                                                                                  
#  conv2d_20 (Conv2D)             (None, 253, 253, 32  4736        ['input_2[0][0]']                
#                                 )                                                                 
                                                                                                  
#  dropout (Dropout)              (None, 125, 125, 16  0           ['conv2d_19[1][0]']              
#                                 0)                                                                
                                                                                                  
#  conv2d_21 (Conv2D)             (None, 125, 125, 64  51264       ['conv2d_20[1][0]']              
#                                 )                                                                 
                                                                                                  
#  sequential_2 (Sequential)      (None, None, None,   87520       ['dropout_4[1][0]',              
#                                 64)                               'dropout_1[1][0]',              
#                                                                   'dropout_3[1][0]',              
#                                                                   'dropout[1][0]',                
#                                                                   'conv2d_24[1][0]']              
                                                                                                  
#  sequential_3 (Sequential)      (None, None, None,   149120      ['dropout_4[1][0]',              
#                                 96)                               'dropout_1[1][0]',              
#                                                                   'dropout_3[1][0]',              
#                                                                   'dropout[1][0]',                
#                                                                   'conv2d_24[1][0]']              
                                                                                                  
#  batch_normalization_7 (BatchNo  (None, 125, 125, 64  256        ['conv2d_21[1][0]']              
#  rmalization)                   )                                                                 
                                                                                                  
#  concatenate_1 (Concatenate)    (None, 125, 125, 16  0           ['sequential_2[8][0]',           
#                                 0)                                'sequential_3[8][0]']           
                                                                                                  
#  conv2d_22 (Conv2D)             (None, 125, 125, 12  73856       ['batch_normalization_7[1][0]']  
#                                 8)                                                                
                                                                                                  
#  sequential_4 (Sequential)      (None, None, None,   190832      ['conv2d_27[1][0]',              
#                                 48)                               'concatenate_1[1][0]',          
#                                                                   'dropout_2[1][0]']              
                                                                                                  
#  sequential_5 (Sequential)      (None, None, None,   190832      ['conv2d_27[1][0]',              
#                                 48)                               'concatenate_1[1][0]',          
#                                                                   'dropout_2[1][0]']              
                                                                                                  
#  sequential_6 (Sequential)      (None, None, None,   154784      ['conv2d_27[1][0]',              
#                                 64)                               'concatenate_1[1][0]',          
#                                                                   'dropout_2[1][0]']              
                                                                                                  
#  conv2d_23 (Conv2D)             (None, 125, 125, 12  147584      ['conv2d_22[1][0]']              
#                                 8)                                                                
                                                                                                  
#  concatenate_2 (Concatenate)    (None, 31, 31, 160)  0           ['sequential_4[4][0]',           
#                                                                   'sequential_5[4][0]',           
#                                                                   'sequential_6[4][0]']           
                                                                                                  
#  batch_normalization_8 (BatchNo  (None, 125, 125, 12  512        ['conv2d_23[1][0]']              
#  rmalization)                   8)                                                                
                                                                                                  
#  dropout_1 (Dropout)            (None, 31, 31, 160)  0           ['concatenate_2[1][0]']          
                                                                                                  
#  conv2d_24 (Conv2D)             (None, 125, 125, 16  20640       ['batch_normalization_8[1][0]']  
#                                 0)                                                                
                                                                                                  
#  concatenate_4 (Concatenate)    (None, 125, 125, 16  0           ['sequential_2[9][0]',           
#                                 0)                                'sequential_3[9][0]']           
                                                                                                  
#  conv2d_25 (Conv2D)             (None, 125, 125, 16  25760       ['concatenate_4[1][0]']          
#                                 0)                                                                
                                                                                                  
#  dropout_2 (Dropout)            (None, 125, 125, 16  0           ['conv2d_25[1][0]']              
#                                 0)                                                                
                                                                                                  
#  concatenate_5 (Concatenate)    (None, 31, 31, 160)  0           ['sequential_4[5][0]',           
#                                                                   'sequential_5[5][0]',           
#                                                                   'sequential_6[5][0]']           
                                                                                                  
#  dropout_3 (Dropout)            (None, 31, 31, 160)  0           ['concatenate_5[1][0]']          
                                                                                                  
#  concatenate_6 (Concatenate)    (None, 31, 31, 160)  0           ['sequential_2[7][0]',           
#                                                                   'sequential_3[7][0]']           
                                                                                                  
#  concatenate_3 (Concatenate)    (None, 31, 31, 160)  0           ['sequential_2[6][0]',           
#                                                                   'sequential_3[6][0]']           
                                                                                                  
#  conv2d_26 (Conv2D)             (None, 31, 31, 480)  77280       ['concatenate_6[1][0]']          
                                                                                                  
#  concatenate_7 (Concatenate)    (None, 31, 31, 640)  0           ['concatenate_3[1][0]',          
#                                                                   'conv2d_26[1][0]']              
                                                                                                  
#  conv2d_27 (Conv2D)             (None, 31, 31, 160)  102560      ['concatenate_7[1][0]']          
                                                                                                  
#  concatenate_8 (Concatenate)    (None, 7, 7, 160)    0           ['sequential_4[3][0]',           
#                                                                   'sequential_5[3][0]',           
#                                                                   'sequential_6[3][0]']           
                                                                                                  
#  dropout_4 (Dropout)            (None, 7, 7, 160)    0           ['concatenate_8[1][0]']          
                                                                                                  
#  concatenate_9 (Concatenate)    (None, 7, 7, 160)    0           ['sequential_2[5][0]',           
#                                                                   'sequential_3[5][0]']           
                                                                                                  
#  p_re_lu (PReLU)                (None, 7, 7, 160)    7840        ['concatenate_9[1][0]']          
                                                                                                  
#  conv2d_28 (Conv2D)             (None, 7, 7, 32)     5152        ['p_re_lu[1][0]']                
                                                                                                  
#  flatten (Flatten)              (None, 1568)         0           ['conv2d_28[1][0]']              
                                                                                                  
#  dense (Dense)                  (None, 128)          200832      ['flatten[1][0]']                
                                                                                                  
#  dense_1 (Dense)                (None, 96)           12384       ['dense[1][0]']                  
                                                                                                  
#  batch_normalization_9 (BatchNo  (None, 96)          384         ['dense_1[1][0]']                
#  rmalization)                                                                                     
                                                                                                  
#  p_re_lu_1 (PReLU)              (None, 96)           96          ['batch_normalization_9[1][0]']  
                                                                                                  
#  dropout_5 (Dropout)            (None, 96)           0           ['p_re_lu_1[1][0]']              
                                                                                                  
#  dense_2 (Dense)                (None, 1)            97          ['dropout_5[1][0]']              
                                                                                                  
# ==================================================================================================
# Total params: 1,840,993
# Trainable params: 1,839,393
# Non-trainable params: 1,600
# __________________________________________________________________________________________________
