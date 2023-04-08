# 필요 라이브러리 설치.
!pip install konlpy pandas seaborn gensim wordcloud python-mecab-ko wget

# import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
from collections import Counter
import wget, os
from konlpy.tag import Okt
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from textblob import TextBlob
import re
import numpy as np
import mecab
import tensorflow as tf

# 파일 다운로드 (경로 지움)
!gdown 
data = pd.read_csv('/content/train.csv')
# 입력받는 데이터는 'text', 'label'
# 텍스트는 문의사항이며, 코드 블럭이 포함되어 있을 수 있다.
# 줄바꿈이 있다.
# 한글로 되어있다.

# label 전처리
label_dict = {
    '코드1': 0,
    '코드2': 0,
    '웹': 1,
    '이론': 2,
    '시스템 운영': 3,
    '원격': 4
}

data['label'] = data['label'].replace(label_dict)

# 전처리 통합 함수화
import re
import numpy as np
import mecab
import tensorflow as tf

def standardization_data(data, thres = 1.8, ngrams = 2, max_tokens = 35000, output_length = 50, test=False):

    data = np.array(data).reshape(-1)

    labels = ['plain', 'code', 'error', 'warning']
    n = len(labels)

    linecount = [[], [], [], [], [], []]
    lines = [[], [], [], []]

    mecab_instance = mecab.MeCab()
    okt = Okt()
    for idx, rawtext in enumerate(data):
        texts = rawtext.split(sep='\n')
        # 빈 줄 제거
        texts = [text for text in texts if text != '']
        flags = [0] * n
        feature = [[], [], [], []]

        flag = 0
        # 입력받은 텍스트 데이터를 스캔하여 코드, 에러, 경고 줄 분리
        for i, text in enumerate(texts):
            # 코드를 res에 저장
            result = re.findall(r"[a-zA-Z]*.*[a-zA-Z]+\([^)]*\)|[a-zA-Z0-9_ \+-=!]+=[a-zA-Z0-9_ ]+|import[a-zA-Z0-9 ]+|#[\w ]+", text)
            res = ''.join(result)
            # 해당 문의사항에 "원격"이란 글자가 포함된 경우
            if not flag:
                if len(re.findall(r"원격", text)):
                    flag = 1
            # 이전 문장이 평범한 텍스트가 아니였고, 이번 문장에 한글이 포함되어 있지만 #이 없을 경우 텍스트로 분류
            if flags[0]:
                a = re.findall(r"[ㄱ-ㅎㅏ-ㅣ가-힣]+", text)
                if len(a):
                    b = re.findall(r"#+[ㄱ-ㅎㅏ-ㅣ가-힣 ]+", text)
                    if not len(b):
                        flags[0] = 0
            # 경고 탐지
            wa = re.findall(r"Warning[\w\(\) )*:]", text, re.I)

            if len(wa):
                flags[0] = 3
            # 에러 탐지
            er = re.findall(r"(Error|Exception|NotFound|Traceback)[\w\(\)) ]*(Traceback|:)", text, re.I)
            
            if len(er):
                flags[0] = 2
            # 위에서 평범한 텍스트로 분류되었지만, 문장의 구성요소의 코드 비율이 임계 이상이면 코드로 재분류.
            # 이러한 방식을 차용한 이유는 이전 문장이 코드, 에러, 경고였을 경우 다음 문장을 일단 해당 분류로 체크한 후, 텍스트일 확률을 계산하여
            # 코드 블럭 뭉치, 에러 블럭 뭉치 등등을 일관성 있게 분류할 수 있게 하기 위함.
            if not flags[0]:
                if len(res) * thres > len(text):
                    flags[0] = 1
            # 최종적으로 분류된 문장을 분리하여 저장한 후 분류 별 문장 수를 카운트
            feature[flags[0]].append(text)
            flags[flags[0]] += 1
        # 분류된 문장들을 차례대로 형태소로 분리
        for _ in range(n):
            if len(feature[_]):
                lines[_].append(preprocess_text(" ".join(feature[_]), _, mecab_instance, norm=okt))
            else:
                lines[_].append("")
        # 문장 수 카운트 후 Softmax 형태로 저장
        for _ in range(n):
            if _:
                linecount[_].append(flags[_] / (i + 1))
            else:
                linecount[_].append((i + 1 - sum(flags[1:])) / (i + 1))
        # 해당 문의의 전체 문장 수
        linecount[4].append(i + 1)
        # "원격" 단어 포함 여부
        linecount[5].append(flag)

    global vectorized_layer, vectorized_layer2, vectorized_layer3, vectorized_layer4, vectorized_layer_merged
    global outlength
    outlength = output_length
    # Plain Text를 제외한 코드, 에러, 경고 문장 통합
    lines.append([code + error + warning for code, error, warning in zip(lines[1], lines[2], lines[3])])
    # Test 데이터를 가공하는 경우, adapt 과정 생략
    if not test:
        vectorized_layer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            split="whitespace",
            ngrams=ngrams,  # Unigrams and bigrams
            output_mode='int',
            # pad_to_max_tokens=True,
            output_sequence_length=outlength,
            # standardize=standardization_data
        )

        vectorized_layer2 = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens // 15,
            split="whitespace",
            ngrams=ngrams,  # Unigrams and bigrams
            output_mode='int',
            # pad_to_max_tokens=True,
            output_sequence_length=outlength,
            # standardize=standardization_data
        )

        vectorized_layer3 = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens // 15,
            split="whitespace",
            ngrams=ngrams,  # Unigrams and bigrams
            output_mode='int',
            # pad_to_max_tokens=True,
            output_sequence_length=outlength,
            # standardize=standardization_data
        )

        vectorized_layer4 = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens // 15,
            split="whitespace",
            ngrams=ngrams,  # Unigrams and bigrams
            output_mode='int',
            # pad_to_max_tokens=True,
            output_sequence_length=outlength,
            # standardize=standardization_data
        )

        vectorized_layer_merged = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens // 5,
            split="whitespace",
            ngrams=ngrams,  # Unigrams and bigrams
            output_mode='int',
            # pad_to_max_tokens=True,
            output_sequence_length=outlength,
            # standardize=standardization_data
        )

        vectorized_layers = [vectorized_layer, vectorized_layer2, vectorized_layer3, vectorized_layer4, vectorized_layer_merged]

        # labels = ['plain', 'code', 'error', 'warning']
        for _ in range(len(vectorized_layers)):
            vectorized_layers[_].adapt(np.array(lines[_], dtype=object))

        global vocab_size, vocab_size2, vocab_size3, vocab_size4, vocab_size5
        
        vocab_size = vectorized_layer.vocabulary_size()
        vocab_size2 = vectorized_layer2.vocabulary_size()
        vocab_size3 = vectorized_layer3.vocabulary_size()
        vocab_size4 = vectorized_layer4.vocabulary_size()
        vocab_size5 = vectorized_layer_merged.vocabulary_size()
    else:
        vectorized_layers = [vectorized_layer, vectorized_layer2, vectorized_layer3, vectorized_layer4, vectorized_layer_merged]

    result = []

    for _ in range(len(vectorized_layers)):
        result.append(np.array(vectorized_layers[_](np.array(lines[_], dtype=object))))

    return result[0], result[1], result[2], result[3], result[4], np.array(linecount).T

def preprocess_text(target = None, index = 0, method = mecab.MeCab(), method2 = TextBlob, norm=Okt()):
    anots = [
        'NNG', # 일반 명사
        'NNP', # 고유 명사
        # 'NNB', # 의존 명사
        # 'NNBC',# 단위를 나타내는 명사
        'NR',  # 수사
        'NP',  # 대명사
        'VV',  # 동사
        'VA',  # 형용사
        'VX',  # 보조용언
        'VCP', # 긍정 지정사
        'VCN', # 부정 지정사
        # 'MM',  # 관형사
        'MAG', # 일반 부사
        'MAJ', # 접속 부사
        # 'IC',  # 감탄사
        # 'JKS', # 주격 조사
        # 'JKC', # 보격 조사
        # 'JKG', # 관형격 조사
        # 'JKO', # 목적격 조사
        # 'JKB', # 부사격 조사
        # 'JKV', # 호격 조사
        # 'JKQ', # 인용격 조사
        # 'JC',  # 접속 조사
        # 'JX',  # 보조사
        # 'EP',  # 선어말어미
        # 'EF',  # 종결 어미
        # 'EC',  # 연결 어미
        # 'ETN', # 명사형 전성 어미
        # 'ETM', # 관형형 전성 어미
        # 'XPN', # 체언 접두사
        # 'XSN', # 명사 파생 접미사
        # 'XSV', # 동사 파행 접미사
        # 'XSA', # 형용사 파생 접미사
        'XR',  # 어근
        'SF',  # 마침표, 물음표, 느낌표
        # 'SE',  # 줄임표 ...
        # 'SSO', # 여는 괄호 (, [
        # 'SSC', # 닫는 괄호 ), ]
        # 'SC',  # 구분자 , · / :
        # 'SY',  # 기타 기호
        # 'SH',  # 한자
        'SL',  # 외국어
        'SN',  # 숫자
    ]
    
    anots_eng=[
        # 'CC',   # : coordinating conjunction (등위 접속사)
        # 'CD',   # : cardinal digit (기수)
        # 'DT',   # : determiner (한정사)
        # 'EX',   #: existential there (존재의 there)
        'FW',   #: foreign word (외래어)
        # 'IN',   #: preposition/subordinating conjunction (전치사/종속 접속사)
        # 'JJ',   #: adjective (형용사)
        # 'JJR',  #: adjective, comparative (비교급 형용사)
        # 'JJS',  #: adjective, superlative (최상급 형용사)
        # 'LS',   #: list marker (리스트 마커)
        # 'MD',   #: modal (조동사)
        'NN',   #: noun, singular (명사, 단수형)
        'NNS',  #: noun plural (명사, 복수형)
        'NNP',  #: proper noun, singular (고유 명사, 단수형)
        'NNPS', #: proper noun, plural (고유 명사, 복수형)
        # 'PDT',  #: predeterminer (전치 한정사)
        # 'POS',  #: possessive ending (소유격 종결 어미)
        'PRP',  #: personal pronoun (인칭 대명사)
        'PRP$', # : possessive pronoun (소유 대명사)
        # 'RB',   #: adverb (부사)
        # 'RBR',  #: adverb, comparative (비교급 부사)
        # 'RBS',  #: adverb, superlative (최상급 부사)
        'RP',   # : particle (관계사)
        # 'TO',   #: to go 'to' the store (가게에 '가다')
        # 'UH',   #: interjection (감탄사)
        # 'VB',   #: verb, base form (동사, 기본형)
        # 'VBD',  #: verb, past tense (동사, 과거형)
        # 'VBG',  #: verb, gerund/present participle (동사, 동명사/현재 분사)
        # 'VBN',  #: verb, past participle (동사, 과거 분사)
        # 'VBP',  #: verb, sing. present, non-3d (동사, 단수형 현재, 3인칭이 아닌)
        # 'VBZ',  #: verb, 3rd person sing. present (동사, 3인칭 단수형 현재)
        # 'WDT',  #: wh-determiner (의문사 한정사)
        # 'WP',   #: wh-pronoun (의문사 대명사)
        # 'WP$',  #: possessive wh-pronoun (소유 의문사 대명사)
        # 'WRB',  #: wh-abverb (의문사 부사)
    ]
    # MeCab은 대상이 None일 경우 코랩을 즉각적으로 셧다운시킴.
    if target == None:
        return ""
    # Plain Text일 경우 한글일 확률이 높으므로 MeCab 이용
    if not index:
        try:
            # 형태소를 분리하기 전 Okt를 이용해 텍스트 가공
            target = norm.normalize(target)
            mepos = method.pos(target)
            word = [pos[0] for pos in mepos if pos[1] in anots]
            return " ".join(word)
        except Exception as e:
            print(f"Error processing text at index {index}: {e}")
            return ""
    # 코드, 에러, 경고는 영어일 확률이 높으므로 BlobText 이용
    else:
        try:
            # mecab_instance = mecab.MeCab()
            target = norm.normalize(target)
            mepos = method2(target).pos_tags
            word = [pos[0] for pos in mepos if pos[1] in anots_eng]
            return " ".join(word)
        except Exception as e:
            print(f"Error processing text at index {index}: {e}")
            return ""
          
# Data Split 구현
import random as rd

def data_split(data, target = None, train_size = 0.8, thres = 1.8, ngrams = None, max_tokens = 30000, output_length = 50, val_size = 0, random_state = None):
    # 추후에 테스트 데이터 분류 결과를 확인하기 위해 global로 선언
    global test_idx

    if target == None:
        target = list(data)[-1]

    rd.seed(random_state)

    nontest_idx = []
    
    for _ in data[target].unique():
        idx = set(data.loc[data[target] == _].index)
        nontest_idx.append( set(rd.sample(list(idx), int(len(idx) * train_size))) )

    train_idx = set()

    if val_size:
        for _ in nontest_idx:
            train_idx.update( set(rd.sample(list(_), int(len(_) * (1 - val_size)))) )
    
    nontest_idx = set([idx for sublist in nontest_idx for idx in sublist])

    data_size = len(data)

    test_idx = list(set(range(data_size)) - nontest_idx)
    val_idx = list(nontest_idx - train_idx)
    train_idx = list(train_idx)
    nontest_idx = list(nontest_idx)
    # 위에서 선언한 함수는 텍스트 데이터만을 받을 것을 상정함.
    x = np.array(data.drop(target, axis=1))
    y = np.array(data[target])

    vector = standardization_data(data = x, thres = thres, ngrams = ngrams, max_tokens = max_tokens, output_length = output_length)

    data_size = vector[0].shape[0]

    x_train = []
    x_val = []
    x_test = []

    if val_size:
        for _ in vector:
            x_train.append(_[train_idx])
            x_val.append(_[val_idx])
            x_test.append(_[test_idx])

        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

        return x_train, x_val, x_test, y_train, y_val, y_test

    else:
        for _ in vector:
            x_train.append(_[nontest_idx])
            x_test.append(_[test_idx])

        y_train = y[nontest_idx]
        y_test = y[test_idx]

        return x_train, x_test, y_train, y_test

# split 테스트용 코드

# x_train, x_val, x_test, y_train, y_val, y_test = data_split(data, ngrams=1, val_size=0.2)
# np.unique(y_train, return_counts=True), np.unique(y_val, return_counts=True), np.unique(y_test, return_counts=True)

# 변수 initializing
result = data_split(data)


# 모델 생성 함수화
from tensorflow import keras

def create_model(output_length=50):
    global vocab_size, vocab_size2, vocab_size3, vocab_size4, vocab_size5, outlength
    if vocab_size == None:
        vocab_size = 5000
    if vocab_size2 == None:
        vocab_size2 = 5000
    if vocab_size3 == None:
        vocab_size3 = 5000
    if vocab_size4 == None:
        vocab_size4 = 5000
    if vocab_size5 == None:
        vocab_size5 = 5000
    if outlength == None:
        output_length = 50
    else:
        output_length = outlength

    keras.backend.clear_session()

    reg = keras.regularizers.L1L2(l1 = 0.01, l2=0.02)

    il1 = keras.layers.Input(shape=(output_length,), name='TextInput')
    il2 = keras.layers.Input(shape=(output_length,), name='CodeInput')
    il3 = keras.layers.Input(shape=(output_length,), name='ErrorInput')
    il4 = keras.layers.Input(shape=(output_length,), name='WarningInput')
    il5 = keras.layers.Input(shape=(output_length,), name='MergedInput')
    il6 = keras.layers.Input(shape=(6,), name='InfoInput')

    embedding_layer = keras.layers.Embedding(vocab_size, 256)(il1)
    hl = keras.layers.Bidirectional(keras.layers.LSTM(96, return_sequences=True))(embedding_layer)
    # hl = keras.layers.BatchNormalization()(hl)
    # hl = keras.layers.Conv1D(256, 3, 2)(hl)
    hl1 = hl

    embedding_layer2 = keras.layers.Embedding(vocab_size2, 64)(il2)
    hl = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(embedding_layer2)
    # hl = keras.layers.BatchNormalization()(hl)
    # hl = keras.layers.Conv1D(96, 3, 2)(hl)
    hl2 = hl

    embedding_layer3 = keras.layers.Embedding(vocab_size3, 64)(il3)
    hl = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(embedding_layer3)
    # hl = keras.layers.BatchNormalization()(hl)
    # hl = keras.layers.Conv1D(96, 3, 2)(hl)
    hl3 = hl

    embedding_layer4 = keras.layers.Embedding(vocab_size4, 64)(il4)
    hl = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(embedding_layer4)
    # hl = keras.layers.BatchNormalization()(hl)
    # hl = keras.layers.Conv1D(96, 3, 2)(hl)
    hl4 = hl
    
    embedding_layer5 = keras.layers.Embedding(vocab_size5, 128)(il5)
    hl = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(embedding_layer5)
    # hl = keras.layers.BatchNormalization()(hl)
    # hl = keras.layers.Conv1D(196, 3, 2)(hl)
    hl5 = hl
    
    hl = keras.layers.Concatenate()((hl1, hl2, hl3, hl4, hl5))
    hl = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(hl)
    hl = keras.layers.LSTM(512)(hl)
    hlc = keras.layers.BatchNormalization()(hl)

    hl = keras.layers.Dense(256, 'relu')(il6)
    hl6 = keras.layers.Dense(256, 'relu')(hl)

    hl = keras.layers.Concatenate()((hlc, hl6))

    hl = keras.layers.Dense(512, 'relu')(hl)
    hl = keras.layers.BatchNormalization()(hl)
    hl = keras.layers.Dense(256, 'relu')(hl)
    hl2 = keras.layers.Dense(64, 'softmax')(hl6)
    hl = keras.layers.Concatenate()((hl, hl2))
    hl = keras.layers.Dropout(0.2)(hl)
    ol = keras.layers.Dense(5, 'softmax')(hl)

    model = keras.models.Model((il1, il2, il3, il4, il5, il6), ol)
    model.compile('rmsprop', keras.losses.sparse_categorical_crossentropy, metrics='accuracy')
    
    return model

# 모델 생성 및 컴파일
model = create_model()
model.summary()

# 모델 학습 함수화
from sklearn.metrics import classification_report, confusion_matrix

def model_fitting(ngram=1, engrams=1, thres=1.8, max_tokens=30000, output_length=50, patience=8, stage=10, stage_patience=3, random_es_patience=True, train_size=0.8, val_size=0.2, val_split=False, random_state=None, mode='val_loss'):
    global x_train, x_val, x_test, y_train, y_val, y_test, y_pred, y_pred_max, model
    lr = keras.callbacks.ReduceLROnPlateau('val_loss', 0.9, patience, 0)
    best_score = None
    s_patience = stage_patience
    print(f"ngrams = {ngram}, eng_ngrams={engrams}")
    # Data Split 진행
    if val_split:
        x_train, x_test, y_train, y_test = data_split(data, target = None, train_size = train_size, thres = thres, ngrams = ngram, max_tokens = max_tokens, output_length = output_length, val_size = 0, random_state = random_state)
    else: 
        x_train, x_val, x_test, y_train, y_val, y_test = data_split(data, target = None, train_size = train_size, thres = thres, ngrams = ngram, max_tokens = max_tokens, output_length = output_length, val_size = val_size, random_state = random_state)
    # 모델 생성 함수 불러오기
    model = create_model(outlength)
    # For문을 이용해 모델을 여러번 fitting
    for _ in range(stage):
        # Early Stopping 의 Patience 인자를 랜덤화하여 변수를 만들어 봄
        if random_es_patience:
            es_patience = rd.randrange(patience, patience * 2)
        else:
            es_patience = patience
        print(f"Stage {_ + 1}: Early Stopping Patience is {es_patience}, fitting start...")
        es = keras.callbacks.EarlyStopping('val_loss', 0, es_patience, 0, restore_best_weights=True)
        if val_split:
            history = model.fit(x_train, y_train, epochs=10000, batch_size=256, callbacks=[es, lr], validation_split=val_size, verbose=0)
        else:
            history = model.fit(x_train, y_train, epochs=10000, batch_size=256, callbacks=[es, lr], validation_data=(x_val, y_val), verbose=0)
        print(f"\t {history.epoch[-1] + 1} Epochs, Early Stopping at {history.epoch[-es_patience]}")
        idx = history.history['val_loss'].index(min(history.history['val_loss']))
        print(f"\t Val Loss: {history.history['val_loss'][idx]}, Val Accuracy: {history.history['val_accuracy'][idx] * 100}%")
        # mode에 따라 Early Stopping과 유사하게 해당 인자를 모니터링하여 최고의 상태 저장.
        # 개선점: Early Stopping의 인자도 model_fitting의 인자로 지정할 수 있게 하면 좋을 것 같다.
        if mode[-4:] == 'loss':
            if best_score is None or best_score > history.history[mode][idx]:
                best_score = history.history[mode][idx]
                best_model = model.get_weights()
                best_stage = _ + 1
                s_patience = stage_patience
            else:
                s_patience -= 1
        else:
            if best_score is None or best_score < history.history[mode][idx]:
                best_score = history.history[mode][idx]
                best_model = model.get_weights()
                best_stage = _ + 1
                s_patience = stage_patience
            else:
                s_patience -= 1
        # Patience thres에 도달하면 루프 강제 종료
        if s_patience == 0:
            print(f"\n...Skipping Stage {_ + 2} ~ Stage {stage}\n")
            break
    # 최상의 상태 모델 불러오기
    print(f"Loading Stage {best_stage}'s weights...", end="")
    model.set_weights(best_model)
    print(" ...Done\n")
    score_model(model)

def score_model(model=model):
    global x_test, y_test, y_pred, y_pred_max
    result = model.evaluate(x_test, y_test, verbose = 0)
    print(f"Test Loss: {result[0]}, Test Accuracy: {result[1] * 100}%")
    print("="*100)
    y_pred = model.predict(x_test, verbose = 0)
    y_pred_max = y_pred.argmax(axis=1)
    print(classification_report(y_test, y_pred_max))
    print('='*100)
    print("코드, 웹, 이론, 운영, 원격")
    print(confusion_matrix(y_test, y_pred_max))
    
  # 학습
 model_fitting(1, 2, max_tokens=10000, output_length=50, thres=1.4, patience=10, stage=20, stage_patience=3, train_size=0.85, val_size=0.1, mode='val_accuracy')

## 결과 미리보기

# ngrams = 1, eng_ngrams=2
# Stage 1: Early Stopping Patience is 13, fitting start...
# 	 26 Epochs, Early Stopping at 13
# 	 Val Loss: 1.3579931259155273, Val Accuracy: 34.700316190719604%
# Stage 2: Early Stopping Patience is 16, fitting start...
# 	 23 Epochs, Early Stopping at 7
# 	 Val Loss: 0.979568600654602, Val Accuracy: 51.41955614089966%
# Stage 3: Early Stopping Patience is 12, fitting start...
# 	 13 Epochs, Early Stopping at 1
# 	 Val Loss: 1.0512292385101318, Val Accuracy: 61.51419281959534%
# Stage 4: Early Stopping Patience is 17, fitting start...
# 	 33 Epochs, Early Stopping at 16
# 	 Val Loss: 1.1238540410995483, Val Accuracy: 60.883283615112305%
# Stage 5: Early Stopping Patience is 17, fitting start...
# 	 18 Epochs, Early Stopping at 1
# 	 Val Loss: 1.0179804563522339, Val Accuracy: 68.13880205154419%
# Stage 6: Early Stopping Patience is 15, fitting start...
# 	 29 Epochs, Early Stopping at 14
# 	 Val Loss: 1.1933960914611816, Val Accuracy: 70.97792029380798%
# Stage 7: Early Stopping Patience is 13, fitting start...
# 	 14 Epochs, Early Stopping at 1
# 	 Val Loss: 1.1907294988632202, Val Accuracy: 71.29337787628174%
# Stage 8: Early Stopping Patience is 10, fitting start...
# 	 12 Epochs, Early Stopping at 2
# 	 Val Loss: 1.5392203330993652, Val Accuracy: 68.13880205154419%
# Stage 9: Early Stopping Patience is 18, fitting start...
# 	 21 Epochs, Early Stopping at 3
# 	 Val Loss: 1.345597743988037, Val Accuracy: 74.13249015808105%
# Stage 10: Early Stopping Patience is 16, fitting start...
# 	 24 Epochs, Early Stopping at 8
# 	 Val Loss: 1.6800564527511597, Val Accuracy: 75.07886290550232%
# Stage 11: Early Stopping Patience is 11, fitting start...
# 	 16 Epochs, Early Stopping at 5
# 	 Val Loss: 2.2046847343444824, Val Accuracy: 69.71608996391296%
# Stage 12: Early Stopping Patience is 19, fitting start...
# 	 34 Epochs, Early Stopping at 15
# 	 Val Loss: 1.9556611776351929, Val Accuracy: 71.92429304122925%
# Stage 13: Early Stopping Patience is 11, fitting start...
# 	 12 Epochs, Early Stopping at 1
# 	 Val Loss: 1.8915295600891113, Val Accuracy: 73.18611741065979%

# ...Skipping Stage 14 ~ Stage 20

# Loading Stage 10's weights... ...Done

# Test Loss: 1.5233747959136963, Test Accuracy: 71.8638002872467%
# ====================================================================================================
#               precision    recall  f1-score   support

#            0       0.72      0.82      0.77       238
#            1       0.66      0.59      0.62       110
#            2       0.66      0.66      0.66       110
#            3       0.90      0.62      0.74        85
#            4       0.88      0.93      0.90        15

#     accuracy                           0.72       558
#    macro avg       0.76      0.73      0.74       558
# weighted avg       0.73      0.72      0.72       558

# ====================================================================================================
# 코드, 웹, 이론, 운영, 원격
# [[196  11  27   3   1]
#  [ 34  65   7   3   1]
#  [ 33   4  73   0   0]
#  [  9  19   4  53   0]
#  [  1   0   0   0  14]]
