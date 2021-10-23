# Modeling

## Limitations of Seq2Seq

<p align="center">
   <img src=".\../img/basic_seq2seq.png"/>
</p>

 RNN Cell을 사용한 Seq2Seq 모델은 Input Sequence의 길이가 길 경우 Long-Term Dependency (RNN에서 forward Propagation의 경우 뒷 단으로 갈 수록 앞의 정보가 유실되고, Back Propagation의 경우 Gradient Vanishing의 문제가 생기는 현상) 문제가 생길 수 있습니다.

 이를 개선하기 위해 그림과 같은 구조의 LSTM Cell로 구성된 Seq2Seq 모델을 만들면 Long-Term Dependency 문제는 해결될 수 있지만, 정보 손실의 문제는 여전히 생기게 됩니다. 그 이유는 Encoder에서 Decoder로 Input Sequence에 대한 정보를 전달하는 방식에 있습니다.

 Encoder는 Decoder로 Context Vector를 전달합니다. 이 때 Context Vector는 Encoder의 맨 마지막 LSTM의 Hidden State와 Cell State로, 이는 Input Sequence의 정보를 하나로 압축한 고정 길이 Vector입니다.  모든 정보를 하나의 고정 길이 Vector로 압축하였기 때문에  Encoder에서 입력받은 Input Sequence의 길이가 길 수록 정보 손실이 발생할 가능성이 커집니다. 이 문제를 해결하기 위한 방법이 Attention Mechanism입니다.

## Attention Mechanism in Seq2Seq

<p align="center">
   <img src=".\../img/Attention_mechanism.png"/>
</p>

 Attention을 적용하기 이전의 Seq2Seq는 Decoder에서 Encoder로부터 받는 정보가 Context Vector 하나였습니다. 이와 달리 Attention을 적용한 Seq2Seq는 Encoder로부터 Context Vector를 받고, Encoder내 각각의 LSTM들이 출력한 Hidden State 값들 또한 입력받습니다. Attention은 이 Hidden State값들을 이용하여 예측해야할 단어와 연관이 있는 부분에 좀 더 집중(Attnetion)해서 결과를 내도록 도와줍니다. 

 그러기 위해서 Seq2Seq의 Decoder에서는 하나의 출력 $y_i$를 만들 때마다 그림의 구조와 같은 하나의 Attention block을 거쳐야합니다. 그림을 보면 Query, Key, Value를 볼 수 있는데, Attention은 입력받은 Query를 모든 Key값들과 비교해서 각각의 유사도를 구해냅니다. 그런 다음, 구해낸 유사도 값 $c_i$를 각각의 Key와 Mapping되어있던 Value값에 반영해주어 최종적으로 유사도가 반영된 결괏값들의 합 $a_i$을 출력합니다.

 여기서 Key와 Value는 모두 Encoder의 Hidden state들을 의미하고, Query는 특정 시점에서 Decoder의 Hidden state를 의미합니다. (맨 처음 들어가는 Query인 $s_0$은 Encoder의 Context Vector입니다.)

## Encoder

<p align="center">
   <img src=".\../img/Encoder.png"/>
</p>

 Encoder는 Training 상태의 경우 `questions_padded` 라는 질문 data를 입력받고, Prediction 상태의 경우엔 사용자가 입력한 문장을 `questions_padded` 와 같은 형태로 전처리 하여 그 값을 입력받습니다. (각 과정에서 입력받는 data는 서로 다르지만, Encoder 내에서 입력 data를 처리하는 과정과 출력 형태는 같기 때문에 Training 과정을 대표로 설명하였습니다.)

 Encoder에서는 LSTM의 개수만큼 Hidden State값들이 출력됩니다. 이 값들을 H로 표현해봅시다. Attention을 적용하지 않은 Seq2Seq에서는 맨 마지막 LSTM의 Hidden State만 Context Vector로 사용되고, H는 사용되지 않습니다. 반면에 Attention을 적용한 경우에는 Context Vector와 함께 H도 사용됩니다. Encoder의 Output값 중 Context Vector는 Decoder의 첫 번째 Query(Hidden State) $s_0$가 되고, H는 Decoder의 모든 Attention block에 입력됩니다. Encoder를 좀 더 자세히 살펴보면 아래 과정들을 따릅니다.

**Embedding**

 Encoder의 입력으로 받은 `questions_padded` 는 질문에 해당하는 data들을 vocab을 참조해 문자를 숫자로 바꿔놓은 다음, padding을 해놓은 상태의 data입니다. 이를 model에서 사용하기 위해선 지금처럼 단순하게 숫자가 나열된 형태가 아닌, 의미를 가진 형태로의 변환이 필요합니다. 이 과정을 Embedding이라고 합니다.

 Embedding된 vector는 One-Hot Encoding vector의 Sparse한 vector가 아닌 Dense한 형태의 vector로, Keras의 Embedding Layer에서는 이를 Random Initialization하는 방식을 사용합니다. 그런 다음, Training 단계에서 weight가 업데이트되는 것처럼 Embedding값이 조정됩니다.

**Dropout**

 앞의 과정에서 Embedding된 input값인 `questions_padded` 는 model에 overfitting 문제를 일으키지 않기 위해 dropout 함수를 통과시켜주어야 합니다. Dropout은 인자로 준 값의 크기 만큼의 값들을 0으로 만들어버리는 역할을 합니다. Dropout까지 마쳤다면 Encoder에 입력되는 data가 training에 적절한 형태로 변환된 상태가 됩니다.

**LSTM**

 Dropout까지 완료된 question data는 LSTM에 입력되어 Context Vector와 H를 얻어내어야 합니다. 여기서 Context Vector는 Encoder내 LSTM들의 마지막 시점에서의 Hidden State와 Cell State를 의미하고, H는 Encoder안의 모든 시점에서의 Hidden State값들을 의미합니다.

 Keras의 LSTM에 전달한 인자들을 자세히 살펴보면 다음과 같습니다.

- `units` : Hidden State의 크기로, Encoder 안에 있는 LSTM들의 개수라고 생각할 수 있습니다.
- `return_sequences` : 값을 True로 하면 모든 시점에서의 Hidden State 값들을 얻을 수 있고, False로 하면 마지막 시저멩서의 Hidden State값만을 얻을 수 있습니다.
- `return_state` : 앞선 return_sequences 옵션을 보충해주는 옵션으로 생각할 수 있습니다. 이 값을 True로 하면 return_sequences 옵션 설정에 의한 출력 값에다 마지막 시점에서의 Hidden State값과 Cell State값 또한 얻을 수 있습니다.

 units의 값을 128로 하고, 나머지 인자들에 True를 입력했으므로 Encoder는 최종적으로 H와 Context Vector들을 반환하게 됩니다.

## Decoder

<p align="center">
   <img src=".\../img/Decoder.png"/>
</p>

 Decoder는 Training 상태의 경우 model의 학습이 목적이 되기 때문에 Encoder의 Training 상태에서 입력으로 받았던 질문 데이터 `questions_padded`의 답변 쌍인 `answer_input_padded` 를 입력받습니다. Prediction의 경우엔 사용자의 질문에 대한 답변을 직접 예측하여 생성해내야 하므로, 현재 시점에서의 Decoder의 출력 $y_i$가 다음 시점의 Decoder에 입력되는 방식을 사용합니다. 이 때 맨 처음에만 <SOS> 토큰을 입력해주어 Decoder가 직접 예측을 시작하게 해줍니다. 그림을 보면 이해가 더 쉽습니다. (Decoder도 Encoder와 마찬가지로 Training과 Prediction 각각 입력받는 data는 다르지만 처리과정은 동일하기 때문에 Training 과정을 대표적으로 설명하였습니다.)

**Embedding & Dropout**

 Encoder와 마찬가지로, Decoder가 입력받은 값들 중, `answer_input_padded` 를 model에서 사용할 수 있게 변환해줍니다.  그런 다음 overfitting을 방지하기 위해 Dropout을 진행해주면 data가 training에 적절한 상태로 변환됩니다.

**LSTM**

 Decoder의 LSTM은 Encoder와 달리 맨 처음 입력에서 Encoder의 output값인 Context Vector를 입력 받아야 합니다. 이를 위해 LSTM 호출 시 `initial_state` 값을 Encoder의 Context Vector로 지정해줍니다. (Keras의 LSTM은 `initial_state` 값을 명시하지 않을 경우엔 Default로 0으로 채워진 Tensor를 만듭니다.)

 LSTM의 output으로는 Decoder내 LSTM의 모든 Hidden State값들(S)과 마지막 시점에서의 Hidden State과 Cell State값을 받습니다.

**Attention**

 각각의 Attention block에는 Encoder로부터 받았던 H(Encoder 내 모든 LSTM들의 Hidden State값)와 Decoder의 현재 시점에서의 Hidden State값인 $s_i$가 입력됩니다.

 Attention block의 출력으로 attention value들을 얻을 수 있습니다. (Attention의 자세한 동작은 위의 Attention Mechanism in Seq2Seq 설명을 참고해주세요.)

**Concat**

 Attention값 $a_i$와 Decoder의 Hidden State값 $s_i$는 tf.concat을 통해 하나의 vector로 결합되어 출력 $y_i$를 만들고, 다음 LSTM으로 전달되게 해야합니다. Attention Value인 A와 Decoder Hidden State값 S를 axis = -1로 하여 tf.concat을 호출하였는데, 이는 가장 낮은 차원에서 두 Vector를 연결한다는 의미입니다. 이렇게 하여 Concat까지 마친 값들은 마지막 Layer인 Dense Layer로 입력됩니다.

**Dense**

 Dense Layer는 Decoder에서 출력을 계산하기 바로 전의 Layer로, Attention의 결괏값과 현재 시점에서의 Decoder Hidden State값(두 값은 Concat된 상태)을 입력받습니다.

 Dense Layer는 이렇게 입력받은 값을 출력과 연결해주는 역할을 합니다. Concat처럼 단순한 연결을 해주는 것이 아니라, Weight값을 포함하는 연결선을 만듦으로써 입력과 출력 각각의 연결에 강도를 설정해주게됩니다. (입력 뉴런의 개수에 출력 뉴런의 개수를 곱하면 연결선의 개수를 알 수 있고, 연결선의 개수는 곧 weight의 개수로 볼 수 있습니다.)

 Keras의 Dense layer에 전달한 인자들을 자세히 살펴보면 다음과 같습니다.

- `vocab_size` : 첫 번째 인자로, Output Neuron의 개수로 설정할 값입니다.
- `activation` : 사용할 Activation Function의 종류를 결정하는 인자입니다.

 Dense Layer에 Activation Function으로 준 Softmax는 챗봇과 같이 Multiclass classification에서 주로 쓰이는 함수입니다. 이는 실수 값을 가진 K개의 Vector를 입력받아 각각의 요소들을 정규화하여 0에서 1사이 값을 K개 출력합니다.

# Training & Prediction

## Train and Prediction in Seq2Seq

<p align="center">
   <img src=".\../img/Attention_seq2seq.png"/>
</p>

**Training**

 Encoder는 input으로 질문 data인 `questions_padded` 가 입력받아서 output으로 Context Vector와 모든 Hidden State값(H)를 내보냅니다.

 Decoder에서는 input으로 Encoder의 output값인 Context Vector, H를 받고, 이와 함께 질문 data `questions_padded` 의 답변 쌍인 `answer_input_padded` 를 입력받습니다. `answer_input_padded` 는 Decoder가 직접 예측해낸 값이 아닌, 질문에 대한 올바른 답변 쌍으로 지정된 Label값입니다. Label값을 학습에 이용하는 이유는 예측값을 사용했을 경우에 생기는 문제점 때문입니다. (예를 들어 Training 과정에서 예측값 $y_{i-1}$이 잘못 되었는데 이 값을 현재 예측에 사용해야한다면 현재의 예측값 $y_i$역시 잘못 예측될 가능성이 커지는 문제가 생깁니다.)

**Prediction**

 Encoder는 input으로 실제 사용자의 입력을 받습니다. 사용자로부터 입력받은 문장은 Training 과정에서 Encoder가 입력받았던 질문 data와 같은 형태로 전처리되어 Encoder로 입력됩니다. Encoder에서는 output으로 Context Vector와 모든 Hidden State값(H)을 내보냅니다.

 Decoder는 사용자의 입력과 Encoder로부터 받은 Context Vector, H값을 바탕으로 직접 답변을 예측하여 생성해내야 합니다. 답변 예측을 시작하기 위해서는 <SOS> 토큰이 필요한데, Prediction 과정에서는 이 시작 토큰 값을 맨 처음 한 번 직접 넣어주는 작업이 필요합니다. (Training 과정에서는 `answer_input_padded` 의 data 형태 자체가 문장의 맨 처음에 <SOS>토큰이 붙어있는 상태입니다.)

 Decoder가 답변을 예측하기 위해서는 이전의 예측값 $y_{i-1}$와 $s_i$( Hidden State, Cell State )를 입력 받고, 이 값들을 갱신하는 과정이 <EOS> 토큰을 예측해낼 때까지 반복되어야 합니다. 이렇게 최종적으로 예측한 값들은 숫자 형태로 되어있으므로, 이를 문자 형태로 바꾸면 model이 예측한 답변을 확인해볼 수 있게됩니다.

## Compile

**model.compile( )**

 Keras의 compile( )함수는 model의 Training 환경을 설정해주는 함수입니다. compile 함수에 전달한 인자들을 자세히 살펴보면 다음과 같습니다.

- `optimizer` : Optimizer 함수를 설정하기 위한 인자입니다. model 학습에 있어서 핵심적인 부분으로, Loss함수의 값을 최소화 하기 위해 Parameter의 최적값을 찾는 방법을 정해줍니다. 여러 Optimizer를 넣어 테스트 해 본 결과 adam을 사용했을 때 가장 좋은 결과를 보여 adam을 Optimizer로 설정하였습니다.
- `loss` : Loss 함수를 설정하기위한 인자입니다. 해결하고자 하는 문제 유형에 따라 Loss 함수의 선택이 달라지는데, 챗봇은 Multiclass classification 유형이므로 catagorical_crossentropy를 Loss 함수로 설정하였습니다.
- `metrics` : Training 과정을 확인해볼 때 어떤 값들을 볼 지 정하는 인자입니다. 정확도를 확인해보기 위해 acc를 인자로 주었습니다.

## Train

**model.fit( )**

 Keras의fit( )함수는 compile( )에서 지정한 방식으로 model의 Training을 진행하는 함수입니다. fit 함수의 인자들을 살펴보면 다음과 같습니다.

- `x` : model에 입력되는 입력데이터로, [questions_padded, answer_input_padded]를 전달해주었는데, Training 상태에서 questions_padded가 Encoder에 입력되고, answer_input_padded는 Decoder에 입력됩니다.
- `y` : 예측의 정답이 되는 Label 값으로, 질문 데이터 questions_padded의 답변 쌍인 answer_input_padded와 내용은 동일하지만 맨 앞이 아닌 맨 뒤에 <EOS> 토큰이 붙어있는게 차이점입니다.
- `epochs` : 전체 Training Set의 학습을 몇 번 반복할지 학습의 반복횟수를 결정합니다.
- `batch_size` : 1번의 Weight update(forward/backward propagation)에 필요한 Training Sample의 개수입니다. 전체 Training Set의 크기가 3,000이기 때문에 batch_size를 16으로 설정했을 경우 1번의 epochs 당 188번의 Weight update가 일어납니다. (이 때 188에 해당하는 용어가 iteration입니다.)

## Prediction