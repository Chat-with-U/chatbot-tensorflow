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