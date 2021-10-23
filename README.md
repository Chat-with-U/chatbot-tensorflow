# :speech_balloon: Chatbot_Tensorflow

Open Domain Chatbot을 Tensorflow와 Keras로 구현한 기록입니다.

<p align="center">
   <img src="img\block_diagram.png"/>
</p>

## Document

구현에 관한 자세한 설명을 아래 3가지 document로 정리했습니다.

- [Preprocessing](https://github.com/Chat-with-U/chatbot-tensorflow/blob/master/docs/Preprocessing.md)
- [Modeling](https://github.com/Chat-with-U/chatbot-tensorflow/blob/master/docs/Modeling.md)
- [Training](https://github.com/Chat-with-U/chatbot-tensorflow/blob/master/docs/Training.md)

3가지 과정을 한 번에 보고싶으시다면 다음 md파일을 참고해주세요.

- [Chatbot_Document](https://github.com/Chat-with-U/chatbot-tensorflow/blob/master/docs/Chatbot_Document.md)

## Dataset

송영숙님의 Chatbot dataset을 사용하였습니다.

- [songys/Chatbot_data](https://github.com/songys/Chatbot_data)

## Model

기본 Seq2Seq 모델을 사용한 버전과 Attention Mechanism을 적용한 버전을 구현했습니다.

- [Seq2Seq](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
- [Attention Mechanism](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention)

## Predictoin

직접 문장을 입력해 생성한 Chatbot과 대화를 해 본 결과입니다. 긍정적인 어조의 문장보다는 고민 상담 어조의 문장을 더 잘 인식하는 것으로 보였습니다.

**Best case**

<p align="center">
   <img src="img\best.png"/>
</p>

**Normal case**

<p align="center">
   <img src="img\normal.png"/>
</p>

**Worst case**

<p align="center">
   <img src="img\worst1.png"/>
</p>
:scream:
<p align="center">
   <img src="img\worst2.png"/>
</p>