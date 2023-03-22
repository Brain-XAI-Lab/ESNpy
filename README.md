# ESNpy
ESN(Echo-State-Network)
1. 전체적으로 ESN을 학습에 있어서는 먼저 ESN에 알맞은 parameters를 입력하고 정의한다
2. train values를 esn모델에 learning을 통해 input w, W를 learning 하고 fix시킨다
3. train values의 learning된 weight들과 구하고자 하는 output을 linear regression에 fit 한다
3. test values를 esn모델에 learning 하지말고 통과 시킨다
4. test values의 learning된 weight들을 linear regression에 predict를 통해 구하고자 하는 target을 알 수 있다
* 참고사항으로 esn에서 fit은 input 길이에 제한이 없지만 linear regression을 할 때에는 input의 길이에서 initLen을 뺀 길이 만큼을 피팅한다

--------------------
3/10 추가 수정
* generative mode를 통해 fit 까지 학습되어 있던 데이터에서 직후 원하는 데이터의 길이만큼을 predict 할 수 있는 강력한 성능을 지님. 
* 데이터를 input 할때에는 row는 feature, column은 time으로 작성되어 있으며, output weight는 row는 time, column은 weight의 갯수.
* numpy에서 torch를 통해 gpu 연산이 가능하도록 변경하였음.
---------------------
3/19 수정
* detach().numpy() tensor를 plt에 사용.
* Esn1.py는 backpropagation을 통해 학습하며 loss 함수로는 mseLoss를 사용 (regression에서는 mseloss를 주로 사용하고, classify에서는 crossentropy사용)
----------------------
3/20 수정
* Esn2.py를 auto encoder를 생성(back propagation을 이용하여 WL과 Wout을 업데이트)
* auto encoder를 통해 input값을 넣으면 esn 망을 지나서 latent space로 이동하는데 WL의 크기를 지정하여 노드는 원하는 만큼 생성하여 사용(노드의 수에 따라 성능의 차이가 있을 수도 있음).
* latent를 생성하여 다시 input 값으로 넣고 한번 더 esn망을 지나서 output을 생성하는데 이에 역전파를 통해서 output과 예측된 값이 같아지도록 WL과 Wout을 업데이트.
----------------------
3/22 수정
* detach 함수를 통해 업데이트 하고자 하는 것만 업데이트 가능.
* n_readout과 n_out을 통해 latent node가 1부터 다양하게 사용가능.
* tensor의 사이즈를 변경하여 vstack 과 matmul등이 가능하도록 하였음.
* 설명을 작성할 주석을 첨부할 예정.
