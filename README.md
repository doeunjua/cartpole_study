# cartpole_study
# **cartpole 공부 과정**
### **직접 카트폴 운전해보기 코드**
 
```import gymnasium as gym
import time

action = 0



from pynput import keyboard  # pip install pynput

def left():
    global action
    action = 0

def right():
    global action
    action = 1


listener = keyboard.GlobalHotKeys({
    'k': left,  # k는 왼쪽으로 이동
    'l': right  # l은 오른쪽으로 이동
})

listener.start()


env = gym.make('CartPole-v1', render_mode="human")
env.reset()
print("READY!")
time.sleep(2)

score = 0

while True:
    # env.step 진행
    _, _, done, _, _ = env.step(action)

    if done:
        print("GAME OVER! score: {}".format(score))
        time.sleep(1)
        break

    score += 1
    time.sleep(0.1)
```
스스로 개임을 했더니 다섯번 평균 점수가 7.5점이었다. 시작 하자마자 거의 바로 죽었다.


### **DQN 아이디어**
1. Q learning에서는 Q 테이블을 통해 상태를 행동으로 치환했다면, DQN에서는 Q 네트워크를 통해 상태를 행동으로 치환한다.**(상태 ->Deep Q network -> 행동)** DQN은 Q함수의 테이블을 네트워크로 치환한다는 기본적인 아이디어 외에도 성능을 높이기 위한 experience replay, target network 같은 아이디어들이 적용
2. target network: 타겟 네트워크는 DQN과 똑같은 neural network를 하나 더 만들어, 그 weight 값이 가끔씩만 업데이트 되도록 한것이다. Q(s,a)를 학습하는 순간, 타겟값도 따라 변하면서 학습 성능이 떨어지는 문제를 개선. 타겟 네트워크의 weight 값들은 주기적으로 DQN의 값을 복사해 온다. 
3. neural network는 3개의 convolution layer와 2개의 fully-connected layer가 있는 구조. input state는 이전의 84X84 크기의 프레임 4장이고, output은 18개의 joystick/button 동작에 대한 Q(s,a)값, reward는 소스의 변경 값
4. 네트워크 구조는 에이전트의 시야에 해당하는 입력을 받은 다음 convolution layer를 이용해서 이미지 정보를 압축하고 특징을 잡아냅니다. 그다음 dense layer를 연결하고 마지막에 액션의 크기와 같은 4개의 노드를 가진 dense layer를 연결해서, 네트워크의 출력을 에이전트의 행동으로 사용

 ## **강화학습으로 학습시키기**
목표: DQN으로 카트폴 게임 학습시키기

### **keras-rl2를 사용할 것이다.**
[keras-rl2 튜토리얼 링크](https://github.com/tensorneko/keras-rl2)
Keras-RL2는 Keras를 사용하여 강화학습을 구현할 수 있도록 지원하는 라이브러리. Keras-RL2는 OpenAI Gym 환경에서 강화학습을 구현할 수 있도록 다양한 알고리즘들을 제공함.


### **필수적인 라이브러리 불러오기**
```import gym
import random
import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory 
```
dense:인공신경망의 fully connected layer를 만들어주는 라이브러리
flatten: 추출된 주요 특징을 fully connected layer에 전달하기 위해 1차원 자료로 바꿔주는 layer 라이브러리

### **env**
```
env = gym.make('CartPole-v1')
states = env.observation_space.shape[0]
actions = env.action_space.n
```
states는 4이고, anction은 2이다.(left or right)
### **10회 실행**
```
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = random.choice([0,1])
        n_state,reward, done, info, _ = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
```
-env.reset(): step을 실행하다가 episode가 끝나서 이를 초기화해서 재시작해야할 때, 초기 state를 반환
-env.render(): gui(그래픽 유저 인터페이스)로 현재 진행상황을 출력하는 함수
action을 취하기 이전에 환경에 대해 얻은 관찰값 적용하여 그린다.
-random.choice([0,1]): 0과 1 중 하나를 랜덤으로 뽑아준다.
-env.step(): action 취하기 이후 환경에 대해 얻은 관찰값 적용하여 제어

처음에 n_state,reward, done, info = env.step(action) 네가지를 반환받는 줄 알고 이렇게 했다가 too many values to unpack (expected 4)라는 오류가 나왔는데 변수의 개수와 리턴해주는 변수의 개수가 차이가 날 때 발생한다는 것을 알고 수정하였다.

```
Episode:1 Score:27.0
Episode:2 Score:22.0
Episode:3 Score:11.0
Episode:4 Score:24.0
Episode:5 Score:14.0
Episode:6 Score:34.0
Episode:7 Score:15.0
Episode:8 Score:11.0
Episode:9 Score:14.0
Episode:10 Score:18.0
```



### **인공지능 모델 만들기**
 flatten layer 1층, dense layer 3층 이렇게 총 4층 layer를 만들거임.
 ```def build_model(states, actions):
    model = tensorflow.keras.Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
```
`model.add(Dense(24,activiation='relu'))` 에서 24는 출력 뉴런수를 activiation 은 활성화 함수를 의미. 성능이 좋은 방향으로 설정해주면 됨

#### **위에서 만든 모델 확인해보기**
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 4)                 0

 dense (Dense)               (None, 24)                120

 dense_1 (Dense)             (None, 24)                600

 dense_2 (Dense)             (None, 2)                 50

=================================================================
Total params: 770
Trainable params: 770
Non-trainable params: 0
_________________________________________________________________
```

flatten layer 1층 + Dense layer 3층 = 4층 layer 가 만들어졌다.

### **강화학습 에이전트 만들어주기**
build_agent라는 함수에 만들어 줄것이다. 코파일럿의 도움을 빌렸다.
```
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn
```
nb_steps_warmup은 파라미터가 너무 진동하는 것을 방지하기 위해 예열하는 작업이다.
target_model_update=1e-2은 얼마나 자주 타겟 모델을 업데이트 할것인가를 나타내 준다.

### **DQN 알고리즘 정의해주고, fitting**
```dqn = build_agent(build_model(states, actions), actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=10, visualize=False)
print(np.mean(scores.history['episode_reward']))
```
실행만 하면 될것이라 예상했으나 오류가 발생했다.
```ValueError: Error when checking input: expected flatten_input to have shape (1, 4) but got array with shape (1, 2)```
서치를 하며 이유를 찾아봤다. keras-rl2 라이브러리가 2021년 중단되었다고 한다.ㅎㅎ
처음부터 다시 시작해야 할것 같은 생각에 막막하다. 일단 자고 생각해야겠다.

## **다른 방법으로 시작**
### **라이브러리 임포트**
```
import numpy as np
import random
import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
```
`from collections import deque` 
:deque는 양방향 큐 이다. 앞, 뒤 모든 방향에서 원소를 추가, 제거 가능(double-ended queue의 약자)
### **학습을 위한 파라미터**
```
studyrate=0.9 
discount_rate=0.99 
eps=0.9 
eps_decay=0.999 
batch_siz=64 
n_episode=100
```
-studyrate:학습률

-discount_rate:할인율

-eps:엡실론

-eps_decay:엡실론 감소비율

-batch_siz:리플레이에서 샘플링할 배치 크기

-n_episode: 학습에 사용할 에피소드 개수

### **신경망 설계함수 만들기**
```
def deep_network():
    sqt=Sequential()
    sqt.add(Dense(32,input_dim=env.observation_space.shape[0], activation='relu'))
    sqt.add(Dense(32, activation='relu'))
    sqt.add(Dense(env.action_space.n, activation='linear'))
    sqt.compile(loss='mse', optimizer='Adam')
    return sqt
```
은닉층이 2개인 다층 퍼셉트론을 만들어 반환하는 함수이다.
`sqt.add(Dense(32,input_dim=env.observation_space.shape[0], activation='relu'))`여기에서 input_dim 매개 변수를 env.observation_space로 설정하였다. 이 문제를 위한 입력벡터(수레위치, 수레 속도, 막대 각도, 막대 각속도)의 정보가 env.observation_space.shape[0]에 들어 있다. 

`sqt.add(Dense(env.action_space.n, activation='linear'))` 여기서도  첫번째 매개변수는 출력 벡터의 크기인데, 이 정보도 env.action_space.n에 들어있다. 또한 활성함수를 linear로 설정한 이유는 누적 보상을 출력해야 하기 때문이다. 만약에 softmax로 바꾸면 [0,1]사이의 확률값으로 변환하므로 누적 보상액을 제대로 추정하지 못한다고 한다.

### **DQN 학습 함수 만들기**
```
def model_learning():
    mini_batch=np.asarray(random.sample(D, batch_siz))
    state= np.asarray([mini_batch[i,0] for i in range(batch_siz)])
    action=mini_batch[:,1]
    reward =mini_batch[:,2]
    state1= np.asarray([mini_batch[i,3] for i in range(batch_siz)])
    done=mini_batch[:,4]

    target=model.predict(state)
    target1=model.predict(state1)

    for i in range(batch_siz):
        if done[i]:
            target[i][action[i]]=reward[i]
        else:
            target[i][action[i]]+=studyrate*((reward[i]+discount_rate*np.amax(target1[i]))-target[i][action[i]]) #식 19
    model.fit(state,target,batch_size=batch_siz, epochs=1, verbose=0)
```
함수 안에 첫번째 줄 부터 6번째 줄까지 잘 살펴보면 이 부분은 리플레이 메모리 D에서 미니배치를 랜덤하게 샘플링한 다음에 미니배치에 저장되어 있는 현재상태(state), 행동(action), 보상(reward), 다음상태(state1), 에피소드 끝 여부(done)를 추출한다.

`target=model.predict(state)` target은 위에 dqn아이디어 부분에서 설명했던 개념이다. 이 코드는 현재 상태 state를 신경망에 입력해 예측을 수행한다.

`target1=model.predict(state1)` 이 코드는 다음 상태 state1을 신경망에 입력해 예측을 수행한다.

그 밑의 for문은 미니배치에 있는 샘플 각각에 대해 에피소드의 끝에 해당하면 reward를 target에 저장하고 그렇지 않으면 Q러닝식을 적용한 결과를 target에 저장한다.

마지막줄은 state와 target을 훈련 집합의 특정 벡터와 레이블로 활용해 fit 함수를 적용하여 학습한다. `epochs`에 대해서도 알아 봐야겠다. epochs는 일반적으로 데이터셋을 몇 번 반복해서 학습을 수행할지를 결정하는 매개변수이다. 강화학습에서는 데이터셋이라는 개념이 없기 때문에, epochs는 에이전트가 환경과 상호작용하여 수집한 정보를 몇 번 반복해서 사용할지를 결정하는 매개변수라고 한다. 강화학습에서 epochs를 조절하는 것은 학습의 효과와 속도에 영향을 미친다. 너무 적게 설정하면 학습이 충분히 이루어지지 않아 성능이 저하될 수 있고, 너무 많이 설정하면 학습 시간이 길어질 뿐만 아니라, 과적합(overfitting) 문제가 발생할 수 있다. 따라서 적절한 epochs 값을 찾는 것이 중요하다고 한다. 여기서 나는 epochs를 1로 설정하여 세대는 단 한번만 반복하도록 한다.

### **메인 프로그램 작성**
`env=gym.make("CartPole-v1")`: CartPole-v1에 해당하는 env 객체를 생성

```model=deep_network()
D=deque(maxlen=2000)
scores=[]
max_steps=env.spec.max_episode_steps
```
다중 퍼셉트론 모델을 생성하여 model 객체에 저장한 후 리플레이 메모리로 쓸 객체 D를 생성한다. 여기서 deque는 꽉찬 상태에서 새로운 요소 삽입 할 때 먼저 들어온 원소를 삭제한다.

scores는 에피소드의 누적 보상액을 저장할 리스트이다. 그 밑의 코드는 카트폴 문제가 허용하는 최대 에피소드 길이를 알아내는 코드다.

**신경망학습**
```
for i in range(n_episode):
    s,_=env.reset()
    long_reward=0

    while True:
        r=np.random.random()
        eps=max(0.01, eps*eps_decay)
        if(r<eps):
            a=np.random.randint(0, env.action_space.n) 
        else:
            q=model.predict(np.reshape(s,[1,4]))
            a=np.argmax(q[0])
        s1, r, done,_,_ =env.step(a)
        if done and long_reward<max_steps-1:
            r=-100

        D.append((s,a,r,s1,done))

        if len(D)>batch_siz*3:
            model_learning()

        s=s1
        long_reward+=r

        if done:
            long_reward=long_reward if long_reward==max_steps else long_reward+100
            print(i,"번째 에피소드의 점수:",long_reward)
            scores.append(long_reward)
            break
    if i>10 and np.mean(scores[-5:])>(0.95*max_steps):
        break
```
처음 for문 밑의 코드는 env를 reset하고 누적 보상액을 초기화한다.(새로운 에피소드 시작)
이 이후에 나오는 while문 내의 구문이 에피소드 하나를 처리하는 부분이다.
```
r=np.random.random()
        eps=max(0.01, eps*eps_decay)
        if(r<eps):
            a=np.random.randint(0, env.action_space.n) 
        else:
            q=model.predict(np.reshape(s,[1,4]))
            a=np.argmax(q[0])
```
while문 안의 이 부분은 현재상태 s에서 행동 a를 결정하는데 입실론 탐욕을 적용하는 부분이다. 입실론 값은 eps변수에 저장해 놓았고 eps비율만큼 랜덤하게  1-eps만큼은 현재 상태 s를 신경망에 입력하여 예측한 결과를 보고 최적 행동 a를 결정한다. 이 부분은 행동 a를 어떻게 결정할건지에 대한 것이다. 

`s1, r, done,_,_ =env.step(a)` : 위에서 행동 a를 결정하면 step 함수로 실행하여 다음상태 보상 에피소드의 끝여부를 얻는다.

그 다음 if문은 에피소드가 max_steps에 도달하지 못한 채 중간에 끝나면 실패했다고 간주(보상을 -100으로)

`D.append((s,a,r,s1,done))`: 새로 만든 샘플을 리플레이 메모리(D)에 추가함

그 밑에 if문에서는 model_learning()함수를 호출하여 신경망을 학습하는데 리플레이 메모리가 일정 크기가 되기 전에는 적용하지 않는다.(훈련 집합이 충분하지 않다고 판단)

if문 밑에 부분은 다음 상태를 현재 상태로 대치하고 누적 보상액을 갱신하는 부분이다.

```
if done:
            long_reward=long_reward if long_reward==max_steps else long_reward+100
            print(i,"번째 에피소드의 점수:",long_reward)
            scores.append(long_reward)
            break
```
에피소드가 끝난 경우를 처리한다. 에피소드가 중간에 실패로 끝냈다면 위에서 100만큼 삭감했기 떄문에, 몇 스텝인지 출력하기 위해 100을 다시 더해서 출력한다.

```
if i>10 and np.mean(scores[-5:])>(0.95*max_steps):
        break
```
멈춘 조건에 도달하지 못하더라도 수렴했다면 루프를 탈출한다. 여기서 수렴은 최근 에피소드 5개의 누적 보상액 평균이 최대 보상액의 95%를 초과하는 경우라고 가정했다.

처음 이 신경망학습 코드 for문 밑의 코드를 `s=env.reset()`로 했을 때에 `cannot reshape array of size 2 into shape(1,4)`라는 에러기 생겼다. 어레이 사이즈가 2인데, reshape(1,4)로 하면, 어레이 사이즈가 4여야 한다는 것을 깨달았고 문제가 된 부분은 `env.reset()` 부분인 것을 찾았다. 그래서 `s,_=env.reset()`로 바꿨더니 해결되었다. 


## **시뮬레이션**
처음에 20번째 에피소드까지는 0에서 50사이의 점수가 진동하는 형태로 전혀 학습이 되지 않는것 처럼 보이다가 45번째에피소드 이상부터는 안정적으로 200점 이상의 점수가 나왔다.
```
50 번째 에피소드의 점수: 257.0
51 번째 에피소드의 점수: 267.0
```
















