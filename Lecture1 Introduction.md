#### CS234 Reinforcement Learning Winter 2019
#### Lecture 1：Introduction (Sutton and Barton Chp 1)
##### 一、简介  
**1、	强化学习定义**
强化学习关注的是智能代理如何学会做出一系列良好决策的基础问题。因此首先需要关注的是决策序列。应用：网页推荐、医学治疗、机器人等。

**2、	与机器学习的联系**
机器学习(此处可以看作是智能agent/agents)做出一系列决策，而不是仅有一个决策，并判断决策的优劣，即做出一系列最优决策。   
在强化学习中agent并不能提前决策是怎样影响世界的，也不知道决策是否能产生好的结果，但它能够**通过经验获取信息**。  

**3、	例子：video games**  
视频游戏会让人类玩家经常学习一些复杂的任务，而这些任务是未知的，需要通过积累的经验来完成。而智能agent能够通过**直接学习视频的像素输入来玩游戏**。因此，从agent的角度来看，他们只能看到这些彩色像素的出现，并且通过**学习什么样的决定是正确的**来学会更好地玩游戏。  

**4、	重要特点** 
1) 优化(Optimization)  
目的：找到制定决策或至少找到一个良好的制定策略  
2)	延迟结果(Delayed results)  
决定会在一定时间过后产生影响。例如为退休所存储的存款；在游戏中提前收集对后续发展有利的钥匙。  
两个挑战：  
a)	策划(Planning): 即时利益和长期利益之间的权衡  
b)	学习(learning): 信用分配问题(credit assignment)，过去做出的决定与未来结果之间的因果关系，即找到产生高/底reward的因素。  
3)	探索(Exploration)    
将agent看作科学家，通过尝试和失败不断学习。而决策会影响我们学习到了什么。例如如果一个人决定去了MIT上学而不是Stanford，他就会有另一种经历，以为一个人不能同时过两种生活。  
4)	泛化(Generalization)    
通过policy实现，policy从经验到所采取的action的映射。    

**5、	几种学习方法**  
1)	AI planning  
缺少Exploration元素。该方法计算一系列好的决策，但是提前给定了决策影响世界的模型。  
2)	监督学习(Supervised learning)  
缺少exploration和delayed consequences元素。该方法是从经验中获取信息，但是已知数据的正确标签。  
3)	无监督学习(Unsupervised learning)  
缺少exploration和delayed consequences元素。该方法不能实现得到标签，但可以通过数据分布学习数据空间到标签空间的映射，但是所学的标签不一定是真正的标签。
4)	模仿学习(Imitation learning)
缺少exploration元素。该方法学习他人经验(人类专家等)中学习信息，提前假设了所学习的样本就是正确的policy，能够将强化学习问题简化为监督学习。  
优点：a) 监督学习的有利工具；b) 避免了exploration问题；c) 拥有关于决策结果的大批数据。  
限制：a) 代价太大； b) 受限于所收集的数据  
结论：模仿学习加上强化学习能够产生更好的结果。 

**6、	课程主要内容**  
1) 探索世界；
2) 利用经验来指导后续决策  

**7、	主要议题**  
1) Reward从哪儿来？Agent用来指导决策优劣的信息从哪儿来？那个元素提供这些信息？如果信息错误会发生什么？  
2) 鲁棒性和风险敏感性；  
3) 	Multi-agent 强化学习方法。  

##### 二、不确定下的序贯决策 (sequential decision-making)  
**1、	定义：**
一种**交互式的闭环过程**，我们有一些agent，一个智能agent采取影响state的行动，然后再给予回报观察 (observation)和奖励 (reward)，如图 1所示：  

 ![序贯决策](https://github.com/flyingfQiu/imgs/blob/master/seq_dec.jpg?raw=true''序贯决策'')

**2、	目的：**
选择能够最大化总未来期望奖励的行动 (action)。  

**3、	挑战：**
1) 保持当前reward和长期reward之间的平衡;
2) 通过一定的战略行为来得到高的奖励值，即可能需要牺牲当前的奖励来得到最终目的。 

例1：网页广告推荐  
	Observation: 浏览所花费时间；reward：广告点击量；action：选择广告；  
	目的：获得有关人们是否点击广告的信息，说明人们如何更多的点击广告  
例2：机器人洗碗机  
	Observation: 厨房照片；reward：+1，如果柜台上没有碗碟；  action：移动；
在这种情况下，通常会是一个延迟的奖励 (delayed reward)，当机器人将所有碗碟扫除之后或者撞到地板上柜台上的菜肴才会消失，但是撞到地板上并不是最终目的。所以，需要做出一系列决策，在这些决策节点上可能不能得到任何奖励。  
	例3：血压控制仪  
    Observation: 血压；reward：+1，如果在健康范围内，-0.05，药物有副作用，0，其他情况；action：运动或者服药；  
	检验对该问题的理解：(智能导师，Artificial tutor)  
    Observation: 学生给出的题目答案；reward：+1，答题正确，-0.05，大体错误；action：给出加减算术题；  
    Reward hacking：因为学生在做简单题目时得到的reward值高，所以tutor会一直给出简单题目。

**4、	过程简述**  
在每个时刻$t$：1) Agent采取action $a_{t}$; 2) World更新给定的 $a_{t}$，并产生observation $o_{t}$ 和reward $r_{t}$；3) Agent接收observation $o_{t}$和reward $r_{t}$。  

**5、	History的定义：**
过去的observation、actions和rewards的集合  
1) 标记：$h_{t}=\left(a_{1}, o_{1}, r_{1}, \cdots, a_{t}, o_{t}, r_{t}\right)$   
2) Agent基于History来选择action 
3) State是假设确定接下来会发生什么的信息，即History的函数：$s_{t}=\left(h_{t}\right)$。

##### 三、马尔科夫假设  
**1、	意义：**
马尔科夫假设agent所使用的**state是历史信息的充分统计量**，并且只需要当前状态就能够预测未来。  

**2、	定义：**
如果状态$s_{t}$满足下式，则具有马尔科夫性：   
$p\left(s_{t+1} | s_{t}, a_{t}\right)=p\left(s_{t+1} | h_{t}, a_{t}\right)$
**3、	特点：**
状态$s_{t}$包含了所有历史相关信息，因此给定当前状态，未来与过去无关。  
例：血压控制系统，state：当前血压值，action：是否服药。该模型不是马尔科夫的，原因是状态仅有当前的血压值，还需要知道其他特征，如是否就餐，是否运动等，都会影响后续时刻的状态。  

**4、	马尔科夫决策过程** 
假设观测(observation)为当前的状态，那么agent得到的是最新的观测结果，将其视作状态，agent对世界的建模即为马尔科夫决策过程。  
1）	全观测(MDP)：环境中所有的state都能被观测到$s_{t}=o_{t}$；  
2）	部分观测(POMDP): Agent state与World state不一致(如扑克牌游戏中只能看到自己的牌和已被丢弃的牌，但是能够通过对其他牌和玩家的信念状态来判断)；利用history $s_{t}=h_{t}$来建立Agent的状态，或者RNN等  

##### 四、	序贯决策类型  
**1、	Bandits：**
actions对后续的观测值没有影响；没有延迟奖励(delayed reward)；  
例子：网站向连续地向多个客户推送广告，无论向客户展示什么广告都不会影响下次登录网站的人，因此，做出的决定只会影响当前客户，与客户2完全独立。  

**2、	MDPs and POMDPs：**
actions对后续的观测值有影响；需要利用信用分配(credit assignment)和战略行为(strategic actions)  

**3、	How the world changes** 
a)	确定的(Deterministic)：单个观测和奖励  
b)	随机的(Stochastic)：多个可能的观测值和奖励。  

**4、	例子：火星车(Mars Rover)**
如图 2  
States：车的位置$\left(s_{1}, s_{2}, \cdots, s_{7}\right)$;  
Actions: 向左或者向右；  
Rewards：+1，在状态1($s_1$)处；+10，在状态7($s_7$)处；0，在其他所有状态处。
![](https://github.com/flyingfQiu/imgs/blob/master/mars_rover.jpg?raw=true)
图 2 火星车  
##### 五、	强化学习算法组成元素
**1、	模型(Model)：**
世界是怎样随着agent的action而变化的一种agent表征。
1)动态模型：预测下一agent状态：$p\left(s_{t+1}=s^{\prime} | s_{t}=s, a_{t}=a\right)$
2)模型：预测顺时奖励：$r\left(s_{t}=s, a_{t}=a\right)=E\left[r_{t} | s_{t}=s, a_{t}=a\right]$

**2、	策略(policy)：**
决定agent选择actions的方法，$\pi : S \rightarrow A$，state到actions的映射
确定性：$\pi(s)=a$
随机性：$\pi(a | s)=\operatorname{Pr}\left(a_{t}=a | s_{t}=s\right)$

**3、	价值(Value):**
特定策略下未来奖励的期望折现和：$V^{x}\left(s_{t}=s\right)=E_{\pi}\left[r_{t}+\gamma r_{t+1}+\gamma^{2} r_{t+2}+\cdots | s_{t}=s\right]$
   
折现系数(Discount factor)$\gamma$ 权衡即时奖励和未来奖励，能够用来量化states和actions的优劣，通过对比策略来决定行为。

##### 六、	强化学习Agents的类型
1) 基于模型：在agent中维持了世界运作方式的直接模型，可能有也可能没有policy或者value函数；
2) 无模型方法：有明确的value函数和policy函数，但是没有模型
![](https://github.com/flyingfQiu/imgs/blob/master/agent.jpg?raw=true) 
图 3 强化学习Agents

##### 七、	探索和利用(Exploration and Exploitation)
难点：**探索-利用困境**
探索(exploration)：尝试在未来能够使得agent做出更好决策的**新事物**，如尝试看新的电影，推送不同类型的广告等；
利用(exploitation)：**根据过去经验选择能够产生高reward的actions**，如选择欣赏之前最喜欢的电影，根据经验推送更有效的广告等。

##### 八、	评价和控制
评价(evaluation)：从一个给定的策略中评价/预测期望奖励；
控制(control)：利用优化找到最好的策略。
