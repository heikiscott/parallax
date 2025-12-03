# LoCoMo-All 评测详细分析报告

**评测时间**: 2024-12-03
**系统**: Parallax
**数据集**: LoCoMo (全部10个对话)

---

## 1. 总体结果

| 指标 | 数值 |
|------|------|
| 总问题数 | 1540 |
| 正确数 | 1419 |
| 错误数 | 121 |
| **准确率** | **92.19%** |

---

## 2. 按Category分类统计

LoCoMo数据集将问题分为4个Category：
- **Category 1**: 基础事实类问题（人物属性、基本信息）
- **Category 2**: 时间相关问题（事件发生时间）
- **Category 3**: 推理类问题（需要推断的问题）
- **Category 4**: 细节类问题（对话中的具体细节）

| Category | 正确数 | 总数 | 准确率 | 错误数 |
|----------|--------|------|--------|--------|
| Category 1 | 259 | 282 | 91.84% | 23 |
| Category 2 | 284 | 321 | 88.47% | 37 |
| Category 3 | 67 | 96 | **69.79%** | 29 |
| Category 4 | 809 | 841 | 96.20% | 32 |

### 关键发现
- **Category 3（推理类）准确率最低**，仅69.79%，显著低于其他类别
- **Category 4（细节类）表现最好**，达到96.20%
- **Category 2（时间相关）** 有37个错误，是错误最多的类别

---

## 3. 按对话(Conv)分布统计

LoCoMo数据集包含10个独立对话(conv0-conv9)，每个对话涉及不同的人物和场景。

### 3.1 各Conv准确率对比

| Conv | 人物 | 正确 | 错误 | 总数 | 准确率 | Cat1错 | Cat2错 | Cat3错 | Cat4错 |
|------|------|------|------|------|--------|--------|--------|--------|--------|
| conv0 | Caroline & Melanie | 143 | 9 | 152 | 94.1% | 4 | 4 | 1 | 0 |
| conv1 | Gina & Jon | 76 | 5 | 81 | 93.8% | 0 | 2 | 0 | 3 |
| conv2 | John & Maria | 144 | 8 | 152 | 94.7% | 0 | 3 | 3 | 2 |
| conv3 | Nate & Joanna | 178 | **21** | 199 | **89.4%** | 6 | 3 | 6 | 6 |
| conv4 | John & Tim | 156 | **22** | 178 | **87.6%** | 5 | 4 | 6 | 7 |
| conv5 | Andrew & Audrey | 111 | 12 | 123 | 90.2% | 0 | 4 | 4 | 4 |
| conv6 | John & James | 137 | 13 | 150 | 91.3% | 1 | 6 | 4 | 2 |
| conv7 | Jolene & Deborah | 179 | 12 | 191 | 93.7% | 3 | 3 | 1 | 5 |
| conv8 | Sam & Evan | 145 | 11 | 156 | 92.9% | 2 | 6 | 2 | 1 |
| conv9 | Dave & Calvin | 150 | 8 | 158 | 94.9% | 2 | 2 | 2 | 2 |

### 3.2 问题对话分析

**表现最差的对话：**

1. **conv4 (87.6%)** - John & Tim，22个错误
   - Cat4错误最多(7个)：大量细节类问题出错
   - Cat3错误较多(6个)：推理类问题表现差
   - 典型错题：John的favorite book series、Tim的语言、Ireland的Star Wars地点等

2. **conv3 (89.4%)** - Nate & Joanna，21个错误
   - 各类别错误较均匀分布
   - 数量统计类问题较多（tournaments、screenplays等）
   - 典型错题：Nate的gaming room灯光、Joanna的screenplay内容等

3. **conv5 (90.2%)** - Andrew & Audrey，12个错误
   - Cat3推理题出错较多(4个)：涉及地理位置推断
   - 典型错题：居住州(Minnesota)、国家公园名称等

**表现较好的对话：**

1. **conv9 (94.9%)** - Dave & Calvin，仅8个错误
2. **conv2 (94.7%)** - John & Maria，8个错误
3. **conv0 (94.1%)** - Caroline & Melanie，9个错误

### 3.3 各Conv错题详情

<details>
<summary>点击展开各Conv错题详细列表</summary>

#### conv0 (Caroline & Melanie) - 9个错题

| # | Cat | 问题 | 标准答案 |
|---|-----|------|---------|
| 1 | 2 | When did Caroline meet up with her friends, family, and mentors? | The week before 9 June 2023 |
| 2 | 1 | Where did Caroline move from 4 years ago? | Sweden |
| 3 | 2 | When did Melanie read the book "nothing is impossible"? | 2022 |
| 4 | 1 | What kind of art does Caroline make? | abstract art |
| 5 | 1 | What are Melanie's pets' names? | Oliver, Luna, Bailey |
| 6 | 2 | When did Caroline apply to adoption agencies? | The week of 23 August 2023 |
| 7 | 3 | Would Caroline be considered religious? | Somewhat, but not extremely religious |
| 8 | 1 | What items has Melanie bought? | Figurines, shoes |
| 9 | 2 | When did Melanie buy the figurines? | 21 October 2023 |

#### conv1 (Gina & Jon) - 5个错题

| # | Cat | 问题 | 标准答案 |
|---|-----|------|---------|
| 1 | 2 | When did Gina open her online clothing store? | 16 March, 2023 |
| 2 | 2 | When did Jon host a dance competition? | May, 2023 |
| 3 | 4 | What do the dancers in the photo represent? | They are performing at the festival |
| 4 | 4 | What did Gina design for her store? | the space, furniture, and decor |
| 5 | 4 | How does Gina describe the feeling that dance brings? | magical |

#### conv2 (John & Maria) - 8个错题

| # | Cat | 问题 | 标准答案 |
|---|-----|------|---------|
| 1 | 3 | What might John's financial status be? | Middle-class or wealthy |
| 2 | 3 | What might John's degree be in? | Political science |
| 3 | 3 | Does John live close to a beach or the mountains? | beach |
| 4 | 2 | When did John have his first firefighter call-out? | The Sunday before 3 July 2023 |
| 5 | 2 | When did Maria start volunteering at the homeless shelter? | Around August 2022 |
| 6 | 2 | When did Maria take up community work with her church friends? | August 4, 2023 |
| 7 | 4 | What did Maria donate to a homeless shelter in December 2023? | old car |
| 8 | 4 | What event did John volunteer at last weekend? | career fair at a local school |

#### conv3 (Nate & Joanna) - 21个错题 ⚠️

| # | Cat | 问题 | 标准答案 |
|---|-----|------|---------|
| 1 | 3 | What pets wouldn't cause any discomfort to Joanna? | Hairless cats or pigs |
| 2 | 3 | What nickname does Nate use for Joanna? | Jo |
| 3 | 1 | How many times has Joanna found new hiking trails? | twice |
| 4 | 1 | What book recommendations has Joanna given to Nate? | "Little Women", "A Court of Thorns and Roses" |
| 5 | 2 | When did Nate take time off to chill with his pets? | Weekend of 22 August, 2022 |
| 6 | 2 | When did Joanna make a chocolate tart with raspberries? | 5 October, 2022 |
| 7 | 2 | How long did it take for Joanna to finish writing her book? | four months |
| 8 | 1 | How many of Joanna's writing have made it to the big screen? | two |
| 9 | 1 | How many times has Nate taken his turtles on a walk? | Twice |
| 10 | 3 | What alternative career might Nate consider after gaming? | animal keeper at zoo |
| 11 | 3 | What state did Joanna visit in summer 2021? | Indiana |
| 12 | 1 | How many video game tournaments has Nate participated in? | nine |
| 13 | 1 | How many screenplays has Joanna written? | three |
| 14 | 3 | Was the first half of September 2022 good career-wise for Nate and Joanna? | No |
| 15 | 3 | What state did Nate visit? | Florida |
| 16 | 4 | What is Nate's favorite book series about? | dragons |
| 17 | 4 | What kind of lighting does Nate's gaming room have? | red and purple lighting |
| 18 | 4 | What is Joanna's third screenplay about? | loss, identity, and connection |
| 19 | 4 | What does Nate feel he could do at Whispering Falls? | write a whole movie |
| 20 | 4 | What game did Nate play at game convention on 9 October? | Catan |
| 21 | 4 | What does Nate want to do when he goes to Joanna's place? | Watch movie together or go to park |

#### conv4 (John & Tim) - 22个错题 ⚠️

| # | Cat | 问题 | 标准答案 |
|---|-----|------|---------|
| 1 | 2 | In which month's game did John achieve a career-high score? | June 2023 |
| 2 | 3 | Which outdoor gear company likely signed up John? | Under Armour |
| 3 | 1 | What sports does John like besides basketball? | surfing |
| 4 | 1 | How many games has John mentioned winning? | 6 |
| 5 | 2 | Which city was John in before traveling to Chicago? | Seattle |
| 6 | 1 | When did John get an ankle injury in 2023? | around November 16, 2023 |
| 7 | 1 | How many times has John injured his ankle? | two times |
| 8 | 1 | Which book was John reading during his recovery? | The Alchemist |
| 9 | 3 | What kind of yoga for building core strength might John benefit from? | Hatha Yoga |
| 10 | 3 | What other exercises can help John with basketball performance? | Sprinting, long-distance running, boxing |
| 11 | 2 | When did Tim start playing the violin? | August 2023 |
| 12 | 2 | When did John achieve a career-high assist performance? | December 11, 2023 |
| 13 | 3 | What is a Star Wars book that Tim might enjoy? | Star Wars: Jedi Apprentice |
| 14 | 3 | What would be a good hobby for Tim related to travel dreams? | Writing a travel blog |
| 15 | 3 | Which Star Wars-related locations would Tim enjoy in Ireland? | Skellig Michael, Malin Head, etc. |
| 16 | 4 | What did John share with the person he skyped about? | Characters from Harry Potter |
| 17 | 4 | What did Tim say about his injury on 16 November? | The doctor said it's not too serious |
| 18 | 4 | What is the topic of discussion on 11 December? | Academic and sports successes |
| 19 | 4 | What language does Tim know besides German? | Spanish |
| 20 | 4 | What book did Tim get in Italy that inspired him to cook? | a cooking book |
| 21 | 4 | What is John's favorite book series? | Harry Potter |
| 22 | 4 | What kind of painting does John have as a reminder? | a painting of Aragorn |

#### conv5 (Andrew & Audrey) - 12个错题

| # | Cat | 问题 | 标准答案 |
|---|-----|------|---------|
| 1 | 2 | When did Audrey make muffins for herself? | The week of April 3rd to 9th |
| 2 | 2 | When did Audrey see a hummingbird? | first week of May 2023 |
| 3 | 3 | What can Andrew do to improve his stress and accommodate his dogs? | Change to hybrid/remote job |
| 4 | 3 | Which US state do Audrey and Andrew potentially live in? | Minnesota |
| 5 | 3 | Which national park could they be referring to? | Voyageurs National Park |
| 6 | 2 | How many pets did Andrew have, as of September 2023? | one |
| 7 | 3 | What could Andrew do to make birdwatching fit in his city schedule? | Install a bird feeder |
| 8 | 2 | When did Andrew make his dogs a fun indoor area? | few days before November 22, 2023 |
| 9 | 4 | What kind of pastries did Andrew and his girlfriend have? | croissants, muffins, tarts |
| 10 | 4 | What did Andrew and his GF do on Monday before July 24? | volunteered at pet shelter |
| 11 | 4 | What did Andrew and Audrey plan to do on Saturday after October 28? | Go hiking |
| 12 | 4 | What did Audrey share to show ways to keep dogs active? | photo of basket full of stuffed animals |

#### conv6 (John & James) - 13个错题

| # | Cat | 问题 | 标准答案 |
|---|-----|------|---------|
| 1 | 3 | What are John's suspected health problems? | Obesity |
| 2 | 1 | What are John and James' favorite games? | CS:GO, Apex Legends |
| 3 | 3 | Does James live in Connecticut? | Likely yes |
| 4 | 2 | How was John feeling on April 10, 2022? | seeking solitude |
| 5 | 3 | What is the board game where you find the imposter? | Mafia |
| 6 | 2 | How many days did James plan to spend in Canada? | 19 days |
| 7 | 3 | Did John and James study together? | Yes |
| 8 | 2 | When did James try Cyberpunk 2077? | October 20, 2022 |
| 9 | 2 | What was James' big moment with Samantha in October 2023? | They decided to live together |
| 10 | 2 | How long did James and Samantha date before moving in together? | nearly three months |
| 11 | 2 | When did John work with a game developer? | November 5-6, 2022 |
| 12 | 4 | What is the name of the board game John tried in September 2022? | Dungeons of the Dragon |
| 13 | 4 | What sparked James' passion for gaming as a kid? | Super Mario and Zelda games |

#### conv7 (Jolene & Deborah) - 12个错题

| # | Cat | 问题 | 标准答案 |
|---|-----|------|---------|
| 1 | 2 | When do Jolene and her partner plan to complete "Walking Dead"? | Saturday after 27 January, 2023 |
| 2 | 2 | When did Jolene finish her robotics project? | May 2023 |
| 3 | 2 | When did Deborah go to a community meetup? | last week of August 2023 |
| 4 | 3 | What card game is Deborah talking about? | Exploding Kittens |
| 5 | 1 | Where did Jolene and her partner find a cool diving spot? | Phuket |
| 6 | 1 | What gifts has Deborah received? | appreciation letter, flower bouquet, motivational quote |
| 7 | 1 | Which countries has Deborah traveled to? | Thailand, Brazil |
| 8 | 4 | When did Jolene buy her pet snake? | A year ago |
| 9 | 4 | What cool stuff did Jolene accomplish at retreat on 9 February? | Neat solutions for engineering project |
| 10 | 4 | What activity does Deborah incorporate after morning jog? | spending time with loved ones |
| 11 | 4 | What did Deb share a photo of that brought smile to Jolene? | yellow coffee cup with handwritten message |
| 12 | 4 | What did Jolene recently play that she described to Deb? | a card game about cats |

#### conv8 (Sam & Evan) - 11个错题

| # | Cat | 问题 | 标准答案 |
|---|-----|------|---------|
| 1 | 2 | Which hobby did Sam take up in May 2023? | painting |
| 2 | 2 | When did Evan have his heart palpitation incident? | first week of June 2023 |
| 3 | 3 | What would be an appropriate gift for Evan and Sam? | healthy cookbook or meal delivery |
| 4 | 2 | What significant event happened in Sam's life end of summer 2023? | He fell in love with a Canadian woman |
| 5 | 2 | When did Evan get back from vacation with his SO? | August 13, 2023 |
| 6 | 3 | How often does Sam get health checkups? | every three months |
| 7 | 1 | What personal health incidents does Evan face in 2023? | heart palpitations, twisted ankle (x2) |
| 8 | 2 | When did Evan's son fall off his bike? | Thursday before December 17, 2023 |
| 9 | 2 | When did Evan have a drunken night with his friends? | January 9, 2023 |
| 10 | 1 | What is a stress reliever for Sam? | Unhealthy snacks, sweets, yoga, beautiful views |
| 11 | 4 | What did Evan share with Sam after their hiking trip? | photo of man standing on rock over valley |

#### conv9 (Dave & Calvin) - 8个错题

| # | Cat | 问题 | 标准答案 |
|---|-----|------|---------|
| 1 | 3 | Does Dave's shop employ a lot of people? | Yes |
| 2 | 2 | When did Dave start his car maintenance shop? | May 1, 2023 |
| 3 | 2 | Which city was Calvin visiting in August 2023? | Miami |
| 4 | 3 | Did Calvin and Dave have a meeting in Boston Aug-Nov 2023? | No |
| 5 | 1 | How many car shows has Dave attended? | two |
| 6 | 1 | What gifts has Calvin received from his artist friends? | gold chain, custom-made guitar with octopus |
| 7 | 4 | When did Calvin first get interested in cars? | at an early age |
| 8 | 4 | What hobby did Calvin take up recently? | Photography |

</details>

---

## 4. 错误原因分类

将121个错题按错误表现分为4类：

| 错误类型 | 数量 | 占比 | 描述 |
|---------|------|------|------|
| 检索失败 | 21 | 17.4% | 模型回答"没有相关信息" |
| 时间/日期错误 | 24 | 19.8% | 时间推理偏差 |
| 数量统计错误 | 13 | 10.7% | 计数或统计错误 |
| 信息混淆/错误 | 63 | 52.1% | 信息理解错误或混淆 |

---

## 4. 检索质量分析

### 4.1 检索充分性

| 检索状态 | 错题数量 | 占比 |
|---------|---------|------|
| 检索不充分 | 95 | 78.5% |
| 检索充分但回答错误 | 26 | 21.5% |

**发现**: 近80%的错题是因为检索阶段没有找到足够的相关信息。

### 4.2 按检索策略统计正确率

系统根据问题类型自动选择3种检索策略：

- **gec_insert_after_hit**: 用于 `general` 类型问题，基于GEC聚类后在命中点后插入相关内容
- **gec_cluster_rerank**: 用于事件类问题(`event_temporal`, `event_aggregation`, `event_activity`)，基于GEC聚类重排序
- **agentic_only**: 用于属性类问题(`attribute_*`, `time_calculation`)，纯agentic检索

#### 4.2.1 各策略总体正确率

| 策略 | 正确数 | 总数 | 准确率 | 错题数 |
|------|--------|------|--------|--------|
| gec_insert_after_hit | 1286 | 1388 | **92.65%** | 102 |
| agentic_only | 38 | 41 | **92.68%** | 3 |
| gec_cluster_rerank | 95 | 111 | **85.59%** | 16 |

**关键发现**:

- `agentic_only` 和 `gec_insert_after_hit` 准确率相当（~92.7%），表现较好
- `gec_cluster_rerank` 准确率偏低（85.59%），比其他策略低约7个百分点
- `gec_insert_after_hit` 处理了绝大多数问题（90.1%），是主力策略

> ⚠️ **重要说明：准确率差异的本质原因**
>
> `gec_cluster_rerank` 准确率较低**不是因为策略本身差**，而是因为它被分配处理**最难的问题类型**：
>
> | 问题类型 | 难度 | 策略 | 准确率 |
> |----------|------|------|--------|
> | event_temporal | ⭐⭐⭐ 需要精确时间推理 | gec_cluster_rerank | 86.17% |
> | event_aggregation | ⭐⭐⭐⭐ 需要遍历+计数 | gec_cluster_rerank | 76.92% |
> | general | ⭐ 找到信息即可 | gec_insert_after_hit | 92.62% |
>
> **典型错题模式**：
>
> - 时间推理错误：如"When did John have his first firefighter call-out?" 标准答案是"The Sunday before 3 July 2023"，模型答"July 23, 2023"
> - 统计漏计：如"How many times has Nate taken his turtles on a walk?" 标准答案2次，模型答1次
> - 检索失败：如"When did Melanie read the book?" 模型称"没有相关信息"
>
> **结论**：不同策略处理的问题难度不同，直接比较准确率不公平。真正需要改进的是**时间推理**和**事件聚合/计数**能力。

#### 4.2.2 各策略在不同Category的表现

##### gec_insert_after_hit 策略 (1388题)

| Category | 正确 | 总数 | 准确率 |
|----------|------|------|--------|
| Category 1 | 237 | 256 | 92.58% |
| Category 2 | 200 | 224 | 89.29% |
| Category 3 | 64 | 93 | **68.82%** |
| Category 4 | 785 | 815 | 96.32% |

**问题**: 该策略在Category 3（推理类）问题上表现很差，准确率仅68.82%

##### gec_cluster_rerank 策略 (111题)

| Category | 正确 | 总数 | 准确率 |
|----------|------|------|--------|
| Category 1 | 15 | 18 | 83.33% |
| Category 2 | 79 | 92 | 85.87% |
| Category 4 | 1 | 1 | 100% |

**问题**: 该策略主要处理Category 2（时间相关）问题，但准确率仅85.87%

##### agentic_only 策略 (41题)

| Category | 正确 | 总数 | 准确率 |
|----------|------|------|--------|
| Category 1 | 7 | 8 | 87.50% |
| Category 2 | 5 | 5 | 100% |
| Category 3 | 3 | 3 | 100% |
| Category 4 | 23 | 25 | 92.00% |

**表现**: 该策略样本量较小，但整体表现均衡

#### 4.2.3 错题检索充分性分析

| 策略 | 错题数 | 检索不充分数 | 不充分比例 |
|------|--------|-------------|-----------|
| gec_insert_after_hit | 102 | 83 | 81.4% |
| gec_cluster_rerank | 16 | 9 | 56.3% |
| agentic_only | 3 | 3 | 100% |

**发现**: `gec_insert_after_hit` 策略的错题最多，且检索不充分比例高达81.4%。

### 4.3 按问题类型(question_type)统计正确率

系统根据问题模式将问题分为9种类型：

| 问题类型 | 正确数 | 总数 | 准确率 | 使用策略 |
|----------|--------|------|--------|----------|
| attribute_identity | 8 | 8 | **100%** | agentic_only |
| event_activity | 4 | 4 | **100%** | gec_cluster_rerank |
| reasoning_hypothetical | 6 | 6 | **100%** | gec_insert_after_hit |
| time_calculation | 13 | 13 | **100%** | agentic_only |
| general | 1280 | 1382 | 92.62% | gec_insert_after_hit |
| attribute_preference | 16 | 18 | 88.89% | agentic_only |
| event_temporal | 81 | 94 | **86.17%** | gec_cluster_rerank |
| event_aggregation | 10 | 13 | **76.92%** | gec_cluster_rerank |
| attribute_location | 1 | 2 | 50.00% | agentic_only |

**关键发现**:

- 4种问题类型达到100%准确率：`attribute_identity`, `event_activity`, `reasoning_hypothetical`, `time_calculation`
- **`event_temporal`（时间事件）准确率仅86.17%**，是错误高发区
- **`event_aggregation`（事件聚合/统计）准确率最低**，仅76.92%
- `general` 类型问题占绝大多数（89.7%），准确率92.62%

---

## 5. 典型错误案例分析

### 5.1 检索失败类（模型称"没有信息"）

| # | 问题 | 标准答案 | 问题分析 |
|---|------|---------|---------|
| 1 | Where did Caroline move from 4 years ago? | Sweden | 信息可能分散在对话中，检索未找到 |
| 2 | When did Melanie read the book "nothing is impossible"? | 2022 | 具体事件细节检索失败 |
| 3 | What items has Melanie bought? | Figurines, shoes | 购买记录分散，未能聚合 |
| 4 | What state did Nate visit? | Florida | 地理信息检索失败 |
| 5 | Which outdoor gear company likely signed up John? | Under Armour | 具体品牌名检索失败 |

### 5.2 时间推理错误类

| # | 问题 | 标准答案 | 模型答案 | 分析 |
|---|------|---------|---------|------|
| 1 | When did Caroline apply to adoption agencies? | The week of 23 August 2023 | August 14-20, 2023 | 对"earlier that week"理解错误 |
| 2 | When did Gina open her online clothing store? | 16 March, 2023 | Before January 29, 2023 | 时间推理方向错误 |
| 3 | When did Nate take time off to chill with his pets? | Weekend of 22 August, 2022 | August 27-28, 2022 | 周末时间偏移 |
| 4 | When did Joanna make a chocolate tart? | 5 October, 2022 | Before September 14, 2022 | 分享时间≠制作时间 |

### 5.3 数量统计错误类

| # | 问题 | 标准答案 | 模型答案 | 分析 |
|---|------|---------|---------|------|
| 1 | How many video game tournaments has Nate participated in? | 9 | 7 | 统计不完整 |
| 2 | How many screenplays has Joanna written? | 3 | 4 | 过度计数 |
| 3 | How many times has Nate taken his turtles on a walk? | 2 | 1 | 漏计 |
| 4 | How many car shows has Dave attended? | 2 | 3 | 过度计数 |

### 5.4 信息混淆/错误类

| # | 问题 | 标准答案 | 模型答案 | 分析 |
|---|------|---------|---------|------|
| 1 | What are Melanie's pets' names? | Oliver, Luna, Bailey | Oliver, Bailey | 漏掉Luna |
| 2 | What kind of art does Caroline make? | abstract art | LGBTQ advocacy art | 回答角度错误 |
| 3 | What might John's degree be in? | Political science | Mechanical engineering | 推理错误 |
| 4 | What is Nate's favorite book series about? | dragons | Lord of the Rings | 混淆不同系列 |
| 5 | What sports does John like besides basketball? | surfing | soccer | 信息混淆 |

---

## 6. 根本原因分析

### 6.1 检索层面问题

1. **general类型问题策略匹配不佳**
   - 102个错题中有102个是general类型
   - 全部使用 `gec_insert_after_hit` 策略
   - 81.4%检索不充分

2. **关键词匹配不足**
   - 专有名词（宠物名、书名、品牌名）检索命中率低
   - 分散信息未能有效聚合

3. **时间语义理解弱**
   - "earlier that week"、"last weekend" 等相对时间表述处理不当

### 6.2 生成层面问题

1. **信息遗漏**
   - 检索到多条相关信息时，部分信息被忽略（如Luna）

2. **推理能力不足**
   - Category 3准确率仅69.79%
   - 对需要综合推断的问题表现差

3. **过度推断**
   - 部分问题模型给出了过于详细但错误的答案
   - 如将mechanical engineering误判为John的专业

### 6.3 数据/标注层面

1. **时间标注歧义**
   - Memunit中使用相对时间表述可能导致歧义
   - 如"earlier that week"的锚点不明确

---

## 7. 改进建议

### 7.1 检索优化（高优先级）

| 改进项 | 具体措施 | 预期收益 |
|--------|---------|---------|
| 优化general类型策略 | 对general问题使用更激进的多轮检索策略 | 减少81个检索不充分错题 |
| 增强专有名词匹配 | 建立实体词典，强化宠物名/书名/品牌名检索 | 提高检索召回率 |
| 改进时间检索 | 对时间相关问题增加时间范围扩展检索 | 减少时间类错误 |

### 7.2 生成优化（中优先级）

| 改进项 | 具体措施 | 预期收益 |
|--------|---------|---------|
| 信息聚合检查 | 对列举类问题增加完整性验证 | 避免遗漏信息 |
| 推理增强 | 对Category 3问题使用CoT推理 | 提高推理准确率 |
| 时间解析优化 | 在prompt中强调绝对时间计算 | 减少时间偏差 |

### 7.3 数据层面（低优先级）

| 改进项 | 具体措施 | 预期收益 |
|--------|---------|---------|
| 时间标注标准化 | Memunit中使用绝对时间而非相对时间 | 减少歧义 |
| 充分性判断校准 | 调整is_sufficient的判断阈值 | 提高判断准确性 |

---

## 8. 优先修复项

根据影响范围和修复成本，建议优先处理：

1. **【P0】优化general类型问题的检索策略**
   - 影响：102个错题（84.3%）
   - 方案：改进 `gec_insert_after_hit` 策略或为general类型问题分配更合适的策略

2. **【P1】增强时间语义理解**
   - 影响：24个时间相关错题
   - 方案：在时间解析模块增加相对时间转绝对时间的逻辑

3. **【P2】改进推理类问题处理**
   - 影响：29个Category 3错题
   - 方案：对推理类问题使用更复杂的推理链

---

## 附录：全部121道错题列表

<details>
<summary>点击展开完整错题列表</summary>

### Category 1 错题（23道）

1. Where did Caroline move from 4 years ago? → Sweden
2. What kind of art does Caroline make? → abstract art
3. What are Melanie's pets' names? → Oliver, Luna, Bailey
4. How many times has Joanna found new hiking trails? → twice
5. What book recommendations has Joanna given to Nate? → "Little Women", 'A Court of Thorns and Roses'
6. How many times has Nate taken his turtles on a walk? → Twice
7. How many video game tournaments has Nate participated in? → nine
8. How many screenplays has Joanna written? → three
9. What sports does John like besides basketball? → surfing
10. How many games has John mentioned winning? → 6
11. When did John get an ankle injury in 2023? → around November 16, 2023
12. How many times has John injured his ankle? → two times
13. Which book was John reading during his recovery? → The Alchemist
14. What are John and James' favorite games? → CS:GO, Apex Legends
15. Where did Jolene and her partner find a cool diving spot? → Phuket
16. What gifts has Deborah received? → appreciation letter, flower bouquet, motivational quote
17. Which countries has Deborah traveled to? → Thailand, Brazil
18. What is a stress reliever for Sam? → Unhealthy snacks, sweets, yoga, places with beautiful views
19. What personal health incidents does Evan face in 2023? → heart palpitations, twisted ankle (x2)
20. How many car shows has Dave attended? → two
21. What gifts has Calvin received from his artist friends? → gold chain, custom-made guitar with octopus
22. What items has Melanie bought? → Figurines, shoes
23. How many of Joanna's writing have made it to the big screen? → two

### Category 2 错题（37道）

1. When did Caroline meet up with her friends, family, and mentors? → The week before 9 June 2023
2. When did Melanie read the book "nothing is impossible"? → 2022
3. When did Caroline apply to adoption agencies? → The week of 23 August 2023
4. When did Melanie buy the figurines? → 21 October 2023
5. When did Gina open her online clothing store? → 16 March, 2023
6. When did Jon host a dance competition? → May, 2023
7. When did John have his first firefighter call-out? → The Sunday before 3 July 2023
8. When did Maria start volunteering at the homeless shelter? → Around August 2022
9. When did Maria take up community work with her church friends? → August 4, 2023
10. When did Nate take time off to chill with his pets? → Weekend of 22 August, 2022
11. When did Joanna make a chocolate tart with raspberries? → 5 October, 2022
12. How long did it take for Joanna to finish writing her book? → four months
13. In which month's game did John achieve a career-high score? → June 2023
14. When did Tim start playing the violin? → August 2023
15. When did John achieve a career-high assist performance? → December 11, 2023
16. When did Audrey make muffins for herself? → The week of April 3rd to 9th
17. When did Audrey see a hummingbird? → first week of May 2023
18. How many pets did Andrew have, as of September 2023? → one
19. When did Andrew make his dogs a fun indoor area? → few days before November 22, 2023
20. When do Jolene and her partner plan to complete "Walking Dead"? → Saturday after 27 January, 2023
21. When did Jolene finish her robotics project? → May 2023
22. When did Deborah go to a community meetup? → last week of August 2023
23. Which hobby did Sam take up in May 2023? → painting
24. When did Evan have his heart palpitation incident? → first week of June 2023
25. What significant event happened in Sam's life end of summer 2023? → He fell in love with a Canadian woman
26. When did Evan get back from a vacation with his SO? → August 13, 2023
27. When did Evan's son fall off his bike? → Thursday before December 17, 2023
28. When did Evan have a drunken night with his friends? → January 9, 2023
29. When did Dave start his car maintenance shop? → May 1, 2023
30. Which city was Calvin visiting in August 2023? → Miami
31. How was John feeling on April 10, 2022? → seeking solitude
32. How many days did James plan to spend on his trip in Canada? → 19 days
33. When did James try Cyberpunk 2077 game? → October 20, 2022
34. What was James' big moment with Samantha in October 2023? → They decided to live together
35. How long did James and Samantha date before moving in together? → nearly three months
36. When did John work with a game developer on a project? → November 5-6, 2022
37. Which city was John in before traveling to Chicago? → Seattle

### Category 3 错题（29道）

1. Would Caroline be considered religious? → Somewhat, but not extremely religious
2. What might John's financial status be? → Middle-class or wealthy
3. What might John's degree be in? → Political science
4. Does John live close to a beach or the mountains? → beach
5. What pets wouldn't cause any discomfort to Joanna? → Hairless cats or pigs
6. What nickname does Nate use for Joanna? → Jo
7. What alternative career might Nate consider after gaming? → animal keeper at zoo
8. What state did Joanna visit in summer 2021? → Indiana
9. Was first half of September 2022 good career-wise for Nate and Joanna? → No
10. What state did Nate visit? → Florida
11. Which outdoor gear company likely signed up John? → Under Armour
12. What kind of yoga for building core strength might John benefit from? → Hatha Yoga
13. What other exercises can help John with basketball performance? → Sprinting, long-distance running, boxing
14. What is a Star Wars book that Tim might enjoy? → Star Wars: Jedi Apprentice
15. What would be a good hobby related to travel dreams for Tim? → Writing a travel blog
16. Which Star Wars-related locations would Tim enjoy in Ireland? → Skellig Michael, etc.
17. What can Andrew do to improve stress and accommodate his dogs? → Change to hybrid/remote job
18. Which US state do Audrey and Andrew potentially live in? → Minnesota
19. Which national park could Audrey and Andrew be referring to? → Voyageurs National Park
20. What could Andrew do to make birdwatching fit in his city schedule? → Install a bird feeder
21. What are John's suspected health problems? → Obesity
22. Does James live in Connecticut? → Likely yes
23. What is the board game where you find the imposter John mentions? → Mafia
24. Did John and James study together? → Yes
25. What card game is Deborah talking about? → Exploding Kittens
26. In light of health changes, what gift for both Evan and Sam? → healthy cookbook or meal delivery
27. How often does Sam get health checkups? → every three months
28. Does Dave's shop employ a lot of people? → Yes
29. Did Calvin and Dave have a meeting in Boston Aug-Nov 2023? → No

### Category 4 错题（32道）

1. What do the dancers in the photo represent? → performing at the festival
2. What did Gina design for her store? → the space, furniture, and decor
3. How does Gina describe the feeling that dance brings? → magical
4. What did Maria donate to a homeless shelter in December 2023? → old car
5. What event did John volunteer at last weekend? → career fair at a local school
6. What is Nate's favorite book series about? → dragons
7. What kind of lighting does Nate's gaming room have? → red and purple lighting
8. What is Joanna's third screenplay about? → loss, identity, and connection
9. What does Nate feel he could do at Whispering Falls? → write a whole movie
10. What game did Nate play at game convention on 9 October? → Catan
11. What does Nate want to do when he goes to Joanna's place? → Watch movie together or go to park
12. What did John share with the person he skyped about? → Characters from Harry Potter
13. What did Tim say about his injury on 16 November? → The doctor said it's not too serious
14. What is the topic of discussion between John and Tim on 11 December? → Academic and sports successes
15. What language does Tim know besides German? → Spanish
16. What book did Tim get in Italy that inspired him to cook? → a cooking book
17. What is John's favorite book series? → Harry Potter
18. What kind of painting does John have as a reminder? → a painting of Aragorn
19. What kind of pastries did Andrew and his girlfriend have? → croissants, muffins, tarts
20. What did Andrew and his GF do on the Monday before July 24? → volunteered at a pet shelter
21. What did Andrew and Audrey plan to do on Saturday after October 28? → Go hiking
22. What did Audrey share to show ways to keep dogs active? → photo of basket full of stuffed animals
23. When did Jolene buy her pet snake? → A year ago
24. What cool stuff did Jolene accomplish at retreat on 9 February? → Neat solutions for engineering project
25. What activity does Deborah incorporate after morning jog? → spending time with loved ones
26. What did Deb share a photo of that brought smile to Jolene? → yellow coffee cup with handwritten message
27. What did Jolene recently play that she described to Deb? → a card game about cats
28. What did Evan share with Sam after their hiking trip? → photo of man standing on rock over valley
29. What sparked James' passion for gaming when he was a kid? → Super Mario and Zelda games
30. What is the name of the board game John tried in September 2022? → Dungeons of the Dragon
31. When did Calvin first get interested in cars? → at an early age
32. What hobby did Calvin take up recently? → Photography

</details>

---

*报告生成时间: 2024-12-03*
