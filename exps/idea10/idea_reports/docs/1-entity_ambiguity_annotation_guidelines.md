# 实体模糊标注指南

本文档用于指导人工标注员判断一个 `Text Query` 是否存在 `entity ambiguity`，并在存在时赋予唯一类别标签。本文档与当前 LLM 标注 prompt 保持一致，目标是提高人工标注的一致性。

## 1. 标注任务

每条样本至少包含两个字段：

- `Original Question`
- `Text Query`

你的任务是判断：

1. `Text Query` 是否已经足够明确，能让一个纯文本检索器知道“应该检索哪个实体/对象/概念”
2. 如果不够明确，属于哪一种实体模糊

## 2. 核心判断标准

请始终先问自己下面这个问题：

> 不看图，只看 `Text Query`，一个文本检索器是否已经知道要检索哪个实体、对象、地点、产品、物种、组织、事件或概念？

- 如果答案是“是”，标 `entity_ambiguous = No`
- 如果答案是“否”，标 `entity_ambiguous = Yes`

这一步是整个任务里最重要的判断。

## 3. 一个最重要的原则

不要把“问题难回答”误判成“实体模糊”。

下面两种情况要分开：

- 检索目标很清楚，但答案可能难找
- 连要检索哪个目标都不清楚

只有第二种才是 `entity ambiguity`。

例如：

- `When was Birkenstock founded?`
  检索目标很清楚，是 `Birkenstock`，所以标 `No`
- `Where is it located?`
  `it` 没有明确指向哪个实体，所以标 `Yes`

## 4. 标注流程

建议严格按下面顺序判断。

### 第一步：判断是否实体模糊

标 `No` 的典型情况：

- Query 里直接点名了目标实体
- Query 虽然简短，但检索目标已经足够明确
- Query 只是宽泛、笼统、答案难找，但并不依赖图像来确定检索对象

例子：

- `Address of Notre-Dame de Paris` -> `No`
- `temperature in Tillamook` -> `No`
- `age requirement for commercial driver's license (CDL) in the U.S.` -> `No`

标 `Yes` 的典型情况：

- Query 里用代词指代目标，但没有可用先行词
- Query 依赖图像才能知道“到底是哪一个东西”
- Query 只给了一个过于模糊的描述，仍然无法唯一确定目标
- Query 本质上要求检索器先做图像识别，再去检索

例子：

- `What is the name of the fish in the picture?` -> `Yes`
- `Where is the sculpture located?` -> `Yes`
- `Where is it?` -> `Yes`

### 第二步：如果是 `Yes`，再分配唯一子类

四个子类是：

- `Object Identification`
- `Indirect Entity Ambiguity`
- `Description`
- `No Object Involved`

建议使用以下判定顺序：

1. 是不是在问“这是什么/叫什么”
2. 如果不是，是不是在问“关于这个未解析实体的进一步信息”
3. 如果不是，是不是通过描述语来指向一个仍未确定的对象
4. 如果都不是，再考虑 `No Object Involved`

## 5. 四类标签定义

### 5.1 `Object Identificattion`

定义：

Query 的本质任务是先识别目标对象/实体本身是谁、是什么。也就是说，在真正检索之前，系统必须先解决“图里这个东西到底是什么”。

一个非常有效的测试方法：

> 这个 Query 能不能被化归成 `Identify the object/entity in the image` 或者 `What is this thing?`

如果可以，通常就是 `Object Identificattion`。

典型特征：

- `identify`
- `what is this`
- `what species/breed/model is this`
- `name of the X in the image`
- `identification from image / picture / photo`

例子：

- `fish species identification from image`
- `Identify the sculpture in the image`
- `plant identification from the image`
- `name of the vineyard in the image`
- `What's the name of it?`

注意：

- 这个类别关注的是“识别实体本身”
- 如果 Query 不是问它是谁，而是问它的属性、位置、设计者、价格、历史等进一步信息，就不要标这个类

### 5.2 `Indirect Entity Ambiguity`

定义：

Query 不是在问“这个东西是什么”，而是在问“关于这个未解析实体的进一步信息”。问题的语义目标是某个属性、关系、历史、位置、价格、功能、时间、设计者等，但实体本身仍然没有被有效指明。

这类 Query 的本质是：

- 不是识别任务
- 但又依赖无效指代或未解析实体

典型特征：

- 代词：`it`, `they`, `this`, `that`
- 泛指名词：`the building`, `the flower`, `the company`
- 进一步追问：`who/when/where/how much/what feature/...`

例子：

- `when was it created?`
- `Who designed the building?`
- `Where is the flower planted in`
- `Who designed it?`
- `How much does it cost?`
- `Where is it located?`

和 `Object Identificattion` 的关键区别：

- `Object Identificattion`：问“它是谁/它叫什么”
- `Indirect Entity Ambiguity`：问“关于它的进一步信息是什么”

可以用下面这条规则区分：

> 如果 Query 本身是在问“这到底是什么”，选 `Object Identificattion`  
> 如果 Query 是在问“这个未解析实体的进一步信息”，选 `Indirect Entity Ambiguity`

### 5.3 `Description`

定义：

Query 通过一个带有一定细节的描述短语来指向目标，但这个描述仍不足以让纯文本检索器可靠地确定目标实体。

这类 Query 不是纯代词，也不是明确的识别命令，而是“带描述的模糊指向”。

典型形式：

- `the X with ...`
- `the business / river / building / statue whose ...`
- 通过若干视觉或上下文特征描述目标

例子：

- `river in the image with a building across from it`
- `business name associated with the image of makeup brushes and beauty products`
- `name of the statue in the Louvre Museum`

注意：

- 如果描述已经足够唯一并能支持检索，应标 `No`
- 如果描述仍然不足以唯一确定目标，才标 `Description`

### 5.4 `No Object Involved`

定义：

极少使用。只有在模糊性并不真正围绕某个具体对象或实体，而是围绕背景语境、角色、关系、抽象内容或非对象目标时，才考虑这个标签。

这个类别是最后兜底项。

使用原则：

- 只有当前三个正类都不合适时再用
- 不要把普通难例都扔进这个类

## 6. 重要边界规则

### 6.1 `No` vs `Yes`

问自己：

> 检索器在真正开始检索前，是否已经知道应该搜谁？

- 如果知道，标 `No`
- 如果不知道，标 `Yes`

### 6.2 `Object Identificattion` vs `Indirect Entity Ambiguity`

问自己：

> Query 是在问“这个实体是什么”，还是在问“关于这个未解析实体的进一步信息”？

- 前者：`Object Identificattion`
- 后者：`Indirect Entity Ambiguity`

例子：

- `What's the name of it?` -> `Object Identificattion`
- `Who designed it?` -> `Indirect Entity Ambiguity`

### 6.3 `Description` vs `Indirect Entity Ambiguity`

问自己：

> Query 的模糊性主要来自描述短语，还是来自未解析实体的后续追问？

- 描述短语主导：`Description`
- 后续追问主导：`Indirect Entity Ambiguity`

例子：

- `river in the image with a building across from it` -> `Description`
- `Where is it located?` -> `Indirect Entity Ambiguity`

## 7. 保守标注原则

- 如果 `Text Query` 已经足够支持检索，优先标 `No`
- 不要因为 Query 写得不漂亮、太短、太泛，就直接标 `Yes`
- 不要使用外部世界知识去补足图像信息
- 只根据 `Original Question` 和 `Text Query` 判断

## 8. 推荐备注写法

如果需要写备注，请尽量短而具体。推荐只写“模糊来源”。

例如：

- `pronoun without antecedent`
- `requires image-based identification`
- `descriptive phrase does not uniquely identify entity`

不要写成长解释，不要写推理链。

## 9. 一个简化决策树

可以按下面顺序快速判断：

1. 不看图，检索器是否已经知道该搜谁？
2. 如果知道，标 `No`
3. 如果不知道，这个 Query 是否本质上是在问“这是什么/叫什么”？
4. 如果是，标 `Object Identificattion`
5. 如果不是，它是否是在问这个未解析实体的进一步信息？
6. 如果是，标 `Indirect Entity Ambiguity`
7. 如果不是，它是否通过描述语模糊地指向某个对象？
8. 如果是，标 `Description`
9. 如果以上都不合适，再考虑 `No Object Involved`

## 10. 输出标签

请严格使用以下标签名：

- `Object Identificattion`
- `Description`
- `Indirect Entity Ambiguity`
- `No Object Involved`

注意：

- `Object Identificattion` 的拼写必须保持和系统一致
- 不要自行改写标签名
