# 📘 Machine Learning: Day 1 Notes

> Only what we covered today. Each concept has a **definition**, a **simple explanation**, an **example from today's class**, and the **code** where we used it.

---

## 📑 Contents

**Part A: The basics**
1. Machine Learning
2. Dataset and Data Point
3. Feature and Target
4. Model
5. Linear Regression
6. Weight (w)
7. Bias (b)
8. Parameters
9. Prediction

**Part B: Where to trust a model**

10. Interpolation
11. Extrapolation
12. Noise
13. Outlier

**Part C: Measuring mistakes**

14. Error (Residual)
15. Why Adding Errors Fails
16. Absolute Value
17. MAE (Mean Absolute Error)
18. Loss Function
19. MSE (Mean Squared Error)

**Part D: Finding the best line**

20. Finding w and b by Hand
21. Averaging Opinions
22. Loss Landscape (The Valley)
23. Gradient
24. Gradient Descent
25. Learning Rate
26. Convergence and Divergence
27. Training

**Part E: Probability basics**

28. Probability
29. Complement
30. Independent Events
31. The "At Least One" Trick

**Part F: Python used today**

**One-page summary**

---

## 📊 Today's Datasets

**☕ Chai stall:** temperature → cups sold

| Temperature (°C) | 20 | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|
| Cups sold | 102 | 84 | 71 | 53 | 40 |

**🏠 Flat rents:** size (hundreds of sq ft) → rent (₹ thousands)

| Size | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| Rent | 10 | 12 | 17 | 20 | 26 |

**📚 Students:** hours studied → marks

| Hours | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Marks | 41 | 49 | 60 | 69 | 81 |

---

# Part A: The Basics

## 1. Machine Learning

> **Definition:** Machine learning is teaching a computer to **find a rule from examples**, instead of a human writing the rule by hand.

**In simple words:** We show the computer past data. It figures out the pattern and uses it to guess new answers.

**Example:** The chai stall owner's notebook shows that hotter days sell fewer cups. A computer can learn this pattern and predict sales for tomorrow's temperature.

**Every ML model follows 3 steps:**
1. **Guess** using a rule with knobs
2. **Measure** how wrong the guess is
3. **Adjust** the knobs to be less wrong, and repeat

---

## 2. Dataset and Data Point

> **Definition:** A **dataset** is the full collection of examples. A **data point** (or sample, or row) is one single example.

**Example:**
- Dataset: all 5 students
- Data point: student 1 → (1 hour, 41 marks)

```python
hours = [1, 2, 3, 4, 5]
marks = [41, 49, 60, 69, 81]
```

---

## 3. Feature and Target

> **Definition:**
> - **Feature** (input, X): what we **already know**.
> - **Target** (output, y): what we **want to predict**.

**Example:**

| Dataset | Feature | Target |
|---|---|---|
| Chai stall | Temperature | Cups sold |
| Flat rents | Size | Rent |
| Students | Hours studied | Marks |

**Remember:** In any ML problem, first ask: *what is my feature, and what is my target?*

---

## 4. Model

> **Definition:** A **model** is a **rule** that takes the feature as input and gives a prediction of the target as output.

**In simple words:** A model is a machine. Hours go in, a guess of marks comes out.

**Example:** `marks = 10 × hours + 30` is a model.

```python
def predict(h, w, b):
    return w * h + b

print(predict(3, 10, 30))   # 60 marks for 3 hours of study
```

---

## 5. Linear Regression

> **Definition:** **Linear regression** is a model that predicts a **number** using a **straight-line rule**:
>
> **prediction = w × feature + b**

- **Linear** = straight line
- **Regression** = predicting a number (not a category like yes/no)

**Example:** `rent = 2 × size + 1` is a linear regression model.

---

## 6. Weight (w)

> **Definition:** The **weight** (also called **slope**) is **how much the prediction changes when the input goes up by 1**.

**Examples:**

| Model | w | Meaning |
|---|---|---|
| Students | **+10** | Each extra hour of study adds 10 marks |
| Rents | **+2** | Each extra 100 sq ft adds ₹2,000 rent |
| Chai stall | **−3.1** | Each extra degree means 3.1 fewer cups |

**The sign of w tells the direction:**
- **Positive:** input up → output up
- **Negative:** input up → output down
- **Zero:** input has no effect

**Changing w tilts the line** (steeper or flatter).

---

## 7. Bias (b)

> **Definition:** The **bias** (also called **intercept**) is the **starting number**: the prediction when the input is **0**.

**Examples:**
- Students, b = 30: a student who studies **0 hours** would get about 30 marks.
- Rents, b = 1: a base charge of ₹1,000, like maintenance.
- Chai stall, b = 163: the cups the line would predict at 0°C.

**Changing b moves the whole line up or down.** Every prediction shifts by the same amount.

📻 **Analogy:** b is like a radio tuning knob. Turn it until the "static" (the error) is lowest.

---

## 8. Parameters

> **Definition:** **Parameters** are the knobs of the model that are **learned from data**. In linear regression, the parameters are **w and b**.

**Remember the words:**
- We **predict** the target (marks, rent).
- We **learn** (find) the parameters (w, b).

---

## 9. Prediction

> **Definition:** A **prediction** is the model's **guess** for the target.

**Example:** With `rent = 2 × size + 1`, the prediction for flat A (size 4) is 2 × 4 + 1 = **9**, which means ₹9,000.

---

# Part B: Where to Trust a Model

## 10. Interpolation

> **Definition:** **Interpolation** is predicting for an input that lies **inside the range** of data the model learned from.

**In simple words:** filling in a gap **between** examples you have already seen. Usually **safe**.

**Example from today:** The chai data covers 20°C to 40°C. Predicting sales at **32°C** is interpolation. Your first answer was **64 cups**, and the best line gives **63.8**, nearly the same.

**Another example:** The students studied 1 to 5 hours. Predicting for **3.5 hours** → 65 marks. ✅ Safe.

```python
print(predict(3.5, 10, 30))   # 65.0 → inside 1 to 5 hours, trustworthy
```

---

## 11. Extrapolation

> **Definition:** **Extrapolation** is predicting for an input that lies **outside the range** of data the model learned from.

**In simple words:** guessing about a situation the model has **never seen**. Often **dangerous**.

**Examples from today:**

| Situation | Model says | Problem |
|---|---|---|
| Chai stall at **60°C** | 163 − 3.1 × 60 = **−23 cups** | Can't sell negative chai! |
| Student studies **15 hours** | 10 × 15 + 30 = **180 marks** | The exam is out of 100! |
| A **5000 sq ft** bungalow | 2 × 50 + 1 = **₹1,01,000** | Luxury homes follow different pricing |
| A flat in a **different city** | Any answer | The model never saw that city |

```python
print(predict(15, 10, 30))   # 180 → outside the range, impossible
```

🧑‍🎓 **Analogy:** a student who studied only chapters 1 to 5 gets a question from chapter 10.

⚠️ **Python never warns you about extrapolation.** It happily prints an answer. Knowing when not to trust it is **your** job.

---

## 12. Noise

> **Definition:** **Noise** is the small, natural, random wobble in real data that the model can't explain.

**Example:** By the rule `rent = 2 × size + 1`, flat B should rent for 13, but it actually rents for 12. Maybe the owner was in a hurry. That small difference is noise.

**Why it matters:** Because of noise, two different pairs of points give **slightly different** values of w and b (see Section 20).

---

## 13. Outlier

> **Definition:** An **outlier** is a data point that is **very different** from the rest and doesn't follow the pattern.

**Example:** One day at 30°C, the chai stall sells **300 cups** because a wedding party walked by.

**Outlier vs extrapolation (don't mix them up):**

| | Outlier | Extrapolation |
|---|---|---|
| Where? | A strange point **inside** your data | A question **outside** your data's range |
| Example | 300 cups on a 30°C day | Predicting sales at 60°C |

---

# Part C: Measuring Mistakes

## 14. Error (Residual)

> **Definition:** The **error** (also called **residual**) is the difference between the real value and the model's guess:
>
> **Error = real value − predicted value**

| Sign | Meaning |
|---|---|
| **Positive** | The model guessed **too low** |
| **Negative** | The model guessed **too high** |
| **Zero** | Perfect guess |

**Example:** Your friend is 30 years old.
- You guess 25 → error = 30 − 25 = **+5** (too low)
- You guess 35 → error = 30 − 35 = **−5** (too high)

---

## 15. Why Adding Errors Fails

> **Key idea:** You **cannot** judge a model by simply adding up its errors, because **positive and negative errors cancel out**.

**Example:** The lazy friend at the chai stall guesses **70 cups every day**:

| Real | Guess | Error |
|---|---|---|
| 102 | 70 | +32 |
| 84 | 70 | +14 |
| 71 | 70 | +1 |
| 53 | 70 | −17 |
| 40 | 70 | −30 |
| | **Total** | **0** |

A total of 0 looks perfect, but he was off by **32 cups** on one day!

🎯 **Analogy:** One arrow lands 30 cm left (−30), another 30 cm right (+30). The total is 0, but **both arrows missed**.

**Your fix (you invented this!):** ignore the sign before adding.

```python
total = 0
for h, m in zip(hours, marks):
    error = m - predict(h, 0, 60)     # lazy teacher: always guesses 60
    total = total + error
print(total)    # 0 → misleading!
```

---

## 16. Absolute Value

> **Definition:** The **absolute value** of a number is its **size, ignoring the sign**. It's written with two straight bars.

**Examples:** |−17| = 17  |32| = 32  |0| = 0

```python
print(abs(-17))   # 17
```

---

## 17. MAE (Mean Absolute Error)

> **Definition:** **MAE** is the **average of the absolute errors**:
>
> **MAE = (sum of |errors|) ÷ (number of data points)**

**In simple words:** the average miss. **Lower is better.**

**Example 1: the chai stall**

| Rule | Misses | MAE |
|---|---|---|
| Lazy friend (always 70) | 32, 14, 1, 17, 30 | 94 ÷ 5 = **18.8** |
| Your first rule (174 − 3.6 × temp) | 0, 0, 5, 5, 10 | 20 ÷ 5 = **4** |
| Best line (163 − 3.1 × temp) | 1, 1.5, 1, 1.5, 1 | 6 ÷ 5 = **1.2** 🏆 |

**Example 2: flat rents**

| Rule | MAE |
|---|---|
| Lazy broker (always 17) | **4.8** (misses by ₹4,800 per flat) |
| 2 × size + 1 | **0.8** (misses by ₹800 per flat) 🏆 |

```python
def mae(w, b):
    total = 0
    for h, m in zip(hours, marks):
        total = total + abs(m - predict(h, w, b))
    return total / len(hours)

print(mae(0, 60))    # 12.0  lazy teacher
print(mae(10, 30))   # 0.8   our rule
```

**Remember:** MAE is in the **same units** as the target (cups, ₹, marks), so it's easy to explain to anyone.

---

## 18. Loss Function

> **Definition:** A **loss function** (or **cost function**) is any formula that gives a **score of how wrong** a model is.

**Example:** MAE and MSE are both loss functions.

**The golden rule:** **lower loss = better model.**

---

## 19. MSE (Mean Squared Error)

> **Definition:** **MSE** is the **average of the squared errors**:
>
> **MSE = (sum of errors²) ÷ (number of data points)**

**Why square?**

| Error | Absolute (MAE) | Squared (MSE) |
|---|---|---|
| −3 | 3 | 9 |
| −1 | 1 | 1 |
| 1 | 1 | 1 |
| 3 | 3 | 9 |

1. Squaring **also removes the sign** (−3 × −3 = 9), so errors can't cancel.
2. It **punishes big mistakes much more** (a miss of 3 costs 9, not 3).
3. It makes a **smooth U-shaped valley**, which gradient descent needs (Section 22).

**Example:** Misses of 1, −1, 0, −1, 1 → MSE = (1 + 1 + 0 + 1 + 1) ÷ 5 = **0.8**

```python
def mse(w, b):
    total = 0
    for h, m in zip(hours, marks):
        total = total + (m - predict(h, w, b)) ** 2
    return total / len(hours)
```

**MAE vs MSE:**

| | MAE | MSE |
|---|---|---|
| Removes sign by | Absolute value | Squaring |
| Valley shape | Sharp **V** | Smooth **U** |
| Big mistakes | Counted normally | Punished heavily |
| Best used for | Explaining errors to people | Training with gradient descent |

---

# Part D: Finding the Best Line

## 20. Finding w and b by Hand

> **Method:** Pick two data points.
> - **w = (change in output) ÷ (change in input)**
> - **b = output − w × input** (using either point)

**Example (students 1 and 5):**
- w = (81 − 41) ÷ (5 − 1) = 40 ÷ 4 = **10**
- b = 41 − 10 × 1 = **31**

**Same method, students 2 and 4:**
- w = (69 − 49) ÷ (4 − 2) = 20 ÷ 2 = **10**
- b = 49 − 10 × 2 = **29**

**The problem:** different pairs give different answers (31 vs 29), because of **noise**.

```python
h1, m1 = 1, 41
h2, m2 = 5, 81
w = (m2 - m1) / (h2 - h1)
b = m1 - w * h1
print(w, b)   # 10.0  31.0
```

⚠️ **Brackets matter:** `(81 - 41) / (5 - 1)` = 10, but `81 - 41 / 5 - 1` = 71.8.

---

## 21. Averaging Opinions

> **Key idea:** One pair of points can mislead you. When different data points disagree, **average their opinions** for a more reliable answer.

**Example 1 (students):** b = (31 + 29) ÷ 2 = **30**, so `marks = 10 × hours + 30`

**Example 2 (chai stall):** Your first slope, 3.6, came from just one pair of days. All neighbouring pairs gave 3.6, 2.6, 3.6, 2.6. Average = 12.4 ÷ 4 = **3.1**, which is the best slope.

**Finding b from every point ("walking back" to 0°C):** with w = −3.1, each day gives b = cups + 3.1 × temperature:

| Day | Calculation | b |
|---|---|---|
| 20°C | 102 + 62 | 164 |
| 25°C | 84 + 77.5 | 161.5 |
| 30°C | 71 + 93 | 164 |
| 35°C | 53 + 108.5 | 161.5 |
| 40°C | 40 + 124 | 164 |

Average = 815 ÷ 5 = **163** → best line: `cups = 163 − 3.1 × temperature`

⚠️ **Hidden catch:** averaging only **neighbouring** slopes secretly uses only the **first and last** points, because the middle values cancel:
(102 − 84) + (84 − 71) + (71 − 53) + (53 − 40) = 102 − 40.
That's why computers use a method where **every** point has a say: gradient descent.

---

## 22. Loss Landscape (The Valley)

> **Definition:** The **loss landscape** is the picture of the loss for **every possible setting** of the knobs. For one knob, it looks like a **valley**. The **bottom of the valley** is the best setting.

**Example (rents, w fixed at 2, turning the b knob):**

| b | −2 | −1 | 0 | **1** | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|
| MAE | 3.0 | 2.0 | 1.0 | **0.8** ⬅️ bottom | 1.0 | 2.0 | 3.0 |

The loss goes **down, down, down, then back up**.

**Notice:** the ground is steep far away and **flatter near the bottom**:
- b from −2 to −1: loss drops by 1.0 (steep)
- b from 0 to 1: loss drops by only 0.2 (nearly flat)

```python
for b in range(-2, 5):
    print("b =", b, "-> MAE =", mae(2, b))
```

---

## 23. Gradient

> **Definition:** The **gradient** is the **steepness and direction of the slope** of the loss at your current position.

**In simple words:** what the ground feels like under the blindfolded hiker's feet.

| Gradient | Meaning | What to do |
|---|---|---|
| **Negative** | Downhill to the right | Increase b |
| **Positive** | Downhill to the left | Decrease b |
| **Big** (steep) | Far from the bottom | Take a big step |
| **Small** (gentle) | Close to the bottom | Take a small step |
| **Zero** (flat) | At the bottom | Stop |

**Example (students, w = 10):** steepness at b = 20 is **−20**, at b = 30 is **0**, at b = 40 is **+20**.

**How we measured it in code:** check the loss a tiny bit to the right and left, then compare.

```python
def steepness(b):
    tiny = 0.001
    return (mse(10, b + tiny) - mse(10, b - tiny)) / (2 * tiny)
```

---

## 24. Gradient Descent

> **Definition:** **Gradient descent** is a method for finding the best parameters by **repeatedly taking small steps downhill** on the loss landscape, using the gradient to choose the direction and size of each step.
>
> **new value = old value − learning rate × gradient**

### 🥾 The blindfolded hiker

A hiker is **blindfolded** in a valley and wants to reach the **lowest point**. He can't see, but he **feels the slope** with his feet and steps downhill, again and again.

| Story | ML meaning |
|---|---|
| Where the hiker stands | Current value of b (or w) |
| Height of the ground | Loss (MSE) |
| Bottom of the valley | Best parameters |
| Steepness under his feet | Gradient |
| How bold each step is | Learning rate |

### Why the minus sign?
If the gradient is **negative** (downhill to the right), then *minus a negative* = **plus**, so the value moves right, downhill. The minus sign **always** sends you downhill.

### Why it slows down automatically
> **step size = learning rate × steepness**

Steep ground → big steps. Flat ground near the bottom → tiny steps. Nobody tells the hiker to slow down; it happens on its own.

**Worked example (students, w = 10, start at b = 0, learning rate = 0.1):**

| Step | Steepness | Step taken | New b |
|---|---|---|---|
| 1 | −60 | +6 | 6 |
| 2 | −48 | +4.8 | 10.8 |
| 3 | −38.4 | +3.84 | 14.64 |
| ... | getting flatter | getting smaller | ... |
| end | ≈ 0 | ≈ 0 | **30** ✅ |

```python
b = 0.0
learning_rate = 0.1

for step in range(1, 26):
    s = steepness(b)
    b = b - learning_rate * s        # the gradient descent rule
    print("Step", step, "| b =", round(b, 2))
```

💡 This one rule trains linear models, neural networks, and large AI models. They just have far more knobs.

---

## 25. Learning Rate

> **Definition:** The **learning rate** is a number **you choose** that controls **how big each step** of gradient descent is.

**Example:** learning rate 0.1

| Steepness | Step size (0.1 × steepness) |
|---|---|
| 20 | 2 |
| 2 | 0.2 |
| 0 | 0 → stop |

**Too big vs too small:**

| Learning rate | What happens |
|---|---|
| **Too small** | Right direction, but painfully slow |
| **Just right** | Reaches the bottom smoothly |
| **Too big** | Overshoots, bounces from side to side |
| **Way too big** | Bounces higher each time and blows up |

**Example (rents, start at b = −4, best b = 1, after 30 steps):**

| Learning rate | b after 30 steps | Result |
|---|---|---|
| 0.001 | −3.71 | Too slow, barely moved |
| 0.1 | 0.994 | Just right ✅ |
| 0.9 | 0.994 | Got there, but by bouncing |
| 1.1 | −1185.9 | Exploded ❌ |

**Example from class:** with steps of 3, the computer jumped from b = 163 straight to 166, **skipping** the better spot at 164.

---

## 26. Convergence and Divergence

> **Definitions:**
> - **Convergence:** gradient descent **settles** at the bottom of the valley; the loss stops going down.
> - **Divergence:** the steps get **bigger and bigger**, the loss **blows up**, and the model never settles.

**Example:** learning rate 0.1 → converges to b = 1. Learning rate 1.1 → diverges to b = −1185.9.

---

## 27. Training

> **Definition:** **Training** a model means **finding the parameters (w and b) that give the lowest loss**, usually by running gradient descent on **all the knobs** at once until it converges.

**Example:** Starting from w = 0, b = 0 and knowing nothing, the computer found the students rule **w = 10, b = 30** completely on its own.

```python
def steep_w(w, b):
    tiny = 0.001
    return (mse(w + tiny, b) - mse(w - tiny, b)) / (2 * tiny)

def steep_b(w, b):
    tiny = 0.001
    return (mse(w, b + tiny) - mse(w, b - tiny)) / (2 * tiny)

w, b = 0.0, 0.0
learning_rate = 0.05

for step in range(1, 1001):
    sw = steep_w(w, b)
    sb = steep_b(w, b)
    w = w - learning_rate * sw
    b = b - learning_rate * sb

print(round(w, 2), round(b, 2))   # 10.0  30.0
```

---

# Part E: Probability Basics

## 28. Probability

> **Definition:** **Probability** is the **fraction of times** something happens, out of all the times. It's always between **0** (never) and **1** (always).

**Example:** It rains on 30% of days → **P(rain) = 0.3**. Out of 100 days, about 30 are rainy.

---

## 29. Complement

> **Definition:** The **complement** of an event is the event **not** happening:
>
> **P(not A) = 1 − P(A)**

**Example:** P(dry) = 1 − P(rain) = 1 − 0.3 = **0.7**

---

## 30. Independent Events

> **Definition:** Two events are **independent** if one **doesn't affect** the other. Then the chance of **both** happening is found by **multiplying**:
>
> **P(A and B) = P(A) × P(B)**

**Example: two dry days in a row**
- Imagine 100 two-day periods. Day 1 is dry in about **70** of them.
- Of those 70, day 2 is also dry 70% of the time: 0.7 × 70 = **49**.
- So P(2 dry days) = 0.7 × 0.7 = **0.49**

For **n** dry days in a row: **0.7 × 0.7 × ... (n times) = 0.7ⁿ**

---

## 31. The "At Least One" Trick

> **Key idea:** "At least one" is the **opposite** of "none at all". So:
>
> **P(at least one) = 1 − P(none)**

**Example:** P(at least one rainy day in 10 days)
- P(no rain for 10 days) = 0.7¹⁰ ≈ 0.028
- P(at least one rainy day) = 1 − 0.028 ≈ **0.972** (about 97%)

---

# Part F: Python Used Today

| Code | Meaning | Example |
|---|---|---|
| `# text` | Comment: a note for humans, ignored by Python | `# my data` |
| `x = 5` | Store a value in a named box (variable) | `w = 10` |
| `[a, b, c]` | List: values kept together in order | `hours = [1, 2, 3]` |
| `print(x)` | Show something on screen | `print(w)` |
| `len(list)` | Count items in a list | `len(hours)` → 5 |
| `sum(list)` | Add up all items | `sum([1, 2, 3])` → 6 |
| `abs(x)` | Absolute value | `abs(-7)` → 7 |
| `x ** 2` | x squared | `3 ** 2` → 9 |
| `list[i]` | Item at position i (**counting starts at 0**) | `hours[0]` → 1 |
| `list.append(x)` | Add x to the end of a list | `all_b.append(30)` |
| `def name(...):` | Create a function | `def predict(h, w, b):` |
| `return x` | The answer a function gives back | `return w * h + b` |
| `for x in list:` | Repeat once for each item | `for h in hours:` |
| `zip(a, b)` | Pair two lists item by item | (1, 41), (2, 49), ... |
| `range(a, b)` | Numbers a to b−1 (**stops before b**) | `range(25, 28)` → 25, 26, 27 |
| `round(x, 2)` | Round to 2 decimal places | `round(3.14159, 2)` → 3.14 |

⚠️ **Indentation** (4 spaces at the start of a line) tells Python which lines belong inside a function or loop.

---

# 📌 One-Page Summary

| Concept | One line | Today's example |
|---|---|---|
| **Machine learning** | Computer finds a rule from examples | Chai sales vs temperature |
| **Feature / target** | What we know / what we predict | Hours → marks |
| **Model** | The rule | `marks = 10 × hours + 30` |
| **Linear regression** | Straight-line rule that predicts a number | `rent = 2 × size + 1` |
| **w (weight)** | Change in output per 1 unit of input; tilts the line | +10 marks per hour |
| **b (bias)** | Starting value at input 0; shifts the line | 30 marks for 0 hours |
| **Parameters** | Knobs the model learns | w and b |
| **Interpolation** | Predicting inside the data range ✅ | 32°C → 64 cups |
| **Extrapolation** | Predicting outside the data range ⚠️ | 60°C → −23 cups |
| **Noise** | Small random wobble | Rent 12 instead of 13 |
| **Outlier** | Strange point inside the data | 300 cups on a 30°C day |
| **Error** | Real − predicted | 30 − 25 = +5 |
| **Why not add errors** | + and − cancel | Lazy friend's total = 0 |
| **MAE** | Average absolute miss | 18.8 vs 4 vs 1.2 |
| **MSE** | Average squared miss; smooth U valley | 0.8 for our rule |
| **Loss function** | Score of how wrong; lower is better | MAE, MSE |
| **Averaging** | Combine opinions of all points | b = (31 + 29) ÷ 2 = 30 |
| **Loss landscape** | Valley of loss for each knob setting | Bottom at b = 1 |
| **Gradient** | Steepness under your feet | −60 at b = 0 |
| **Gradient descent** | new = old − learning rate × gradient | Hiker walks 0 → 30 |
| **Learning rate** | Size of each step | 0.1 just right, 1.1 explodes |
| **Convergence** | Settles at the bottom | b → 1 |
| **Divergence** | Blows up | b → −1185.9 |
| **Training** | Finding w, b with the lowest loss | Computer found w = 10, b = 30 |
| **Probability** | Fraction of times something happens | P(rain) = 0.3 |
| **Complement** | P(not A) = 1 − P(A) | P(dry) = 0.7 |
| **Independent events** | Multiply | 0.7 × 0.7 = 0.49 |
| **At least one** | 1 − P(none) | 1 − 0.7¹⁰ ≈ 0.97 |

> 🧭 **Today in one sentence:**
> *A model is a rule with knobs (w and b); a loss function (MAE or MSE) scores how wrong it is; gradient descent turns the knobs downhill until the loss stops shrinking, and the model can be trusted only inside the range of data it learned from.*