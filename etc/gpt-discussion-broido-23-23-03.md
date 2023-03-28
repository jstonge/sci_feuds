

Step 1, I've asked chatgpt to score the valence of each sentence, as well as the reference in those sections, by section (see prompts below):

```shell
In Section IV we validate that this is indeed the case
by applying these estimators to a wide range of synthetic
sequences sampled from a variety of regularly varying
distributions, as well as to degree sequences in paradig-
matic network models—the configuration model, prefer-
ential attachment, and random hyperbolic graphs. In all
the considered cases, all the considered estimators con-
verge as expected, in contrast to the PLFit algorithm
from [18, 19], which behaves abnormally whenever the
regularly varying distribution has a non-trivial slowly
varying function `(k). Remarkably, one of such cases
is preferential attachment. The degree distribution in
preferential attachment networks was proven to be scale-
free [39], but the hypothesis testing methodology in [19]
rejects this statement for high percentages of simulated
preferential attachment networks. The reasons behind
this abnormality are also documented in Section IV.
```

Note that ref [19] is Broido And Clauset's paper about ``Scales-free networks are rare''. Gpt response:

```
Sure! Here's a breakdown of each sentence's valence, arousal, and dominance score, along with the labeled equation, and sentiment towards references:
Sentence: Validation of the estimators in Section IV Valence: 
Positive Arousal: 
Low Dominance:  Low 
Equation: None 
References: None

Sentence: Application of the estimators to synthetic sequences and network models 
Valence: Neutral 
Arousal: Low 
Dominance: Low 
Equation: None 
References: None

Sentence: Convergence of all estimators in contrast to PLFit algorithm from [18, 19] 
Valence: Positive 
Arousal: Low 
Dominance: Low 
Equation: None 
References: [18, 19] - Negative

Sentence: PLFit algorithm behaves abnormally whenever the regularly varying distribution has a non-trivial slowly varying function 
Valence: Negative 
Arousal: Low 
Dominance: Low 
Equation: None 
References: None

Sentence: Preferential attachment networks proven to be scale-free 
Valence: Positive 
Arousal: Low 
Dominance: Low 
Equation: None 
References: [39] - Positive

Sentence: Hypothesis testing methodology in [19] rejects the scale-free statement for high percentages of simulated networks
Valence: Negative 
Arousal: Low 
Dominance: Low 
Equation: None 
References: [19] - Negative

Sentence: Reasons behind abnormality of PLFit algorithm documented in Section IV 
Valence: Neutral 
Arousal: Low 
Dominance: Low 
Equation: None 
References: None
```

This is not bad! I chose a paper that was strongly opposed to the claim that scales free networks are rare. Gpt was somehow able to dissociate the valence of reference and the general tone of the sentence.

Step 2, and this is the crazy step, we try to validate gpt scores by asking the people involved in the controversy if they think this is about right. I am super curious about the relative perception of the same event (a reference from person a against b). But we could sell the project in other ways.

p.s. prompts I use to eventually get to the above response:

 - Can you give me a breakdown of each sentence's main topic, sentiment label, and sentiment score? And can you limit the number of main topics to 3?
 - Can you limit the topics to 1 or 2 words?
 - Again, but try to identify equation and label them as well. You do not need to put sentiment score and topic to equations.
 - But you still need to give me a breakdown of each sentence's main topic, sentiment label, and sentiment score if they are not equation.
 - Can you do the last section again but keep track of the sentence your describing as you go along
 - Can you do that again, but instead of having simply positive-negative sentiments, I want the Valence (positive/negativity), Arousal (excitement/relaxation) and dominance score of each sentence.
 - Can you do it, but everytime you see a reference I want you to add if the sentiment towards the reference as well.
 - Can you do that again, but consider each mention reference on its own, when they are separated in the sentence.
