
```zsh
given: 2 The evaluation is then performed on sentences with "agreement attractors" in which at there is at least one noun between 
the verb and its subject, and all of the nouns between the verb and subject are of the opposite number from the subject. Gulordava 
et al. (2018) also start with existing sentences. However, in order to control for the possibillity of the model learning to rely 
on "semantic" selectional-preferences cues rather than syntactic ones, they replace each content word with random words from the 
same part-ofspeech and inflection. This results in "coloreless green ideas" nonce sentences. The evaluation is then performed similarly 
to the LM setup of <cite>Linzen et al. (2016)</cite> : the sentence is fed into a pretraiend LSTM LM up to the focus verb, and the model 
is considered correct if the probability assigned to the correct inflection of the original verb form given the prefix is larger than 
that assigned to the incorrect inflection.

target: The evaluation is then performed similarly to the LM setup of <cite>Linzen et al. (2016)</cite> : the sentence is fed into a 
pretraiend LSTM LM up to the focus verb, and the model is considered correct if the probability assigned to the correct inflection of 
the original verb form given the prefix is larger than that assigned to the incorrect inflection.

reply: The evaluation is then performed similarly to the LM setup of <cite>Linzen et al. (2016)</cite> : the sentence is fed into a 
pretraiend LSTM LM up to the focus verb, and the model is considered correct if the probability assigned to the correct inflection 
of the original verb form given the prefix is larger than that assigned to the incorrect inflection.
```

```zsh
given: They built training datum of all noun words in WordNet whose size is larger than 7GB, but 
evaluated their method on a small number of nouns of lexical sample task of SENSEVAL-2 as [8] and 
<cite>[9]</cite> . ---------------------------------- **WORD SENSE DISAMBIGUATION BY RELATIVE SELECTION** 
Our method disambiguates senses of a target word in a sentence by selecting only a relative among the relatives 
of the target word that most probably occurs in the sentence. A flowchart of our method is presented in Figure 1 
with an example 3 : 1) Given a new sentence including a target word, a set of relatives of the target word is 
created by looking up in WordNet.

target: They built training datum of all noun words in WordNet whose size is larger than 7GB, but evaluated their 
method on a small number of nouns of lexical sample task of SENSEVAL-2 as [8] and <cite>[9]</cite> .

reply: They built training datum of all noun words in WordNet whose size is larger than 7GB, but evaluated their 
method on a small number of nouns of lexical sample task of SENSEVAL-2 as [8] and <cite>[9]</cite> .
```
```zsh
given: **INTRODUCTION** Recent work on neural constituency parsing<cite> (Dyer et al., 2016</cite>; Choe and Charniak, 2016) has found multiple 
cases where generative scoring models for which inference is complex outperform base models for which inference is simpler. 
Let A be a parser that we want to parse with (here one of the generative models),and let B be a base parser that we use to propose candidate parses
which are then scored by the less-tractable parser A. We denote this cross-scoring setup by B → A. The papers 
above repeatedly saw that the', 'cross-scoring setup B → A under which their generative models were applied outperformed the 
standard singleparser setup B → B. We term', 'this a cross-scoring gain. This paper asks two questions. First, why do 
recent discriminative-to-generative cross-scoring se- * Equal contribution.]

target: Recent work on neural constituency parsing<cite> (Dyer et al.,', '2016</cite>; Choe and Charniak, 2016) has
found multiple cases where generative scoring models for which inference is complex outperform base models for which inference is simpler.

reply: Recent work on neural constituency parsing<cite> (Dyer et al.,', '2016</cite>; Choe and Charniak, 2016) 
has found multiple cases where generative scoring models for which inference is complex outperform base models for which inference is simpler.
```

```zsh
given: We then evaluated the impact of the 1D convolution layers in our <cite>transformer</cite> model 
by removing all the convolution layers. This corresponds to a vanilla <cite>transformer</cite> as 
in <cite>[8]</cite> . We still added the positional encoding feature to the inputs since the 
sequential information from the convolution layers is no longer available. We also have to insert 
another layer normalization to the output of the <cite>transformer</cite> before the output linear
layer to stabilize the training. Otherwise, the training diverges quickly after one or two epochs in our experiments.

target: We then evaluated the impact of the 1D convolution layers in our <cite>transformer</cite> 
model by removing all the convolution layers.

reply: We then evaluated the impact of the 1D convolution layers in our <cite>transformer</cite> 
model by removing all the convolution layers.
```
