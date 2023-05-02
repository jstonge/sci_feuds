Here we propose the use LLMs to resolve feuds at a level never seen before. We prompt GPT3.5, a model trained by Open-AI, to parse articles section by section, taking care to detect and score opposing sentiments in sentences. For example, here is an excerpt from the \textsc{Group-Selection v. Kin Selection} feud by Braestrup (1963):

```zsh
  It may be argued - and this is the line 
  taken by W.-E. - both in the paper 
  mentioned and in the book under review 
  - that Lack’s view rests on a mistaken 
  notion of natural selection. The process 
  of selection works, not only between 
  individuals, but also between groups and
  between species, thus promoting characters 
  which are to the benefit of the group, even 
  in the face of contrary individual selection 
  which may be kept in check by special 
  devices. The existence of such evolutionary 
  mechanisms is not only called for by a great 
  many otherwise unaccountable facts, but is 
  also in  accordance with genetic 
  evolutionary theory (see e.g. WRIGHT, 1945).
```

In this section Braestrup opposes W.-E. (Wynne-Edwards) to (D.L.) Lack, taking position in favor of W.-E.. This is a difficult section to evaluate, as the model needs to know who is Lack and that W.-E. stands for a family name. The sections contain both mentions and a formal citation, which for our purposes we both want to consider as positive or negative interactions between the three authors. Here is GPT-3.5 representation of this section:

```zsh

    Sentence: It may be argued - and this is 
    the line  taken by W.-E. - both in the 
    paper mentioned and in the book 
    under review - that Lack’s  view rests 
    on a mistaken notion of natural selection.
    • Score: -0.3 (slightly negative)
    • Mentions/References: W.-E. (Wynne-Edwards) 
    with a score of +0.1 (slightly positive), 
    Lack with a score of -0.4 (negative)    

    Sentence: The process of selection works, 
    [...] special devices.
    • Score: +0.6 (positive)
    • Mentions/References: None

    Sentence: The existence of such 
    evolutionary mechanisms is not 
    only called for by a great many
    otherwise unaccountable facts, but 
    is also in accordance with genetic 
    evolutionary theory (see e.g. WRIGHT,
    1945).
    • Score: +0.4 (positive)
    • Mentions/References: WRIGHT (1945) 
    with a score of +0.4 (positive)

```

Based on this section alone, I would tend to agree with the above scores. Braestrup seems to be criticizing Lack's view in a negative way, siding with Wynne-Edwards and Wright that group selection can exist and is coherent with genetic evolutionary theory. The last two sentences seem to be especially positive. Note that if gpt3.5 is consistent in his ability to score, it means that we can not only have a fine-grained view of formal citations in a paper, but also all the relevant mentions to a debate. It is worth noting that gpt3.5 very easily parsed sentences from this section, which contains punctuations that could prove very challenging to traditionally NLP models. This is an important leap from anything that has come before.
