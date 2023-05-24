Output contains the elements produced from the following locations:
 * 📂 [group_selection_grobid](group_selection_grobid/):  `data/raw_pdfs` parsed with grobid
 * 📂 [group_selection_mmda](group_selection_mmda/): `data/raw_pdfs` parsed with `allenai/mmda`
 * 📂 [group_selection_grobid](group_selection_grobid/): grobid json files parsed with spacy
 * 📂 [cca](cca/): files coming from `allenai/multicite` used to fine-tuned LLMs on context analysis
 * 📂 [stance_detection](stance_detection/): files coming from `DominikBeese/DidAIGetMoreNegativeRecently` used to fine-tuned LLMS on stance detection

 Also note that:
  * `My-Predictions2.json`: This is simply `groupSel_feud_with_tag.parqet` but with stance predicted with `DominikBeese/DidAIGetMoreNegativeRecently`' model. The script to run stance detection is on `Rhema` b/c I needed a GPU.
