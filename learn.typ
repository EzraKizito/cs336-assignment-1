= Tokenizers

- Word-level tokenization leads to out-of-vocabulary (OOV) issues, where a model can break down due to seeing a word that wasn't in its training set. Subword tokenization alleviates this because if a model sees a word it doesn't recognize, it can break it down into smaller pieces that were in its training set.
- Maybe convert to bytes before converting to pretokens? *No*, this doesn't work. 
