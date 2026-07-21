
= Section 2: Byte-Pair Tokenizer
== 2.1 The Unicode Standard
=== (a)
`>>> chr(0)
'\x00'`

=== (b)
The printed representation returns an empty line, while the character's string representation returns 
`>>> chr(0)
'\x00'`

=== (c)
`chr(0)` encodes a `NULL` byte, so nothing is printed in its position when it occurs in printed text. 


== 2.2 Unicode Encodings
=== (a)
UTF-8 encoded bytes are shorter per character; UTF-32 can have as many as 8 bytes per Unicode character. The byte vocabulary in both UTF-32 and UTF-16 is also much larger than the manageable 256. 

=== (b)
Passing 

`>>> aigue_encoded ="é".encode("utf-8")`

to the incorrect function yields a UnicodeDecode error. This decode function assumes that one byte corresponds to a Unicode character, which is not always true. 

=== (c) 
The bytestring `b'\xe3\x81'` doesn't correspond to any Unicode character; it is a subset of `b'\xe3\x81\x93'` which corresponds to 'こ'

==== Merging algorithm
1. For each word, get word splits, for example the becomes `[b't', b'h', b'e']`. Store these in a word split dict with the word as the key and the array as the value. The counts of each word is already stored in the Counter object from before. 
2. For each loop (to add a word to the vocabulary): 
- reinitialize pair counter
- count pairs across all current word splits 
- pick best pair by lexicographic order 
- record merge and add to vocabulary 
- apply merge everywhere (for each word, recompute pairs. Look at whether any pair exists in the merges recorded; if so, replace the pair in the word splits with a merged version)