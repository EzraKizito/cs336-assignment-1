
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